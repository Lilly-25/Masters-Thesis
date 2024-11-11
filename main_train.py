from src.data_preprocessing_tabular import data_prep
import joblib
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from src.model import mlp_kpi3d
from src.training import mlp_kpi3d_trainer as mlp_kpi3d_trainer, Y1LossRegularisation as loss_reg_y1, Y2LossRegularisation as loss_reg_y2, MSELoss as mseloss
from sklearn.model_selection import KFold
import wandb
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from src.utils import artifact_deletion
import os
from dotenv import load_dotenv
import random


##Data Preprocessing

test_size=50
x_normalized, y1, y2, x_mean, x_stddev, df_test_inputs, df_test_y1_targets, max_mgrenz = data_prep(test_size)

# Store scalers and split test dataset locally to disk
joblib.dump(x_mean, './Intermediate/x_mean.pkl')
joblib.dump(x_stddev, './Intermediate/x_stddev.pkl')
joblib.dump(max_mgrenz, './Intermediate/max_mgrenz.pkl')
df_test_inputs.to_pickle('./data/df_test_inputs.pkl')
df_test_y1_targets.to_pickle('./data/df_test_y1_targets.pkl')

##Model Training

load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
wandb.login(key=wandb_api_key)
group_no=random.randint(100, 999)

#hyperparameters
p_y1=0.35  # dropout rate shared and y1
p_y2=0.2  # dropout rate y2
hidden_sizes = 128  #256 max..else OOM...to be between input features and min(output features)
batch_size = 72     ##try with 72 also
epochs = 10
lr = 0.075#0.075
lambda1_y1=0.5 # Smoothening
lambda2_y1=0 # Decreasing
lambda_y2=3.75 # MM 0
y2_low_mm_threshold=20
y2_low_nn_threshold=20
y2_initial_boundary_threshold = 5 # Threshold to consider MM Max Mgrenz
w= 0.05# ALWAYS ENSURE TO KEEP wBETWEEN 0 AND 1
gamma=0.9
splits=5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kf = KFold(n_splits=splits, shuffle=True, random_state=42)##Goto store splits and load from there to be sure of reproducibiiity
# Store the cross-validation splits
splits_file = './Intermediate/cross_val_splits.npy'
# splits_indices=kf.split(x_normalized)
# splits_indices = [(train_index, val_index) for train_index, val_index in kf.split(x_normalized)]
# np.save(splits_file, np.array(splits_indices, dtype=object), allow_pickle=True)

# Load the cross-validation splits
splits_indices = np.load(splits_file, allow_pickle=True)

# Lists to store results
cross_train_losses_y1, cross_train_losses_y2 = [], []
cross_val_losses_y1, cross_val_losses_y2 = [], []
cross_train_score_y1, cross_train_score_y2 = [], []
cross_val_score_y1, cross_val_score_y2  = [], []

best_combined_loss = 300
best_y1_score = 30
best_y2_score = 30
best_model = None
best_fold = -1

for fold, (train_index, val_index) in enumerate(splits_indices, 1):
    
    # Initialize wandb with a group to aggregate all runs of each cross-validation fold
    wandb.init(project="EM", config={
        "learning_rate": lr,
        "architecture": "MLP",
        "hidden sizes": hidden_sizes,
        "epochs": epochs,
        "y1 lambda smoothening loss regularizer": lambda1_y1,
        "y1 lambda decreasing curve loss regularizer": lambda2_y1,
        "y2 lambda regularizer": lambda_y2,
        "y2 initial boundary threshold": y2_initial_boundary_threshold,
        "y2 low nn threshold": y2_low_nn_threshold,
        "y2 low mm threshold": y2_low_mm_threshold,
        "y1 loss weightage": w,
        "y2 loss weightage": 1 - w,
        "Learning Rate Scheduler": "exponential",
        'LR Scheduler Gamma': gamma,
        "Dropout Rate Shared": p_y1,
        "Dropout Rate Y2": p_y2,
        "Notes": "Removed curve thresholld for y1 envelope stupidity",
        }, group=str(group_no), name=f"Fold {fold}")
    
    
    # Train-Validation data as per cross val splits
    x_train, x_val = x_normalized[train_index], x_normalized[val_index]
    y1_train, y1_val = y1[train_index], y1[val_index]
    y2_train, y2_val = y2[train_index], y2[val_index]

    train_dataset = TensorDataset(torch.FloatTensor(x_train).to(device), torch.FloatTensor(y1_train).to(device), torch.FloatTensor(y2_train).to(device))
    val_dataset = TensorDataset(torch.FloatTensor(x_val).to(device), torch.FloatTensor(y1_val).to(device), torch.FloatTensor(y2_val).to(device))
    
    y2_shape = (batch_size, y2_train.shape[1], y2_train.shape[2])  
    y1_shape = (batch_size, y1_train.shape[1]) 
    input_size = x_train.shape[1]
    #Shouldnt the final layers in model be multipled with batch size? -TODO
    
    model = mlp_kpi3d(input_size, hidden_sizes, y1_shape, y2_shape,p_y1, p_y2)
    
    loss1 = loss_reg_y1(lambda1_y1, lambda2_y1)
    loss2 = loss_reg_y2(lambda_y2, y2_initial_boundary_threshold, y2_low_nn_threshold, y2_low_mm_threshold)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    
    lrscheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    trainer = mlp_kpi3d_trainer(model, train_dataset, val_dataset, batch_size, loss1, loss2, optimizer, lrscheduler, w, 1-w, device)
    
    trainer.train(epochs)

    cross_train_losses_y1.append(trainer.train_loss_history_y1)
    cross_train_losses_y2.append(trainer.train_loss_history_y2)
    cross_val_losses_y1.append(trainer.val_loss_history_y1)
    cross_val_losses_y2.append(trainer.val_loss_history_y2)
    cross_train_score_y1.append(trainer.train_score_history_y1)
    cross_train_score_y2.append(trainer.train_score_history_y2)
    cross_val_score_y1.append(trainer.val_score_history_y1)
    cross_val_score_y2.append(trainer.val_score_history_y2)
    
    # Combined loss of y1 and y2 per fold
    combined_loss = trainer.val_loss_history_y1[-1] + trainer.val_loss_history_y2[-1]
    
    # Combined loss of y1 and y2 to be the least and r2 score to be as close to 1 as possible
    if trainer.val_score_history_y1[-1] < best_y1_score and trainer.val_score_history_y2[-1] < best_y2_score:
        best_y1_score = trainer.val_score_history_y1[-1]
        best_y2_score = trainer.val_score_history_y2[-1]
        print(f"Fold : {fold} Best y1 score: {best_y1_score} Best y2 score: {best_y2_score}")
        best_model = model
        best_fold = fold

    wandb.finish()
    
if best_model is not None:
    model.save('./Intermediate/model.pth')
    print(f"Best model saved from fold {best_fold}")
else:
    print("No model met the criteria for saving")
        
#Save the best model as an artifact to wandb...
# model_artifact = wandb.Artifact(f"best_model_fold_{fold}", type="model")
# model_artifact.add_file('./Intermediate/model.pth')
# model_artifact.metadata['best_y1_score'] = best_y1_score
# model_artifact.metadata['best_y2_score'] = best_y2_score
# model_artifact.metadata['best_combined_loss'] = best_combined_loss
# model_artifact.metadata['model skeleton'] = best_model

# wandb.log_artifact(model_artifact)

# wandb.finish()


# Print the best fold's performance
print(f"Best fold: {best_fold}")
print(f"Best validation Score for y1: {best_y1_score}")
print(f"Best validation Score for y2: {best_y2_score}")

#Disk quota dangerously exceeding, so delete the artifact locally immediately after training
artifact_deletion()