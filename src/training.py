import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import wandb
import numpy as np
    
class LossRegularisation(nn.Module):
    #Penalise monotonic increasing values in the prediction
    def __init__(self, lambda_monotonic):
        super(LossRegularisation, self).__init__()
        self.lambda_monotonic = lambda_monotonic
        
    def mse_loss(self, output, target):
        loss = torch.mean((output - target)**2)
        return loss

    def forward(self, y_pred, y_true):
        
        mse_loss = self.mse_loss(y_pred, y_true)
        
        # y2-y1<=0 - it has to be a declining curve--CONFIRM WITH CLIENT
        #relu clips the negative values to zero so that increasing values are penalised
        violations = torch.relu(torch.pow(y_pred[:, 1:]-y_pred[:, :-1], 2))#Squared difference incase of increasing values in y_pred
        num_elements = violations.numel()
        regularized_factor= violations.sum()/num_elements
        #average is taken to normalise the loss
             
        return mse_loss + self.lambda_monotonic * regularized_factor


class mlp_kpi2d_trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, criterion, optimizer, device=None):
        self.model = model
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.output_size = self.model.output_layer.out_features
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_r2_history = []
        self.val_r2_history = []

    @staticmethod
    def calculate_r2(outputs, labels):
        return r2_score(labels.cpu().numpy(), outputs.cpu().numpy(), multioutput='variance_weighted')

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_r2 = self._train_epoch()
            val_loss, val_r2 = self._validate_epoch()
            
            self.train_loss_history.append(train_loss)
            self.train_r2_history.append(train_r2)
            self.val_loss_history.append(val_loss)
            self.val_r2_history.append(val_r2)
            
            # print(f'Epoch {epoch+1}/{num_epochs}')
            # print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            # print(f'Train R2: {train_r2:.4f}, Validation R2: {val_r2:.4f}')
            # print('-' * 50)
        self.training_plots()
        return self.train_loss_history, self.train_r2_history, self.val_loss_history, self.val_r2_history

    def _train_epoch(self):
        self.model.train()
        train_loss = 0.0
        train_outputs = []
        train_labels = []
        
        for batch_features, batch_labels in self.train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * batch_features.size(0)
            train_outputs.append(outputs.detach())
            train_labels.append(batch_labels)
        
        train_loss /= len(self.train_dataset)
        train_outputs = torch.cat(train_outputs)
        train_labels = torch.cat(train_labels)
        train_r2 = self.calculate_r2(train_outputs, train_labels)
        
        return train_loss, train_r2

    def _validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in self.val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                val_loss += loss.item() * batch_features.size(0)
                val_outputs.append(outputs)
                val_labels.append(batch_labels)
            
            val_loss /= len(self.val_dataset)
            val_outputs = torch.cat(val_outputs)
            val_labels = torch.cat(val_labels)
            val_r2 = self.calculate_r2(val_outputs, val_labels)
        
        return val_loss, val_r2

    def training_plots(self):
        """
        Create plots for Loss and R2 score metrics.
        """
        metrics = [
            ('MSE Loss', self.train_loss_history, self.val_loss_history),
            ('R2 Score', self.train_r2_history, self.val_r2_history)
        ]

        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Training Metrics')

        for i, (metric, train_history, val_history) in enumerate(metrics):
            axs[i].plot(range(1, len(train_history) + 1), train_history, label='Train')
            axs[i].plot(range(1, len(val_history) + 1), val_history, label='Validation')
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(metric)
            axs[i].set_title(f'Train and Validation {metric}')
            axs[i].legend()

        plt.tight_layout()
        plt.show()
        
class mlp_kpi3d_trainer:
    
    def __init__(self, model, train_dataset, val_dataset, batch_size, loss, optimizer, device):
        
        self.model = model
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        
        self.model.to(self.device)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.train_loss_history_y1, self.train_loss_history_y2, self.train_loss_history = [], [], []
        self.train_r2_history_y1, self.train_r2_history_y2 = [],[]
        
        self.val_loss_history_y1, self.val_r2_history_y1  = [],[]
        self.val_loss_history_y2, self.val_r2_history_y2  = [],[]
        self.val_loss_history = []


    @staticmethod
    def calculate_r2(outputs, labels):
        return r2_score(labels.cpu().numpy(), outputs.cpu().numpy(), multioutput='variance_weighted')
    
    @staticmethod
    def y1_score(outputs, labels):
        labels=labels.cpu().numpy()
        outputs=outputs.cpu().numpy()
        deviations = outputs - labels
        variance = np.mean(deviations ** 2)
        std_dev = np.sqrt(variance)
        print(len(labels))
        return  std_dev/len(labels)

    def train(self, num_epochs):
        for _ in range(num_epochs):
            train_loss, train_loss_y1, train_loss_y2, train_r2_y1, train_r2_y2 = self._train_epoch()
            val_loss, val_loss_y1, val_loss_y2, val_r2_y1, val_r2_y2 = self._validate_epoch()
            
            self.train_loss_history_y1.append(train_loss_y1), self.train_loss_history_y2.append(train_loss_y2)
            self.train_r2_history_y1.append(train_r2_y1), self.train_r2_history_y2.append(train_r2_y2)
            self.val_loss_history_y1.append(val_loss_y1), self.val_loss_history_y2.append(val_loss_y2)
            self.val_r2_history_y1.append(val_r2_y1), self.val_r2_history_y2.append(val_r2_y2)
            self.train_loss_history.append(train_loss), self.val_loss_history.append(val_loss)
        
        return self.train_loss_history, self.val_loss_history, self.train_loss_history_y1, self.train_loss_history_y2, self.train_r2_history_y1, self.train_r2_history_y2, self.val_loss_history_y1, self.val_loss_history_y2, self.val_r2_history_y1, self.val_r2_history_y2

    def _train_epoch(self):
        self.model.train()
        train_loss_y1, train_loss_y2, train_loss = 0.0, 0.0, 0.0
        train_losses_y1, train_losses_y2, train_losses = [], [], []
        train_outputs_y1, train_labels_y1 = [], []
        train_outputs_y2, train_labels_y2 = [], []
        
        for batch_x, batch_y1, batch_y2 in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y1 = batch_y1.to(self.device)
            batch_y2 = batch_y2.to(self.device)
            
            self.optimizer.zero_grad()
            y1_pred, y2_pred  = self.model(batch_x)
            
            #-1 values as NAN values in the ETa grid and need to be ignored when calculating loss
            mask = (batch_y2 != -1).float()
                
            # Apply the mask to both target and prediction
            y2_pred_masked = y2_pred * mask
            batch_y2_masked = batch_y2 * mask
            
            loss_y1 = self.loss(y1_pred, batch_y1)
            loss_y2 = self.loss(y2_pred_masked, batch_y2_masked)
            loss = loss_y1 + loss_y2
            
            loss.backward()
            self.optimizer.step()
            
            train_loss_y1 += loss_y1.item() * batch_x.size(0)
            train_loss_y2 += loss_y2.item() * batch_x.size(0)
            train_loss += loss.item() * batch_x.size(0)
            
            train_outputs_y1.append(y1_pred.detach())
            train_labels_y1.append(batch_y1)
            train_outputs_y2.append(y2_pred.detach())
            train_labels_y2.append(batch_y2)
        
        train_loss_y1 /= len(self.train_dataset)
        train_loss_y2 /= len(self.train_dataset)
        train_loss /= len(self.train_dataset)
        
        train_losses_y1.append(train_loss_y1)
        train_losses_y2.append(train_loss_y2)
        train_losses.append(train_loss) 
        
        train_outputs_y1 = torch.cat(train_outputs_y1)
        train_labels_y1 = torch.cat(train_labels_y1)
        
        train_r2_y1 = self.y1_score(train_outputs_y1, train_labels_y1)
        
        train_outputs_y2 = torch.cat(train_outputs_y2)
        train_labels_y2 = torch.cat(train_labels_y2)
        
        # Reshape 3D arrays to 2D arrays for R2 calculation
        train_outputs_y2 = train_outputs_y2.view(train_outputs_y2.size(0), -1)
        train_labels_y2 = train_labels_y2.view(train_labels_y2.size(0), -1)
        
        train_r2_y2 = self.calculate_r2(train_outputs_y2, train_labels_y2)
        
        wandb.log({
            "train_loss": train_loss,
            "train_loss_y1": train_loss_y1,
            "train_loss_y2": train_loss_y2,
            "train_r2_y1": train_r2_y1,
            "train_r2_y2": train_r2_y2
        })
        
        return train_loss, train_loss_y1, train_loss_y2, train_r2_y1, train_r2_y2

    def _validate_epoch(self):
        self.model.eval()
        val_loss_y1, val_loss_y2, val_loss = 0.0, 0.0, 0.0
        val_losses_y1, val_losses_y2, val_losses = [], [], []
        val_outputs_y1, val_labels_y1 = [], []
        val_outputs_y2, val_labels_y2 = [], []
        
        with torch.no_grad():
            
            for batch_x, batch_y1, batch_y2 in self.val_loader:
                
                batch_x = batch_x.to(self.device)
                batch_y1 = batch_y1.to(self.device)
                batch_y2 = batch_y2.to(self.device)

                y1_pred, y2_pred  = self.model(batch_x)
                
                #-1 values as NAN values in the ETa grid and need to be ignored when calculating loss
                mask = (batch_y2 != -1).float()
                
                # Apply the mask to both target and prediction
                y2_pred_masked = y2_pred * mask
                batch_y2_masked = batch_y2 * mask
                
                loss_y1 = self.loss(y1_pred, batch_y1)
                loss_y2 = self.loss(y2_pred_masked, batch_y2_masked)

                val_loss_y1 += loss_y1.item() * batch_x.size(0)
                val_loss_y2 += loss_y2.item() * batch_x.size(0)
                val_loss= val_loss_y1+val_loss_y2
                
                val_outputs_y1.append(y1_pred.detach())
                val_labels_y1.append(batch_y1)
                val_outputs_y2.append(y2_pred.detach())
                val_labels_y2.append(batch_y2)
        
        val_loss_y1 /= len(self.train_dataset)
        val_loss_y2 /= len(self.train_dataset)
        val_loss /= len(self.train_dataset) 
        
        val_losses_y1.append(val_loss_y1)
        val_losses_y2.append(val_loss_y2)
        val_losses.append(val_loss)
        
        val_outputs_y1 = torch.cat(val_outputs_y1)
        val_labels_y1 = torch.cat(val_labels_y1)
        
        val_r2_y1 = self.calculate_r2(val_outputs_y1, val_labels_y1)
        
        val_outputs_y2 = torch.cat(val_outputs_y2)
        val_labels_y2 = torch.cat(val_labels_y2)
        
        # Reshape 3D arrays to 2D arrays for R2 calculation
        val_outputs_y2 = val_outputs_y2.view(val_outputs_y2.size(0), -1)
        val_labels_y2 = val_labels_y2.view(val_labels_y2.size(0), -1)
        
        val_r2_y2 = self.calculate_r2(val_outputs_y2, val_labels_y2)
        
        wandb.log({
            "val_loss": val_loss,
            "val_loss_y1": val_loss_y1,
            "val_loss_y2": val_loss_y2,
            "val_r2_y1": val_r2_y1,
            "val_r2_y2": val_r2_y2
        })
        
        return val_loss, val_loss_y1, val_loss_y2, val_r2_y1, val_r2_y2
        