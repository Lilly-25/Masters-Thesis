import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import wandb
import numpy as np
    
class Y1LossRegularisation(nn.Module):
    #Penalise monotonic increasing values in the prediction
    def __init__(self, lambda_y1):
        super(Y1LossRegularisation, self).__init__()
        self.lambda_y1 = lambda_y1
        
    def mse_loss(self, output, target):
        loss = torch.mean((output - target)**2)
        return loss

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)
        
        # y2-y1<=0 - it has to be a declining curve--
        #relu clips the negative values to zero so that increasing values are penalised
        violations = torch.pow(torch.relu(y_pred[:, 1:]-y_pred[:, :-1]), 2)#Squared difference incase of increasing values in y_pred
        #Do we go with removing the square, coz it doesnt seem to exaggerate loss
        #I fear the loss would try to create a decreasing curve though out the batch
        regularized_factor= violations.sum()/violations.numel()
        #Consider per example although does not seem necessary

        return mse_loss + self.lambda_y1 * regularized_factor
    
class Y2LossRegularisation(nn.Module):
    def __init__(self, lambda_y2):
        super(Y2LossRegularisation, self).__init__()
        self.lambda_y2 = lambda_y2
    
    def mse_loss(self, output, target):
        return torch.mean((output - target)**2)
    
    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)

        middle_row = y_pred.shape[1] // 2
        mid_row_vals = y_pred[:, middle_row, :]#Extract middle row values.
        violations = torch.abs(mid_row_vals) # If not 0, it is a violation
        regularized_factor=  violations.sum()/violations.numel()
        
        return mse_loss + self.lambda_y2 * regularized_factor
    
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def mse_loss(self, output, target):
        loss = torch.mean((output - target)**2)
        return loss

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)  
        return mse_loss 
    


class mlp_kpi3d_trainer:
    
    def __init__(self, model, train_dataset, val_dataset, batch_size, loss1, loss2, optimizer, lrscheduler, w_y1, w_y2, device):
        
        self.model = model
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        
        self.loss1 = loss1
        self.loss2 = loss2
        self.optimizer = optimizer
        self.lr_scheduler=lrscheduler
        self.device = device
        
        self.w_y1 = w_y1
        self.w_y2 = w_y2
        
        self.model.to(self.device)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.train_loss_history_y1, self.train_loss_history_y2, self.train_loss_history = [], [], []
        self.train_score_history_y1, self.train_score_history_y2, self.train_score_history = [],[], []
        
        self.val_loss_history_y1, self.val_score_history_y1, self.val_loss_history = [], [],[]
        self.val_loss_history_y2, self.val_score_history_y2, self.val_score_history = [], [],[]

    @staticmethod
    def y1_score(outputs, labels): #RMSE
        labels=labels.cpu().numpy()
        outputs=outputs.cpu().numpy()
        deviations = outputs - labels
        variance = np.mean(deviations ** 2)
        std_dev = np.sqrt(variance)
        return std_dev
    
    @staticmethod
    def y2_score(outputs, labels): #RMSE
        labels=labels.cpu().numpy()
        outputs=outputs.cpu().numpy()
        
        if labels.shape[0] > outputs.shape[0]:
            larger_eta, smaller_eta = labels, outputs
        else:
            larger_eta, smaller_eta = labels, outputs
        
        rows, _ = larger_eta.shape
        smaller_rows, smaller_cols = smaller_eta.shape
        
        # Create the interpolated grid
        interpolated_smaller = np.full_like(larger_eta, np.nan)
        start_row = (rows - smaller_rows) // 2
        
        # Place the smaller eta directly into the middle of the larger grid
        interpolated_smaller[start_row:start_row+smaller_rows, :smaller_cols] = smaller_eta
        
        Z_diff = larger_eta - interpolated_smaller
        
        valid_diff = np.abs(Z_diff[~np.isnan(Z_diff)])
        variance = np.mean(valid_diff ** 2)
        score = np.sqrt(variance)
        return score

    def train(self, num_epochs):
        for _ in range(num_epochs):
            train_loss, train_loss_y1, train_loss_y2, train_score, train_score_y1, train_score_y2 = self._train_epoch()
            val_loss, val_loss_y1, val_loss_y2, val_score, val_score_y1, val_score_y2 = self._validate_epoch()
            
            self.train_loss_history_y1.append(train_loss_y1), self.train_loss_history_y2.append(train_loss_y2)
            self.train_score_history_y1.append(train_score_y1), self.train_score_history_y2.append(train_score_y2)
            self.val_loss_history_y1.append(val_loss_y1), self.val_loss_history_y2.append(val_loss_y2)
            self.val_score_history_y1.append(val_score_y1), self.val_score_history_y2.append(val_score_y2)
            self.train_loss_history.append(train_loss), self.val_loss_history.append(val_loss)
            self.train_score_history.append(train_score), self.val_score_history.append(val_score)
        
        return self.train_loss_history, self.val_loss_history, self.train_loss_history_y1, self.train_loss_history_y2, self.train_score_history_y1, self.train_score_history_y2, self.val_loss_history_y1, self.val_loss_history_y2, self.val_score_history_y1, self.val_score_history_y2

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
            
            #-1 values as NAN values in the ETA grid need to be ignored when calculating loss..check with NAN instead of -1 - TODO
            mask = (batch_y2 != -1).float()
                
            # Apply the mask to both target and prediction
            y2_pred_masked = y2_pred * mask
            batch_y2_masked = batch_y2 * mask
            
            loss_y1 = self.loss1(y1_pred, batch_y1)
            loss_y2 = self.loss2(y2_pred_masked, batch_y2_masked)
            loss = self.w_y1*loss_y1 + self.w_y2*loss_y2 #Increase weightage for y1 as y2 depends on it
            
            loss.backward()
            self.optimizer.step()
            
            train_loss_y1 += loss_y1.item() * batch_x.size(0)
            train_loss_y2 += loss_y2.item() * batch_x.size(0)
            train_loss += loss.item() * batch_x.size(0)
            
            train_outputs_y1.append(y1_pred.detach())
            train_labels_y1.append(batch_y1)
            train_outputs_y2.append(y2_pred.detach())
            train_labels_y2.append(batch_y2)
        
        self.lr_scheduler.step()
        
        train_loss_y1 /= len(self.train_dataset)
        train_loss_y2 /= len(self.train_dataset)
        train_loss /= len(self.train_dataset)
        
        train_losses_y1.append(train_loss_y1)
        train_losses_y2.append(train_loss_y2)
        train_losses.append(train_loss) 
        
        train_outputs_y1 = torch.cat(train_outputs_y1)
        train_labels_y1 = torch.cat(train_labels_y1)
        
        train_score_y1 = self.y1_score(train_outputs_y1, train_labels_y1)
        
        train_outputs_y2 = torch.cat(train_outputs_y2)
        train_labels_y2 = torch.cat(train_labels_y2)
        
        # Reshape 3D arrays to 2D arrays for score calculation...try to remove this and see if any error-TODO
        train_outputs_y2 = train_outputs_y2.view(train_outputs_y2.size(0), -1)
        train_labels_y2 = train_labels_y2.view(train_labels_y2.size(0), -1)
        
        train_score_y2 = self.y2_score(train_outputs_y2, train_labels_y2)
        
        train_score = train_score_y1 + train_score_y2
        
        wandb.log({
            "train_loss": train_loss,
            "train_loss_y1": train_loss_y1,
            "train_loss_y2": train_loss_y2,
            "train_score_y1": train_score_y1,
            "train_score_y2": train_score_y2,
            "train_score" : train_score
        })
        
        return train_loss, train_loss_y1, train_loss_y2, train_score, train_score_y1, train_score_y2

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
                
                #-1 values as NAN values in the ETA grid and need to be ignored when calculating loss..check with NAN instead of -1 - TODO
                mask = (batch_y2 != -1).float()
                
                # Apply the mask to both target and prediction
                y2_pred_masked = y2_pred * mask
                batch_y2_masked = batch_y2 * mask
                
                loss_y1 = self.loss1(y1_pred, batch_y1)
                loss_y2 = self.loss2(y2_pred_masked, batch_y2_masked)

                val_loss_y1 += loss_y1.item() * batch_x.size(0)
                val_loss_y2 += loss_y2.item() * batch_x.size(0)
                val_loss= self.w_y1*val_loss_y1 + self.w_y2*val_loss_y2
                
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
        
        val_score_y1 = self.y1_score(val_outputs_y1, val_labels_y1)
        
        val_outputs_y2 = torch.cat(val_outputs_y2)
        val_labels_y2 = torch.cat(val_labels_y2)
        
        # Reshape 3D arrays to 2D arrays for y2 calculation
        val_outputs_y2 = val_outputs_y2.view(val_outputs_y2.size(0), -1)
        val_labels_y2 = val_labels_y2.view(val_labels_y2.size(0), -1)
        
        val_score_y2 = self.y2_score(val_outputs_y2, val_labels_y2)
        
        val_score = val_score_y1 + val_score_y2
        
        wandb.log({
            "val_loss": val_loss,
            "val_loss_y1": val_loss_y1,
            "val_loss_y2": val_loss_y2,
            "val_score_y1": val_score_y1,
            "val_score_y2": val_score_y2, 
            'val_score':val_score
        })
        
        return val_loss, val_loss_y1, val_loss_y2,val_score, val_score_y1, val_score_y2
        