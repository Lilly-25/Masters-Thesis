import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
from src.inference import eta_difference, y2_score
    
class Y1LossRegularisation(nn.Module):
    #Penalise monotonic increasing values in the prediction
    def __init__(self, lambda1_y1, lambda2_y1):
        super(Y1LossRegularisation, self).__init__()
        self.lambda1_y1 = lambda1_y1
        self.lambda2_y1 = lambda2_y1
        
    def mse_loss(self, output, target):
        loss = torch.mean((output - target)**2)
        return loss

    def forward(self, y_pred, y_target):
        mse_loss = self.mse_loss(y_pred, y_target)
        # y2-y1<=0 - it has to be a declining curve--
        #relu clips the negative values to zero so that increasing values are penalised
        smoothened_curve_violations = torch.pow(torch.relu(torch.abs(y_pred[:, 1:]-y_pred[:, :-1]) - 1.0), 2) # Violations if difference in consecutive values in prediction are greater than 1

        decreasing_curve_violations = torch.pow(torch.relu(y_pred[:, 1:]-y_pred[:, :-1]), 2)#Squared difference incase of increasing values in y_pred
            # y2-y1>=0 - then it would have to be an increasing curve
            
        regularized_factor_smoothened_curve= smoothened_curve_violations.sum()/smoothened_curve_violations.numel() # Calculation is as expected only acrosss each example, batch total count is only used for averaging
        regularized_factor_decreasing_curve= decreasing_curve_violations.sum()/decreasing_curve_violations.numel() # Calculation is as expected only acrosss each example, batch total count is only used for averaging
        return mse_loss + self.lambda1_y1 * regularized_factor_smoothened_curve + self.lambda2_y1 * regularized_factor_decreasing_curve
    
class Y2LossRegularisation(nn.Module):
    def __init__(self, lambda_y2, border_threshold, low_nn_threshold, low_mm_threshold):
        super(Y2LossRegularisation, self).__init__()
        self.lambda_y2 = lambda_y2
        self.border_threshold = border_threshold
        self.low_nn_threshold = low_nn_threshold
        self.low_mm_threshold = low_mm_threshold
    
    def mse_loss(self, output, target):
        return torch.mean((output - target)**2)
    
    def forward(self, y_pred, y_target):
        mse_loss = self.mse_loss(y_pred, y_target)
        
        violations_nn_max_mgrenz = (y_pred[:, -self.border_threshold, :] - y_target[:, -self.border_threshold, :]) ** 2
        violations_low_nn = (y_pred[:, :, :self.low_nn_threshold] - y_target[:, :, :self.low_nn_threshold]) ** 2
        violations_low_mm = (y_pred[:, :self.low_mm_threshold, :] - y_pred[:, :self.low_mm_threshold, :]) ** 2
        
        # Compute regularized factors as averages of squared violations
        regularized_factor_nn_maxmgrenz = violations_nn_max_mgrenz.sum() / violations_nn_max_mgrenz.numel()
        regularized_factor_low_nn = violations_low_nn.sum() / violations_low_nn.numel()
        regularized_factor_low_mm = violations_low_mm.sum() / violations_low_mm.numel()

        # Combine MSE loss with L2 regularization
        return mse_loss + self.lambda_y2 * (regularized_factor_nn_maxmgrenz + 
                                               regularized_factor_low_nn + 
                                               regularized_factor_low_mm)

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
    def y1_avg_score(outputs, labels): #RMSE
        labels=labels.cpu().numpy()
        outputs=outputs.cpu().numpy()
        deviations = outputs - labels
        variance = np.mean(deviations ** 2)
        std_dev = np.sqrt(variance)
        return std_dev

    @staticmethod
    def y2_avg_score(outputs, labels): #RMSE
        labels = labels.cpu().numpy()
        outputs = outputs.cpu().numpy()
        score = 0
        y2_mlp_scores = []
        for i in range(labels.shape[0]):
            eta_diff = eta_difference(labels[i], outputs[i])
            rmse = y2_score(eta_diff)
            y2_mlp_scores.append(rmse)
            score+=rmse
        
        return score/(labels.shape[0])

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_loss_y1, train_loss_y2, train_score, train_score_y1, train_score_y2 = self._train_epoch()
            val_loss, val_loss_y1, val_loss_y2, val_score, val_score_y1, val_score_y2 = self._validate_epoch()
            wandb.log({
            "train_loss": train_loss,
            "train_loss_y1": train_loss_y1,
            "train_loss_y2": train_loss_y2,
            "train_score_y1": train_score_y1,
            "train_score_y2": train_score_y2,
            "train_score" : train_score,
            "val_loss": val_loss,
            "val_loss_y1": val_loss_y1,
            "val_loss_y2": val_loss_y2,
            "val_score_y1": val_score_y1,
            "val_score_y2": val_score_y2, 
            'val_score':val_score,
            'epoch': epoch + 1
            })
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
        train_outputs_y1, train_labels_y1 = [], []
        train_outputs_y2, train_labels_y2 = [], []
        
        for batch_x, batch_y1, batch_y2 in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y1 = batch_y1.to(self.device)
            batch_y2 = batch_y2.to(self.device)
            
            self.optimizer.zero_grad()
            y1_pred, y2_pred  = self.model(batch_x)
            
            #NAN values in the ETA grid need to be ignored when calculating loss..
            mask = (~torch.isnan(batch_y2)).float()
                
            # Apply the mask to both target and prediction
            y2_pred_masked = y2_pred * mask
            batch_y2_masked = torch.where(torch.isnan(batch_y2), torch.zeros_like(batch_y2), batch_y2)# Else we will get nan losses as nan*0 is still nan
            
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
        
        train_outputs_y1 = torch.cat(train_outputs_y1)
        train_labels_y1 = torch.cat(train_labels_y1)
        
        train_score_y1 = self.y1_avg_score(train_outputs_y1, train_labels_y1)
        
        train_outputs_y2 = torch.cat(train_outputs_y2)
        train_labels_y2 = torch.cat(train_labels_y2)
        
        train_score_y2 = self.y2_avg_score(train_outputs_y2, train_labels_y2)
        
        train_score = train_score_y1 + train_score_y2
        
        return train_loss, train_loss_y1, train_loss_y2, train_score, train_score_y1, train_score_y2

    def _validate_epoch(self):
        self.model.eval()
        val_loss_y1, val_loss_y2, val_loss = 0.0, 0.0, 0.0
        val_outputs_y1, val_labels_y1 = [], []
        val_outputs_y2, val_labels_y2 = [], []
        
        with torch.no_grad():
            
            for batch_x, batch_y1, batch_y2 in self.val_loader:
                
                batch_x = batch_x.to(self.device)
                batch_y1 = batch_y1.to(self.device)
                batch_y2 = batch_y2.to(self.device)

                y1_pred, y2_pred  = self.model(batch_x)
                
                #NAN values in the ETA grid need to be ignored when calculating loss..
                mask = (~torch.isnan(batch_y2)).float()
                    
                # Apply the mask to both target and prediction
                y2_pred_masked = y2_pred * mask
                batch_y2_masked = torch.where(torch.isnan(batch_y2), torch.zeros_like(batch_y2), batch_y2)# Else we will get nan losses as nan*0 is still nan
                
                loss_y1 = self.loss1(y1_pred, batch_y1)
                loss_y2 = self.loss2(y2_pred_masked, batch_y2_masked)
                loss = self.w_y1*loss_y1 + self.w_y2*loss_y2 #Increase weightage for y1 as y2 depends on it

                val_loss_y1 += loss_y1.item() * batch_x.size(0)
                val_loss_y2 += loss_y2.item() * batch_x.size(0)
                val_loss += loss.item() * batch_x.size(0)
                
                val_outputs_y1.append(y1_pred.detach())
                val_labels_y1.append(batch_y1)
                val_outputs_y2.append(y2_pred.detach())
                val_labels_y2.append(batch_y2)
        
        val_loss_y1 /= len(self.train_dataset)
        val_loss_y2 /= len(self.train_dataset)
        val_loss /= len(self.train_dataset) 
        
        val_outputs_y1 = torch.cat(val_outputs_y1)
        val_labels_y1 = torch.cat(val_labels_y1)
        
        val_score_y1 = self.y1_avg_score(val_outputs_y1, val_labels_y1)
        
        val_outputs_y2 = torch.cat(val_outputs_y2)
        val_labels_y2 = torch.cat(val_labels_y2)
        
        val_score_y2 = self.y2_avg_score(val_outputs_y2, val_labels_y2)
        
        val_score = val_score_y1 + val_score_y2
        
        return val_loss, val_loss_y1, val_loss_y2,val_score, val_score_y1, val_score_y2
        