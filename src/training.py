import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

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
        self.train_rmse_history = []
        self.val_rmse_history = []
        self.train_r2_history = []
        self.val_r2_history = []

    @staticmethod
    def calculate_rmse(outputs, labels):
        return torch.sqrt(torch.mean((outputs - labels) ** 2, dim=0))

    @staticmethod
    def calculate_r2(outputs, labels):
        return r2_score(labels.cpu().numpy(), outputs.cpu().numpy(), multioutput='variance_weighted')

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_rmse, train_r2 = self._train_epoch()
            val_loss, val_rmse, val_r2 = self._validate_epoch()
            
            self.train_loss_history.append(train_loss)
            self.train_rmse_history.append(train_rmse.mean().item())
            self.train_r2_history.append(train_r2)
            self.val_loss_history.append(val_loss)
            self.val_rmse_history.append(val_rmse.mean().item())
            self.val_r2_history.append(val_r2)
            
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            print(f'Train RMSE (mean): {train_rmse.mean():.4f}, Validation RMSE (mean): {val_rmse.mean():.4f}')
            print(f'Train R2: {train_r2:.4f}, Validation R2: {val_r2:.4f}')
            print('-' * 50)

    def _train_epoch(self):
        self.model.train()
        train_loss = 0.0
        train_rmse = torch.zeros(self.output_size).to(self.device)
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
            train_rmse += self.calculate_rmse(outputs, batch_labels) * batch_features.size(0)
            train_outputs.append(outputs.detach())
            train_labels.append(batch_labels)
        
        train_loss /= len(self.train_dataset)
        train_rmse /= len(self.train_dataset)
        train_outputs = torch.cat(train_outputs)
        train_labels = torch.cat(train_labels)
        train_r2 = self.calculate_r2(train_outputs, train_labels)
        
        return train_loss, train_rmse, train_r2

    def _validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        val_rmse = torch.zeros(self.output_size).to(self.device)
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in self.val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                val_loss += loss.item() * batch_features.size(0)
                val_rmse += self.calculate_rmse(outputs, batch_labels) * batch_features.size(0)
                val_outputs.append(outputs)
                val_labels.append(batch_labels)
            
            val_loss /= len(self.val_dataset)
            val_rmse /= len(self.val_dataset)
            val_outputs = torch.cat(val_outputs)
            val_labels = torch.cat(val_labels)
            val_r2 = self.calculate_r2(val_outputs, val_labels)
        
        return val_loss, val_rmse, val_r2

    def training_plots(self):
        """
        Create plots for Loss, RMSE, and R2 metrics.
        """
        metrics = [
            ('Loss', self.train_loss_history, self.val_loss_history),
            ('RMSE', self.train_rmse_history, self.val_rmse_history),
            ('R2', self.train_r2_history, self.val_r2_history)
        ]

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
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