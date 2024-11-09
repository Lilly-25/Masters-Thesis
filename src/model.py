import torch
import torch.nn as nn
class mlp_kpi3d(nn.Module):
    def __init__(self, input_size, hidden_size, y1_shape, y2_shape, dropout_rate_shared, dropout_rate_y2):
        super(mlp_kpi3d, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate_shared),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate_shared),
        )
        self.y1_layers = nn.Sequential(
            nn.Linear(hidden_size, y1_shape[1]),
            nn.ReLU()
        )
        self.y2_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 8),
            nn.BatchNorm1d(hidden_size * 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate_y2),
            nn.Linear(hidden_size*8, hidden_size * 16),
            nn.BatchNorm1d(hidden_size * 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate_y2),
            nn.Linear(hidden_size * 16, y2_shape[1] * y2_shape[2]),
            nn.ReLU()
            # nn.Linear(hidden_size * 64, y2_shape[1] * y2_shape[2])
        )
        self.y2_shape = y2_shape

    def forward(self, x):
        shared_features = self.shared_layers(x)
        y1_pred = self.y1_layers(shared_features)
        y2_pred = self.y2_layers(shared_features).view(-1, *self.y2_shape[1:])
        #TODO should i multiply the last layers with batch size
        return y1_pred, y2_pred
    
    def save(self, path):
        torch.save(self, path)
   
