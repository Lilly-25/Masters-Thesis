import torch
import torch.nn as nn

class mlp_kpi2d(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(mlp_kpi2d, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_sizes[i+1])
            )
            for i in range(len(hidden_sizes)-1)
        ])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
    def save(self, path):
        torch.save(self, path)
        


class mlp_kpi3d(nn.Module):
    def __init__(self, input_size, hidden_size, y1_size, y2_shape):
        super(mlp_kpi3d, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.y1_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y1_size)
        )
        self.y2_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, y2_shape[1] * y2_shape[2])
        )
        self.y2_shape = y2_shape

    def forward(self, x):
        shared_features = self.shared_layers(x)
        y1_pred = self.y1_layers(shared_features)
        y2_pred = self.y2_layers(shared_features).view(-1, *self.y2_shape[1:])
        return y1_pred, y2_pred
    
    def save(self, path):
        torch.save(self, path)