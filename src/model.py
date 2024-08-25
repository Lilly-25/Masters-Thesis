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