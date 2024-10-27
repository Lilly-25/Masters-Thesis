import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, global_mean_pool

class obsolete_mlp_kpi3d(nn.Module):
    def __init__(self, input_size, hidden_size, y1_shape, y2_shape, dropout_rate):
        super(mlp_kpi3d, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.y1_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size*2, y1_shape[1]),
            nn.ReLU()
        )
        self.y2_layers = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size * 4),
            # nn.BatchNorm1d(hidden_size * 4),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size * 8),
            nn.BatchNorm1d(hidden_size * 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 8, hidden_size * 16),
            nn.BatchNorm1d(hidden_size * 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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
        
class mlp_kpi3d(nn.Module):
    def __init__(self, input_size, hidden_size, y1_shape, y2_shape, dropout_rate):
        super(mlp_kpi3d, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size*2, hidden_size*4),
            nn.BatchNorm1d(hidden_size*4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.y1_layers = nn.Sequential(
            nn.Linear(hidden_size*4, y1_shape[1]),
            nn.ReLU()
        )
        self.y2_layers = nn.Sequential(
            nn.Linear(hidden_size*4, hidden_size * 8),
            nn.BatchNorm1d(hidden_size * 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 8, hidden_size * 16),
            nn.BatchNorm1d(hidden_size * 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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

class hgnn_kpi2d(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, node_dims):
        super().__init__()
        
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        
        # Node embeddings
        self.node_embeddings = torch.nn.ModuleDict()
        for node_type, in_dim in node_dims.items():
            self.node_embeddings[node_type] = Linear(in_dim, hidden_channels)
        
        # Graph convolution
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = SAGEConv(
                hidden_channels,
                hidden_channels,
                aggr='mean'
            )
        self.conv = HeteroConv(conv_dict, aggr='mean')
        
        # Output layer
        self.output = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        # Transform node features
        x_dict = {
            node_type: self.node_embeddings[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Graph convolution
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Global pooling for each node type
        x_pooled = []
        for node_type, x in x_dict.items():
            if node_type in batch_dict:
                batch = batch_dict[node_type]
                pooled = global_mean_pool(x, batch)
                x_pooled.append(pooled)
        
        # Combine all node types
        x = torch.mean(torch.stack(x_pooled), dim=0)
        
        # Final prediction
        out = self.output(x)
        
        return out
    
    def save(self, path):
        torch.save(self, path)