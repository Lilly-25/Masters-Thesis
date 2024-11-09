import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, global_mean_pool
        
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