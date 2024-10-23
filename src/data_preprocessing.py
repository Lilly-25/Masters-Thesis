import numpy as np
from collections import defaultdict
import torch
from torch_geometric.data import HeteroData

def collect_features_by_type(graphs):
    """
    Collect features for nodes and edges with proper edge type handling.
    """
    node_features_by_type = defaultdict(lambda: defaultdict(list))
    edge_features_by_type = defaultdict(lambda: defaultdict(list))
    
    for G in graphs:
        # Collect node features
        for node, data in G.nodes(data=True):
            node_type = data['type']
            features = data['features']
            for i, feature in enumerate(features):
                node_features_by_type[node_type][i].append(feature)
        
        # Collect edge features with full edge type
        for u, v, data in G.edges(data=True):
            # Get complete edge type descriptor
            src_type = G.nodes[u]['type']
            dst_type = G.nodes[v]['type']
            edge_type = data['type']
            full_edge_type = (src_type, edge_type, dst_type)
            
            features = data['features']
            for i, feature in enumerate(features):
                edge_features_by_type[full_edge_type][i].append(feature)
    
    return dict(node_features_by_type), dict(edge_features_by_type)


def compute_scaling_params(features_by_type):
    """
    Compute mean and standard deviation for each feature.
    
    Args:
        features_by_type (dict): Dictionary containing features by type
    
    Returns:
        dict: Dictionary containing mean and std for each feature
    """
    scaling_params = {}
    
    for type_name, features_dict in features_by_type.items():
        scaling_params[type_name] = {
            'mean': {},
            'std': {}
        }
        
        for feature_idx, feature_values in features_dict.items():
            values = np.array(feature_values)
            mean = np.mean(values)
            std = np.std(values)
            # Handle zero standard deviation
            if std == 0:
                std = 1
                
            scaling_params[type_name]['mean'][feature_idx] = mean
            scaling_params[type_name]['std'][feature_idx] = std
    
    return scaling_params

def scale_graph(G, node_scaling_params, edge_scaling_params):
    """
    Scale features using complete edge type information.
    """
    scaled_G = G.copy()
    
    # Scale node features
    for node, data in scaled_G.nodes(data=True):
        node_type = data['type']
        features = np.array(data['features'])
        
        scaled_features = []
        for i, feature in enumerate(features):
            mean = node_scaling_params[node_type]['mean'][i]
            std = node_scaling_params[node_type]['std'][i]
            scaled_feature = (feature - mean) / std
            scaled_features.append(scaled_feature)
        
        scaled_G.nodes[node]['features'] = scaled_features
    
    # Scale edge features using complete edge type
    for u, v, k, data in scaled_G.edges(data=True, keys=True):
        # Get complete edge type descriptor
        src_type = G.nodes[u]['type']
        dst_type = G.nodes[v]['type']
        edge_type = data['type']
        full_edge_type = (src_type, edge_type, dst_type)
        
        features = np.array(data['features'])
        scaled_features = []
        
        try:
            for i, feature in enumerate(features):
                mean = edge_scaling_params[full_edge_type]['mean'][i]
                std = edge_scaling_params[full_edge_type]['std'][i]
                scaled_feature = (feature - mean) / std
                scaled_features.append(scaled_feature)
            
            scaled_G.edges[u, v, k]['features'] = scaled_features
            
        except KeyError as e:
            print(f"Error scaling edge {u}->{v} of type {full_edge_type}")
            print(f"Available edge types: {list(edge_scaling_params.keys())}")
            raise e
    
    return scaled_G

def create_node_mapping(G):
    """
    Create a mapping of node IDs for each node type.
    
    Args:
        G (NetworkX.MultiDiGraph): Input graph
    
    Returns:
        dict: Dictionary mapping node names to indices for each node type
        dict: Dictionary mapping node types to number of nodes
    """
    node_mappings = {}
    num_nodes_dict = {}
    
    # Group nodes by type
    nodes_by_type = {}
    for node, data in G.nodes(data=True):
        node_type = data['type']
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append(node)
    
    # Create mappings for each node type
    for node_type, nodes in nodes_by_type.items():
        sorted_nodes = sorted(nodes)  # Ensure consistent ordering
        node_mappings[node_type] = {node: idx for idx, node in enumerate(sorted_nodes)}
        num_nodes_dict[node_type] = len(sorted_nodes)
    
    return node_mappings, num_nodes_dict

def create_edge_indices(G, node_mappings):
    """
    Create edge index tensors for each edge type.
    
    Args:
        G (NetworkX.MultiDiGraph): Input graph
        node_mappings (dict): Mapping of node names to indices
    
    Returns:
        dict: Dictionary of edge indices and features for each edge type
    """
    edge_dict = {}
    
    # Group edges by type and source/target node types
    edges_by_type = {}
    for u, v, data in G.edges(data=True):
        edge_type = data['type']
        src_type = G.nodes[u]['type']
        dst_type = G.nodes[v]['type']
        key = (src_type, edge_type, dst_type)
        
        if key not in edges_by_type:
            edges_by_type[key] = {
                'edge_index': [],
                'edge_attr': []
            }
        
        # Add edge indices
        src_idx = node_mappings[src_type][u]
        dst_idx = node_mappings[dst_type][v]
        edges_by_type[key]['edge_index'].append([src_idx, dst_idx])
        
        # Add edge features
        edges_by_type[key]['edge_attr'].append(data['features'])
    
    # Convert to tensors
    for key, data in edges_by_type.items():
        edge_indices = torch.tensor(data['edge_index'], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float)
        edge_dict[key] = (edge_indices, edge_attr)
    
    return edge_dict

def graph_to_heterodata(G):
    """
    Convert a NetworkX graph to PyTorch Geometric HeteroData.
    
    Args:
        G (NetworkX.MultiDiGraph): Input graph
    
    Returns:
        HeteroData: PyTorch Geometric HeteroData object
    """
    data = HeteroData()
    
    # Create node mappings
    node_mappings, num_nodes_dict = create_node_mapping(G)
    
    # Add node features
    for node_type in node_mappings:
        nodes = [node for node in G.nodes() if G.nodes[node]['type'] == node_type]
        features = [G.nodes[node]['features'] for node in nodes]
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float)
        data[node_type].x = x
        data[node_type].num_nodes = num_nodes_dict[node_type]
    
    # Add edge indices and features
    edge_dict = create_edge_indices(G, node_mappings)
    for (src_type, edge_type, dst_type), (edge_index, edge_attr) in edge_dict.items():
        data[src_type, edge_type, dst_type].edge_index = edge_index
        data[src_type, edge_type, dst_type].edge_attr = edge_attr
    
    # Add global graph features if they exist
    if 'mgrenz_values' in G.graph:
        data.mgrenz_values = torch.tensor(G.graph['mgrenz_values'], dtype=torch.float)
    
    return data
