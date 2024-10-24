from collections import defaultdict
import torch
from torch_geometric.data import HeteroData

def type_features(graphs):
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



def node_mapping(G):
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

def edge_mapping(G, node_mappings):
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

def heterodata_graph(G):
    """
    Convert a NetworkX graph to PyTorch Geometric HeteroData.
    
    Args:
        G (NetworkX.MultiDiGraph): Input graph
    
    Returns:
        HeteroData: PyTorch Geometric HeteroData object
    """
    data = HeteroData()
    
    # Create node mappings
    node_mappings, num_nodes_dict = node_mapping(G)
    
    # Add node features
    for node_type in node_mappings:
        nodes = [node for node in G.nodes() if G.nodes[node]['type'] == node_type]
        features = [G.nodes[node]['features'] for node in nodes]
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float)
        data[node_type].x = x
        data[node_type].num_nodes = num_nodes_dict[node_type]
    
    # Add edge indices and features
    edge_dict = edge_mapping(G, node_mappings)
    for (src_type, edge_type, dst_type), (edge_index, edge_attr) in edge_dict.items():
        data[src_type, edge_type, dst_type].edge_index = edge_index
        data[src_type, edge_type, dst_type].edge_attr = edge_attr
    
    # Add global graph features if they exist
    if 'mgrenz_values' in G.graph:
        data.mgrenz_values = torch.tensor(G.graph['mgrenz_values'], dtype=torch.float)
    
    return data
