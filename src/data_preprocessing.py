import torch
import networkx as nx
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

def pyg_graph(G):
    try:
        data = HeteroData()
        node_to_index = {node: i for i, node in enumerate(G.nodes())}
        
        # Node feature handling
        for node_type in set(nx.get_node_attributes(G, 'node_type').values()):
            nodes_of_type = [n for n, data in G.nodes(data=True) if data['node_type'] == node_type]
            features = [G.nodes[n].get('features', []) for n in nodes_of_type]
            
            if features and any(features):
                features = np.array(features)
                imputer = SimpleImputer(strategy='mean')
                scaler = StandardScaler()
                
                features = imputer.fit_transform(features)
                features = scaler.fit_transform(features)
                
                data[node_type].x = torch.tensor(features, dtype=torch.float)
            else:
                data[node_type].x = torch.tensor([[0.0]], dtype=torch.float)
        
        # Edge feature handling
        for edge_type in set(nx.get_edge_attributes(G, 'edge_type').values()):
            edges_of_type = [(u, v) for (u, v, d) in G.edges(data=True) if d['edge_type'] == edge_type]
            if edges_of_type:
                edge_index = torch.tensor([[node_to_index[u], node_to_index[v]] for u, v in edges_of_type], dtype=torch.long).t()
                
                edge_features = []
                for u, v in edges_of_type:
                    try:
                        features = G[u][v][0]['features']
                        edge_features.append(features)
                    except KeyError:
                        print(f"KeyError: Edge ({u}, {v}) of type {edge_type} is missing 'features'")
                        edge_features.append([0.0])
                
                if edge_features and any(edge_features):
                    edge_features = np.array(edge_features)
                    
                    if 'deg_phi' in G[edges_of_type[0][0]][edges_of_type[0][1]][0]['features']:
                        deg_phi_index = G[edges_of_type[0][0]][edges_of_type[0][1]][0]['features'].index('deg_phi')
                        deg_phi = edge_features[:, deg_phi_index]
                        sin_phi = np.sin(np.deg2rad(deg_phi))
                        cos_phi = np.cos(np.deg2rad(deg_phi))
                        edge_features = np.column_stack([
                            edge_features[:, :deg_phi_index],
                            sin_phi,
                            cos_phi,
                            edge_features[:, deg_phi_index+1:]
                        ])
                    
                    imputer = SimpleImputer(strategy='mean')
                    scaler = StandardScaler()
                    
                    edge_features = imputer.fit_transform(edge_features)
                    edge_features = scaler.fit_transform(edge_features)
                    
                    edge_attr = torch.tensor(edge_features, dtype=torch.float)
                else:
                    edge_attr = torch.tensor([[0.0] for _ in range(len(edges_of_type))], dtype=torch.float)
                
                src_type = G.nodes[edges_of_type[0][0]]['node_type']
                dst_type = G.nodes[edges_of_type[0][1]]['node_type']
                data[src_type, edge_type, dst_type].edge_index = edge_index
                data[src_type, edge_type, dst_type].edge_attr = edge_attr
        
        # Global attributes and labels
        data.global_attr = torch.tensor([G.graph['r_a'], G.graph['r_i'], G.graph['r_r']], dtype=torch.float)
        data.y = torch.tensor(G.graph['mgrenz_values'], dtype=torch.float)
        
        return data
    except Exception as e:
        print('Exception when converting graph attributes into tensor', str(e))
        return None