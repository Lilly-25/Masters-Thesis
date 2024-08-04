import torch
import networkx as nx
from torch_geometric.data import HeteroData

def pyg_graph(G):
        try:
            # Node feature matrices (one for each node type)
            x_dict = {}
            for node_type in set(nx.get_node_attributes(G, 'node_type').values()):
                nodes_of_type = [n for n, data in G.nodes(data=True) if data['node_type'] == node_type]
                features = [G.nodes[n].get('features', []) for n in nodes_of_type]
                if features and any(features):
                    max_len = max(len(f) for f in features)
                    features = [f + [0] * (max_len - len(f)) for f in features]##Padding
                    x_dict[node_type] = torch.tensor(features, dtype=torch.float)
                    #print('Padding encountered')
                else:
                    x_dict[node_type] = torch.tensor([[]], dtype=torch.float)

            # Node type mapping
            node_type_dict = {n: G.nodes[n]['node_type'] for n in G.nodes()}
            
            # Edge indices and attributes (one for each edge type)
            edge_index_dict = {}
            edge_attr_dict = {}
            for edge_type in set(nx.get_edge_attributes(G, 'edge_type').values()):
                edges_of_type = [(u, v) for (u, v, data) in G.edges(data=True) if data['edge_type'] == edge_type]
                if edges_of_type:
                    edge_index = torch.tensor([[list(G.nodes()).index(u), list(G.nodes()).index(v)] for u, v in edges_of_type], dtype=torch.long).t()
                    edge_features = [G[u][v].get('features', []) for u, v in edges_of_type]
                    if edge_features and any(edge_features):
                        max_len = max(len(f) for f in edge_features)
                        edge_features = [f + [0] * (max_len - len(f)) for f in edge_features]
                        edge_attr = torch.tensor(edge_features, dtype=torch.float)
                    else:
                        edge_attr = torch.tensor([[0.0]], dtype=torch.float)
                    edge_index_dict[(node_type_dict[edges_of_type[0][0]], edge_type, node_type_dict[edges_of_type[0][1]])] = edge_index
                    edge_attr_dict[(node_type_dict[edges_of_type[0][0]], edge_type, node_type_dict[edges_of_type[0][1]])] = edge_attr

            # Global attributes
            global_attr = torch.tensor([G.graph['r_a'], G.graph['r_i'], G.graph['r_r']], dtype=torch.float)

            # Labels
            mgrenz_values = G.graph['mgrenz_values']
            labels = torch.tensor(mgrenz_values, dtype=torch.float)

            data = HeteroData()

            # Add node features
            for node_type, x in x_dict.items():
                data[node_type].x = x

            # Add edge indices and attributes
            for edge_type, edge_index in edge_index_dict.items():
                src, relation, dst = edge_type
                data[edge_type].edge_index = edge_index
                data[edge_type].edge_attr = edge_attr_dict[edge_type]

            # Add global attributes and labels
            data.global_attr = global_attr
            data.y = labels

            return data
        except Exception as e:
            print('Exception when converting graph attributes into tensor', str(e))