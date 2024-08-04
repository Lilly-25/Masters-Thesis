
import networkx as nx
import numpy as np
import openpyxl
import pandas as pd
import os
import matplotlib.pyplot as plt

def create_heterograph(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='input_data', header=None)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        params_dict = {}
        for index, row in df.iterrows():
            param, value = row[0], row[1]
            if pd.notna(param) and pd.notna(value):
                try:
                    params_dict[param] = float(value)
                except ValueError:
                    params_dict[param] = value

        G = nx.MultiDiGraph()  # Use MultiDiGraph for heterogeneous graph

        # Define node types and their features
        node_types = {
            'v1': ['v1m1', 'v1m2'],
            'v2': ['v2m1', 'v2m2'],
            'rotor': ['rr'],
            'stator': ['s']
        }

        node_features = {
            'v1': ['mbv1', 'mhv1', 'lmsov1', 'lth1v1', 'lth2v1', 'lmov1', 'lmuv1', 'r1v1', 'r11v1', 'r2v1', 'r3v1', 'r4v1', 'rmt1v1', 'rmt4v1', 'rlt1v1', 'rlt4v1', 'lmiv1', 'lmav1', 'rmagv1'],
            'v2': ['mbv2', 'mhv2', 'lmsov2', 'lth1v2', 'lth2v2', 'lmov2', 'lmuv2', 'r1v2', 'r11v2', 'r2v2', 'r3v2', 'r4v2', 'rmt1v2', 'rmt4v2', 'rlt1v2', 'rlt4v2', 'lmiv2', 'lmav2', 'rmagv2'],
            'rotor': [],
            'stator': ['b_nng', 'b_nzk', 'b_s', 'h_n', 'h_s', 'h_zk', 'r_sn', 'r_zk', 'r_ng', 'bhp', 'hhp', 'rhp']
        }

        # Add nodes with their types and features
        for node_type, nodes in node_types.items():
            for node in nodes:
                features = [params_dict.get(param, 0) for param in node_features[node_type]]
                G.add_node(node, node_type=node_type, features=features)

        # Define edge types and their features
        edge_types = {
            'vm1_vm2': [('v2m2', 'v2m1'), ('v1m2', 'v1m1')],
            'v_rotor': [('v1m2', 'rr'), ('v1m1', 'rr'), ('v2m2', 'rr'), ('v2m1', 'rr')],
            'v1_v2': [('v2m2', 'v1m2'), ('v2m1', 'v1m1')],
            'stator_rotor': [('s', 'rr')]
        }

        edge_features = {
            'vm1_vm2': ['dsm', 'dsmu', 'ha', 'deg_phi'],
            'v_rotor': ['amtr', 'dsr'],
            'v1_v2': ['amtr_diff'],
            'stator_rotor': ['airgap']
        }

        #Add edges with their types and features
                
        for edge_type, edges in edge_types.items():
                for edge in edges:
                    features = []
                    for feature in edge_features[edge_type]:
                        if edge[0].startswith('v1'):
                            value = params_dict.get(f"{feature}v1", 0)
                        elif edge[0].startswith('v2'):
                            value = params_dict.get(f"{feature}v2", 0)
                        else:
                            value = params_dict.get(feature, 0)
                        features.append(value)
                    
                    if edge_type == 'v1_v2':
                        features = [params_dict.get('amtrv2', 0) - params_dict.get('amtrv1', 0)]
                    
                    G.add_edge(edge[0], edge[1], edge_type=edge_type, features=features)

        G.graph['r_a'] = params_dict.get('r_a', 0)
        G.graph['r_i'] = params_dict.get('r_i', 0)
        G.graph['r_r'] = params_dict.get('r_i', 0) - params_dict.get('airgap', 0)

        # Read labels from 'Mgrenz' sheet
        wb = openpyxl.load_workbook(file_path)
        sheet_mgrenz = wb['Mgrenz']
        mgrenz_values = [cell.value for cell in sheet_mgrenz[1] if cell.value is not None]

        G.graph['mgrenz_values'] = mgrenz_values
        
        # print("\nNode features:")
        # for node, data in G.nodes(data=True):
        #     print(f"Node {node} ({data['node_type']}): {data['features']}")
        
        # print("\nEdge features:")
        # for u, v, data in G.edges(data=True):
        #     print(f"Edge ({u}, {v}) ({data['edge_type']}): {data['features']}")
        
        # print(f"\nmgrenz_values: {G.graph['mgrenz_values'][:5]}... (showing first 5)")
        # print(f"Global attributes: r_a={G.graph['r_a']}, r_i={G.graph['r_i']}, r_r={G.graph['r_r']}")

        return G
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return None
    

def visualize_heterograph(G):
    
    plt.figure(figsize=(12, 8)) 
    # Define color map for node types
    color_map = {'v1': 'red', 'v2': 'blue', 'rotor': 'green', 'stator': 'yellow'}
    # Get node colors based on node type
    node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]
    # Define edge color map
    edge_color_map = {'vm1_vm2': 'red', 'v_rotor': 'blue', 
                      'v1_v2': 'green', 'stator_rotor': 'yellow'}
    # Get edge colors based on edge type
    edge_colors = [edge_color_map[G[u][v][0]['edge_type']] for u, v in G.edges()]
    # Create layout
    pos = nx.spring_layout(G)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, arrows=True)
    # Draw labels
    nx.draw_networkx_labels(G, pos)
    # Add a legend for node types
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=node_type,
                                  markerfacecolor=color, markersize=10)
                       for node_type, color in color_map.items()]
    plt.legend(handles=legend_elements, title="Node Types")
    plt.title("Heterogeneous Graph Visualization")
    #plt.axis('off')
    plt.tight_layout()
    plt.show()
