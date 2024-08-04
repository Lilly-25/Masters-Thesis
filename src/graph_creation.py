
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
            'v1': ['v1m1', 'v1m2'],##Automate to take based on topology type V1, V3?
            'v2': ['v2m1', 'v2m2'],##Automate to take based on topology type V1, V3?
            'ri':['ri'],
            'stator': ['s1', 's2', 's3', 's4', 's5', 's6']##Automate to take from N?
        }

        #Automate to take based on naming convention maybe?
        node_features = {
            'v1': ['mbv1', 'mhv1', 'lmsov1', 'lth1v1', 'lth2v1', 'lmov1', 'lmuv1', 'r1v1', 'r11v1', 'r2v1', 'r3v1', 'r4v1', 'rmt1v1', 'rmt4v1', 'rlt1v1', 'rlt4v1', 'lmiv1', 'lmav1', 'rmagv1'],
            'v2': ['mbv2', 'mhv2', 'lmsov2', 'lth1v2', 'lth2v2', 'lmov2', 'lmuv2', 'r1v2', 'r11v2', 'r2v2', 'r3v2', 'r4v2', 'rmt1v2', 'rmt4v2', 'rlt1v2', 'rlt4v2', 'lmiv2', 'lmav2', 'rmagv2'],
            'ri':['b_s', 'h_s', 'r_sn', 'r_zk'],
            'stator': ['b_nng', 'b_nzk',  'h_n', 'r_ng', 'bhp', 'hhp', 'rhp']
        }

        # Add nodes with their types and features
        for node_type, nodes in node_types.items():
            for node in nodes:
                features = [params_dict.get(param, 0) for param in node_features[node_type]]
                G.add_node(node, node_type=node_type, features=features)

        # Define edge types and their features
        edge_types = {
            'vm1_vm2': [('v1m1', 'v1m2'), ('v2m1', 'v2m2')],
            'v_ri': [('v1m1', 'ri'), ('v1m2', 'ri'), ('v2m1', 'ri'), ('v2m2', 'ri')],
            'v1_v2': [('v1m2', 'v2m2'), ('v1m1', 'v2m1')],
            'ri_s' : [('ri', 's1'), ('ri', 's2'), ('ri', 's3'), ('ri', 's4'), ('ri', 's5'), ('ri', 's6')],
            's_s': [('s1', 's2'), ('s2', 's3'), ('s3', 's4'), ('s4', 's5'), ('s5', 's6')],
            
        }

        edge_features = {
            'vm1_vm2': ['dsm', 'dsmu', 'ha', 'deg_phi'],
            'v_ri': ['amtr + airgap', 'dsr + airgap'],
            'v1_v2': ['amtr_diff'],
            's_s': ['b_z'],##b_z is 0 for some reason, need to check with Leo
            'ri_s': ['h_zk'],   
        }

        #Add edges with their types and features
                
        for edge_type, edges in edge_types.items():
            for edge in edges:
                features = []
                for feature in edge_features[edge_type]:
                    if edge_type == 'v_ri':
                        if '+' in feature:
                            feat1, feat2 = feature.split('+')
                            value = params_dict.get(f"{feat1.strip()}v1", 0) + params_dict.get(feat2.strip(), 0)
                        else:
                            value = params_dict.get(f"{feature}v1", 0)
                    elif edge_type == 'v1_v2':
                        value = params_dict.get('amtrv2', 0) - params_dict.get('amtrv1', 0)
                    elif edge_type == 's_s' or edge_type == 'ri_s':
                        value = params_dict.get(feature, 0)
                    else:  # vm1_vm2
                        if edge[0].startswith('v1'):
                            value = params_dict.get(f"{feature}v1", 0)
                        elif edge[0].startswith('v2'):
                            value = params_dict.get(f"{feature}v2", 0)
                        else:
                            value = params_dict.get(feature, 0)
                    features.append(value)
                
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
        
        # print(G.nodes)
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
 
    color_map = {'v1': 'red', 'v2': 'blue', 'stator': 'yellow', 'ri': 'green'}
    node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]

    edge_color_map = {'vm1_vm2': 'pink', 'v_ri': 'gray', 
                      'v1_v2': 'green', 'ri_s': 'orange', 's_s': 'purple'}
    edge_colors = [edge_color_map[G[u][v][0]['edge_type']] for u, v in G.edges()]

    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, arrows=True)
   
    nx.draw_networkx_labels(G, pos)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=node_type,
                                  markerfacecolor=color, markersize=10)
                       for node_type, color in color_map.items()]
    plt.legend(handles=legend_elements, title="Node Types")
    plt.title("Heterogeneous Graph Visualization")
    #plt.axis('off')
    plt.tight_layout()
    plt.show()
