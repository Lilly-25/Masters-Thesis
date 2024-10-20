import networkx as nx
import numpy as np
import openpyxl
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict


def create_heterograph(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='input_data', header=None)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        params_dict = {}
        for _, row in df.iterrows():
            param, value = row[0], row[1]
            if pd.notna(param) and pd.notna(value):
                try:
                    params_dict[param] = float(value)
                except ValueError:
                    params_dict[param] = value

        G = nx.MultiDiGraph()  # Use MultiDiGraph for heterogeneous graph

        # Define node types and their features
        node_types = {
            #'v': ['v1m1', 'v1m2', 'v2m1', 'v2m2'],##Automate to take based on topology type V1, V3?
            'v1': ['v11', 'v12'],##Automate to take based on topology type V1, V3?
            'v2': ['v21', 'v22'],##Automate to take based on topology type V1, V3?
            'v1m': ['v1m1', 'v1m2'],##Automate to take based on topology type V1, V3?
            'v2m': ['v2m1', 'v2m2'],##Automate to take based on topology type V1, V3?
            'r':['rr'],
            's': ['s1', 's2', 's3', 's4', 's5', 's6'],##Automate to take from N?
            'sw':['s1w1', 's1w2', 's1w3', 's1w4',
                  's2w1', 's2w2', 's2w3', 's2w4', 
                  's3w1', 's3w2', 's3w3', 's3w4', 
                  's4w1', 's4w2', 's4w3', 's4w4', 
                  's5w1', 's5w2', 's5w3', 's5w4', 
                  's6w1', 's6w2', 's6w3', 's6w4'],
        }

        #Automate to take based on naming convention maybe?
        node_features = {
            'v1': ['lmsov', 'lth1v', 'lth2v', 'r1v', 'r11v', 'r2v', 'r3v', 'r4v', 'rmt1v', 'rmt4v', 'rlt1v', 'rlt4v', 'hav'],
            'v2': ['lmsov', 'lth1v', 'lth2v', 'r1v', 'r11v', 'r2v', 'r3v', 'r4v', 'rmt1v', 'rmt4v', 'rlt1v', 'rlt4v', 'hav'],
            'v1m': ['mbv', 'mhv', 'rmagv'],
            'v2m': ['mbv', 'mhv', 'rmagv'],
            'r':['r_i - airgap'],
            's': ['b_nng', 'b_nzk', 'b_s', 'h_n','h_s', 'r_sn', 'r_zk', 'r_ng', 'h_zk'], #h_zk--constant? to be added?'TODO Check###Added recently
            'sw':['bhp', 'hhp', 'rhp']
        }

        # Add nodes with their types and features
        
        ##TODO add a check when there are no features
        for node_type, nodes in node_types.items():
            for node in nodes:
                features = []
                for feature in node_features[node_type]:
                    if node_type.startswith('v1'):
                        key = f"{feature}1"
                        value = params_dict.get(key, 0)
                    elif node_type.startswith('v2'):
                        key = f"{feature}2"
                        value = params_dict.get(key, 0)
                    elif '-' in feature:
                        feat1, feat2 = feature.split('-')
                        value = params_dict.get(feat1.strip(), 0) - params_dict.get(feat2.strip(), 0)
                    else:
                        value = params_dict.get(feature, 0)
                    features.append(value)
                    
                G.add_node(node, node_type=node_type, features=features)

        # Define edge types and their features
        edge_types = {
            'v_v': [('v11', 'v12'), ('v21', 'v22')],
            'v1_v2': [('v11', 'v21'), ('v12', 'v22')],
            'v1_r': [('v11', 'rr'), ('v12', 'rr')],
            'v2_r': [('v21', 'rr'), ('v22', 'rr')],
            'm_m': [('v1m1', 'v1m2'), ('v2m1', 'v2m2')],
            'v1_m': [('v11', 'v1m1'), ('v12', 'v1m2')],
            'v2_m': [('v21', 'v2m1'), ('v22', 'v2m2')],
            'r_s' : [('rr', 's1'), ('rr', 's2'), ('rr', 's3'), ('rr', 's4'), ('rr', 's5'), ('rr', 's6')],
            's_sw': [('s1', 's1w1'), ('s1', 's1w2'), ('s1', 's1w3'), ('s1', 's1w4'),
                     ('s2', 's2w1'), ('s2', 's2w2'), ('s2', 's2w3'), ('s2', 's2w4'),
                     ('s3', 's3w1'), ('s3', 's3w2'), ('s3', 's3w3'), ('s3', 's3w4'),
                     ('s4', 's4w1'), ('s4', 's4w2'), ('s4', 's4w3'), ('s4', 's4w4'),
                     ('s5', 's5w1'), ('s5', 's5w2'), ('s5', 's5w3'), ('s5', 's5w4'),
                     ('s6', 's6w1'), ('s6', 's6w2'), ('s6', 's6w3'), ('s6', 's6w4')],
            'sw_sw': [('s1w1', 's1w2'), ('s1w2', 's1w3'), ('s1w3', 's1w4'),
                      ('s2w1', 's2w2'), ('s2w2', 's2w3'), ('s2w3', 's2w4'),
                      ('s3w1', 's3w2'), ('s3w2', 's3w3'), ('s3w3', 's3w4'),
                      ('s4w1', 's4w2'), ('s4w2', 's4w3'), ('s4w3', 's4w4'),
                      ('s5w1', 's5w2'), ('s5w2', 's5w3'), ('s5w3', 's5w4'),
                      ('s6w1', 's6w2'), ('s6w2', 's6w3'), ('s6w3', 's6w4')],#based on simq decide nodes
        }

        edge_features = {
            'v_v': ['dsm', 'dsmu'],#Removed hav from edges
            'v1_v2': ['amtr_diff'],
            'v1_r': ['amtrv1', 'dsrv1'],
            'v2_r': ['dsrv2'],
            'm_m':['deg_phi'],
            'v1_m':['lmav1', 'lmiv1', 'lmov1', 'lmuv1'],
            'v2_m':['lmav2', 'lmiv2', 'lmov2', 'lmuv2'],
            'r_s':['airgap'],
            #'s_s': ['b_z'],##b_z is 0 for some reason, need to check with Leo
            's_sw': ['dhphp'],  
            'sw_sw': ['dhpng'] 
        }

        #Add edges with their types and features
                
        for edge_type, edges in edge_types.items():
            for edge in edges:
                features = []
                for feature in edge_features[edge_type]:
                    if edge_type == 'v1_v2':
                        value = params_dict.get('amtrv2', 0) - params_dict.get('amtrv1', 0)
                    elif edge_type == 'v_v' or edge_type == 'm_m':  # v_v case
                        if edge[0].startswith('v1'):
                            value = params_dict.get(f"{feature}v1", 0)
                        elif edge[0].startswith('v2'):
                            value = params_dict.get(f"{feature}v2", 0)
                        else:
                            value = params_dict.get(feature, 0)
                    else:
                        value = params_dict.get(feature, 0)
                    features.append(value)
                
                G.add_edge(edge[0], edge[1], edge_type=edge_type, features=features)

        G.graph['r_a'] = params_dict.get('r_a', 0)
        #G.graph['r_i'] = params_dict.get('r_i', 0)##Was removed recenntly<!!

        # Read labels from 'Mgrenz' sheet
        wb = openpyxl.load_workbook(file_path)
        sheet_mgrenz = wb['Mgrenz']
        mgrenz_values = [cell.value for cell in sheet_mgrenz[1] if cell.value is not None]

        G.graph['mgrenz_values'] = mgrenz_values
        
        # print(f"File name{os.path.basename(file_path)}")
        # print("\nNode features:")
        
        # print(G.nodes)
        # for node, data in G.nodes(data=True):
        #     print(f"Node {node} ({data['node_type']}): {data['features']}")
        
        # print("\nEdge features:")
        # for u, v, data in G.edges(data=True):
        #     print(f"Edge ({u}, {v}) ({data['edge_type']}): {data['features']}")
        
        # print(f"\nmgrenz_values: {G.graph['mgrenz_values'][:5]}... (showing first 5)")
        # print(f"Global attributes: r_a={G.graph['r_a']}, r_i={G.graph['r_i']}")
        #print(G.edges)
        
        return G
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return None
    
def hierarchical_layout(G):
    layers = ['v2m', 'v2', 'v1', 'v1m', 'r', 's', 'sw']
    layer_heights = {layer: i for i, layer in enumerate(layers)}
    
    pos = {}
    layer_counts = defaultdict(int)
    
    for node, data in G.nodes(data=True):
        layer = data['node_type']
        layer_counts[layer] += 1
        y = layer_heights[layer]
        x = layer_counts[layer] - 1
        pos[node] = (x, y)
    
    # Normalize x positions
    for layer in layers:
        nodes = [n for n in pos if G.nodes[n]['node_type'] == layer]
        count = len(nodes)
        for i, node in enumerate(nodes):
            x, y = pos[node]
            pos[node] = ((i - (count - 1) / 2) / max(1, count - 1), y)
    
    return pos

def visualize_heterograph(G):
    plt.figure(figsize=(24, 20))
 
    color_map = {'v1': '#FF6B6B', 'v2': '#FFD93D', 'r': '#4D96FF', 's': '#F9ED69', 
                 'sw': '#6BCB77', 'v1m': '#9B59B6', 'v2m': '#F8C471'}
    node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]
    
    edge_color_map = {'v_v': '#FF92A5', 'v1_v2': '#D3D3D3', 'v1_r': '#C39BD3',
                      'v2_r': '#ABEBC6', 'm_m': '#8E44AD', 'v1_m': '#5DADE2', 
                      'v2_m': '#F1948A', 'r_s': '#F0B27A', 's_sw': '#BDC3C7', 'sw_sw': '#8E44AD'}
    
    pos = hierarchical_layout(G)
    
    # Draw edges
    for (u, v, data) in G.edges(data=True):
        edge_type = data['edge_type']
        color = edge_color_map[edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, 
                               width=1.2, alpha=0.6, 
                               connectionstyle='arc3,rad=0.1',
                               arrows=False)
    
    # Draw edge labels
    edge_labels = {(u, v): data['edge_type'] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.8, edgecolors='black', linewidths=2)
    
    # Add labels (node names)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Create legend elements for nodes
    node_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=node_type,
                                       markerfacecolor=color, markersize=15, markeredgecolor='black', markeredgewidth=1)
                            for node_type, color in color_map.items()]
    
    # Create legend elements for edges
    edge_legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=edge_type)
                            for edge_type, color in edge_color_map.items()]
    
    # Combine node and edge legend elements
    all_legend_elements = node_legend_elements + edge_legend_elements
    
    # Add legend to the plot
    plt.legend(handles=all_legend_elements, title="Node and Edge Types", 
               loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
               ncol=2, columnspacing=1, handletextpad=1)
    
    plt.title("Visualisation of an Electric Motor as a Heterogeneous Graph", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()