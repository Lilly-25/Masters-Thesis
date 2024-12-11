import ast
import networkx as nx
import numpy as np
import openpyxl
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import json


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

        # Load edge dictionaries from the JSON file
        with open('./data/DoubleVGraph.json', 'r') as f:
            graph_dict = json.load(f)

        node_types = graph_dict['node_types']
        node_features = graph_dict['node_features']

        # Add nodes with their types and features
        ##TODO add a check when there are no features
        for node_type, nodes in node_types.items():
            for node in nodes:
                features = []
                for feature in node_features[node_type]:
                    if node.startswith('v1'):
                        key = f"{feature}1"
                        value = params_dict.get(key, 0)
                    elif node.startswith('v2'):
                        key = f"{feature}2"
                        value = params_dict.get(key, 0)
                    elif '0' in feature:
                        value = 0
                    else:
                        value = params_dict.get(feature, 0)
                    features.append(value)
                    
                G.add_node(node, type=node_type, features=features)
        
        edge_types = {
            'a': [('v1m1', 'v1m2'), ('v2m1', 'v2m2')],
            'd': [('v11', 'v12'), ('v21', 'v22'),
            ('v11', 'rr'), ('v12', 'rr'),
            ('v21', 'rr'), ('v22', 'rr'),
            ('v11', 'v1m1'), ('v12', 'v1m2'),
            ('v21', 'v2m1'), ('v22', 'v2m2'),
            ('rr', 's1'), ('rr', 's2'), ('rr', 's3'), ('rr', 's4'), ('rr', 's5'), ('rr', 's6'),
            ('s1', 's1w1'), ('s1', 's1w2'), ('s1', 's1w3'), ('s1', 's1w4'),
            ('s2', 's2w1'), ('s2', 's2w2'), ('s2', 's2w3'), ('s2', 's2w4'),
            ('s3', 's3w1'), ('s3', 's3w2'), ('s3', 's3w3'), ('s3', 's3w4'),
            ('s4', 's4w1'), ('s4', 's4w2'), ('s4', 's4w3'), ('s4', 's4w4'),
            ('s5', 's5w1'), ('s5', 's5w2'), ('s5', 's5w3'), ('s5', 's5w4'),
            ('s6', 's6w1'), ('s6', 's6w2'), ('s6', 's6w3'), ('s6', 's6w4'),
            ('s1w1', 's1w2'), ('s1w2', 's1w3'), ('s1w3', 's1w4'),
            ('s2w1', 's2w2'), ('s2w2', 's2w3'), ('s2w3', 's2w4'),
            ('s3w1', 's3w2'), ('s3w2', 's3w3'), ('s3w3', 's3w4'),
            ('s4w1', 's4w2'), ('s4w2', 's4w3'), ('s4w3', 's4w4'),
            ('s5w1', 's5w2'), ('s5w2', 's5w3'), ('s5w3', 's5w4'),
            ('s6w1', 's6w2'), ('s6w2', 's6w3'), ('s6w3', 's6w4'),#based on simq decide nodes
            ('s1', 'ra'), ('s2', 'ra'), ('s3', 'ra'), ('s4', 'ra'), ('s5', 'ra'), ('s6', 'ra'),
            ('o', 'rr'), ('o', 'ra')]
        }

        edge_features = {
            'a': ['deg_phi'],
            # 'd': ['dsm', 'dsmu',
            # 'amtrv1', 'dsrv1',
            # 'dsrv2',
            # 'lmav1', 'lmiv1', 'lmov1', 'lmuv1',
            # 'lmav2', 'lmiv2', 'lmov2', 'lmuv2',
            # 'airgap',
            # 'dhphp',  
            # 'dhpng',
            # 'r_a - (r_i + h_n  + h_zk)', 
            # 'r_i-airgap','r_a']
            'd1': [
            'dsrv2',
            'airgap',
            'dhphp',  
            'dhpng'],
            'd2': ['dsm', 'dsmu',
            'amtrv1', 'dsrv1',
            'r_a - (r_i + h_n  + h_zk)', 
            'r_i-airgap','r_a'],
            'd4': [
            'lmav1', 'lmiv1', 'lmov1', 'lmuv1',
            'lmav2', 'lmiv2', 'lmov2', 'lmuv2',
            ],
        }
        
        def convert_str_to_tuple(d):
            converted_dict = {}
            for k, v in d.items():
                # Remove the parentheses and split by comma
                k = k.strip('()').replace("'", "").split(',')
                # Create tuple from the two elements
                tuple_key = (k[0].strip(), k[1].strip())
                converted_dict[tuple_key] = v
            return converted_dict

        # Convert the edge dictionaries
        edge_a = convert_str_to_tuple(graph_dict['edge_a'])
        edge_d1 = convert_str_to_tuple(graph_dict['edge_d1'])
        edge_d2 = convert_str_to_tuple(graph_dict['edge_d2'])
        edge_d4 = convert_str_to_tuple(graph_dict['edge_d4'])
        edge_d2_calc = convert_str_to_tuple(graph_dict['edge_d2_calc'])
        edge_d4_calc = convert_str_to_tuple(graph_dict['edge_d4_calc'])

        # edge_a = graph_dict['edge_a']
        # edge_d1 = graph_dict['edge_d1']
        # edge_d2 = graph_dict['edge_d2']
        # edge_d4 = graph_dict['edge_d4']
        # edge_d2_calc = graph_dict['edge_d2_calc']
        # edge_d4_calc = graph_dict['edge_d4_calc']
        

        for edge, desc in edge_a.items():
            features = []
            edge_type='a'
            for feature in desc:
                value = params_dict.get(feature, 0)
                features.append(value)

            G.add_edge(edge[0], edge[1], type=edge_type, features=features)        
        for edge, desc in edge_d1.items():
            features = []
            edge_type='d1'
            for feature in desc:
                value = params_dict.get(feature, 0)
                features.append(value)

            G.add_edge(edge[0], edge[1], type=edge_type, features=features)    
            
        for edge, desc in edge_d2.items():
            features = []
            edge_type='d2'
            for feature in desc:
                value = params_dict.get(feature, 0)
                features.append(value)
                
            G.add_edge(edge[0], edge[1], type=edge_type, features=features)  

        for edge, desc in edge_d4.items():
            features = []
            edge_type='d4'
            for feature in desc:
                value = params_dict.get(feature, 0)
                features.append(value)
                
            G.add_edge(edge[0], edge[1], type=edge_type, features=features)  
            
        for edge, desc in edge_d2_calc.items():
            features = []
            edge_type='d2'
            value = params_dict.get('r_a', 0)-(params_dict.get('r_i', 0)+params_dict.get('h_n', 0)+params_dict.get('h_zk', 0))
            features.append(value)
            G.add_edge(edge[0], edge[1], type=edge_type, features=features)

        for edge, desc in edge_d4_calc.items():
            features = []
            edge_type='d4'
            value = params_dict.get('r_i', 0)-params_dict.get('airgap', 0)
            features.append(value)
            G.add_edge(edge[0], edge[1], type=edge_type, features=features)
 
            
    

        #G.graph['r_a'] = params_dict.get('r_a', 0)
        #G.graph['r_i'] = params_dict.get('r_i', 0)##Was removed recenntly<!!

        # Read labels from 'Mgrenz' sheet
        wb = openpyxl.load_workbook(file_path)
        sheet_mgrenz = wb['Mgrenz']
        mgrenz_values = [cell.value for cell in sheet_mgrenz[1] if cell.value is not None]

        G.graph['mgrenz_values'] = mgrenz_values ##Renamed from mgrenz_values to mgrenz
        # G.graph['eta'] = mgrenz_values
        
        # print(f"File name{os.path.basename(file_path)}")
        # print("\nNode features:")
        
        # print(G.nodes)
        # for node, data in G.nodes(data=True):
        #     print(f"Node {node} ({data['node_type']}): {data['features']}")
        
        # print("\nEdge features:")
        # for u, v, data in G.edges(data=True):
        #     print(f"Edge ({u}, {v}) ({data['edge_type']}): {data['features']}")
        
        # print(f"\nmgrenz_values: {G.graph['mgrenz_values'][:5]}... (showing first 5)")
        #print(G.edges)
        
        return G
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return None
    
def hierarchical_layout(G):
    layers = ['vm', 'v', 'r', 's', 'sw']
    layer_heights = {layer: i for i, layer in enumerate(layers)}
    
    pos = {}
    layer_counts = defaultdict(int)
    
    for node, data in G.nodes(data=True):
        layer = data['type']
        layer_counts[layer] += 1
        y = layer_heights[layer]
        x = layer_counts[layer] - 1
        pos[node] = (x, y)
    
    # Normalize x positions
    for layer in layers:
        nodes = [n for n in pos if G.nodes[n]['type'] == layer]
        count = len(nodes)
        for i, node in enumerate(nodes):
            x, y = pos[node]
            pos[node] = ((i - (count - 1) / 2) / max(1, count - 1), y)
    
    return pos

def visualize_heterograph(G):
    plt.figure(figsize=(40, 38))
 
    color_map = {'v': '#FF6B6B', 'vm': '#FFD93D', 'r': '#4D96FF', 's': '#F9ED69', 
                 'sw': '#6BCB77'}
    node_colors = [color_map[G.nodes[node]['type']] for node in G.nodes()]
    
    edge_color_map = {'a': '#FF92A5', 'd1': '#D3D3D3', 'd2': '#FF5733', 'd4': '#C70039'}
    
    pos = hierarchical_layout(G)
    
    # Draw edges
    for (u, v, data) in G.edges(data=True):
        edge_type = data['type']
        color = edge_color_map[edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, 
                               width=1.2, alpha=0.6, 
                               connectionstyle='arc3,rad=0.1',
                               arrows=False)
    
    # Draw edge labels
    edge_labels = {(u, v): data['type'] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=26, font_weight='bold')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5500, alpha=0.8, edgecolors='black', linewidths=2)
    
    # Add labels (node names)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=23, font_weight='bold')
    
    # Create legend elements for nodes
    node_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=node_type,
                                       markerfacecolor=color, markersize=15, markeredgecolor='black', markeredgewidth=1)
                            for node_type, color in color_map.items()]
    
    # Create legend elements for edges
    edge_legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=edge_type)
                            for edge_type, color in edge_color_map.items()]
    
    # # Combine node and edge legend elements
    # all_legend_elements = node_legend_elements + edge_legend_elements
    
    
    # plt.legend(handles=all_legend_elements, title="Nodes and Edge Types", 
    #            loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=34,  
    #            title_fontsize=30, ncol=10, columnspacing=1.5, handletextpad=1.2) 
    
    # Create the first legend for nodes
    node_legend = plt.legend(
        handles=node_legend_elements,
        title="Node Types",
        loc='lower right',
        bbox_to_anchor=(0.5, -0.025),  # Position nodes legend
        fontsize=30,
        title_fontsize=34,
        ncol=5,
        columnspacing=1.5,
        handletextpad=1.2
    )

    # Add the first legend to the plot
    plt.gca().add_artist(node_legend)

    # Create the second legend for edges
    plt.legend(
        handles=edge_legend_elements,
        title="Edge Types",
        loc='lower left',
        bbox_to_anchor=(0.5, -0.025),  # Position edges legend below nodes legend
        fontsize=30,
        title_fontsize=34,
        ncol=5,
        columnspacing=1.5,
        handletextpad=1.2
    )
    
    # plt.title("Visualisation of an Electric Motor as a Heterogeneous Graph", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./data/DoubleVGraph.png')
    plt.show()