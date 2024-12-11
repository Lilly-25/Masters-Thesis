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
        with open('/home/k64889/Masters-Thesis/data/DoubleVGraphTemp.json', 'r') as f:
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
        edge_d1 = convert_str_to_tuple(graph_dict['edge_d1'])
        edge_d2 = convert_str_to_tuple(graph_dict['edge_d2'])
        edge_d1_calc_airgap = convert_str_to_tuple(graph_dict['edge_d1_calc_airgap'])
        edge_d1_calc_ss = convert_str_to_tuple(graph_dict['edge_d1_calc_ss'])
            
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

        for edge, desc in edge_d1_calc_airgap.items():
            features = []
            edge_type='d1'
            value = params_dict.get('r_i', 0)-params_dict.get('airgap', 0)
            features.append(value)
            G.add_edge(edge[0], edge[1], type=edge_type, features=features)
            
        for edge, desc in edge_d1_calc_ss.items():
            features = []
            edge_type='d1'
            value = params_dict.get('r_a', 0)-(params_dict.get('r_i', 0)+params_dict.get('h_n', 0)+params_dict.get('h_zk', 0))
            features.append(value)
            G.add_edge(edge[0], edge[1], type=edge_type, features=features)
 
        # Read labels from 'Mgrenz' sheet
        wb = openpyxl.load_workbook(file_path)
        sheet_mgrenz = wb['Mgrenz']
        mgrenz_values = [cell.value for cell in sheet_mgrenz[1] if cell.value is not None]

        G.graph['mgrenz_values'] = mgrenz_values 
        
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
    plt.figure(figsize=(24, 20))
 
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
    plt.savefig('.data/temp/DoubleVGraph.png')
    plt.show()