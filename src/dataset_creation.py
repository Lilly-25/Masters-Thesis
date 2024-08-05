import os
import torch
from torch_geometric.data import Dataset
import pandas as pd
from tqdm.notebook import tqdm
import traceback
from src.graph_creation import create_heterograph, visualize_heterograph
from src.data_preprocessing import pyg_graph

class HeterogeneousGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, force_process=False):
        self.root = root
        self.force_process = force_process
        super(HeterogeneousGraphDataset, self).__init__(root, transform, pre_transform)
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        if self.force_process or not self._processed_file_exists:
            self.process()
        else:
            self.data_list = self._load_processed_files()

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.xlsx')]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.raw_file_names))]

    @property
    def _processed_file_exists(self):
        return all(os.path.exists(os.path.join(self.processed_dir, f)) for f in self.processed_file_names)

    def _load_processed_files(self):
        data_list = []
        for file_name in self.processed_file_names:
            file_path = os.path.join(self.processed_dir, file_name)
            data = torch.load(file_path)
            data_list.append(data)
        return data_list
    

    def process(self):
        self.data_list = []  # Clear the list before processing
        print("Starting process method")
        # Create a single progress bar for all files
        with tqdm(total=len(self.raw_file_names), desc="Processing files") as pbar:
            for i, raw_path in enumerate(self.raw_paths):
                try:
                    nx_graph = create_heterograph(raw_path)
                    if nx_graph is not None:
                        try:
                            data = pyg_graph(nx_graph)
                            torch.save(data, os.path.join(self.processed_dir, self.processed_file_names[i]))
                            self.data_list.append(data)
                        except Exception as e:
                            print(f"\nError converting graph to PyG for file {os.path.basename(raw_path)}: {str(e)}")
                            traceback.print_exc()
                except Exception as e:
                    print(f"\nError processing file {os.path.basename(raw_path)}: {str(e)}")
                    traceback.print_exc()
                
                # Update the progress bar
                pbar.update(1)
        print("Finished process method")
        print('Visualizing sample heterograph')
        visualize_heterograph(nx_graph)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if hasattr(self, 'data_list') and self.data_list:
            return self.data_list[idx]
        else:
            file_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
            return torch.load(file_path)
        
    def inspect_dataset(self):
        print(f"Dataset contains {len(self.data_list)} graphs")

        # Inspect the first graph in detail
        sample_graph = self.data_list[0]
        print("\nDetailed information for the first graph:")

        # Node types and features
        print("\nNode Types and Features:")
        for node_type in sample_graph.node_types:
            num_nodes = sample_graph[node_type].num_nodes
            feature_dim = sample_graph[node_type].num_features
            print(f"  {node_type}: {num_nodes} nodes, {feature_dim} features")
            if num_nodes > 0:
                print(f"    Sample features: {sample_graph[node_type].x[0][:5]}...")  # Print first 5 features of first node

        # Edge types and structure
        print("\nEdge Types and Structure:")
        for edge_type in sample_graph.edge_types:
            num_edges = sample_graph[edge_type].num_edges
            print(f"  {edge_type}: {num_edges} edges")
            if num_edges > 0:
                print(f"    Edge indices: {sample_graph[edge_type].edge_index[:, :5]}")  # Print first 5 edge indices
                if hasattr(sample_graph[edge_type], 'edge_attr'):
                    edge_attr_dim = sample_graph[edge_type].edge_attr.size(1)
                    print(f"    Edge features: {edge_attr_dim} dimensions")
                    print(f"    Sample edge features: {sample_graph[edge_type].edge_attr[0][:5]}...")  # Print first 5 features of first edge

        # Global attributes
        if hasattr(sample_graph, 'global_attr'):
            print("\nGlobal Attributes:")
            print(f"  {sample_graph.global_attr}")

        # Labels
        if hasattr(sample_graph, 'y'):
            print("\nLabels:")
            print(f"  Shape: {sample_graph.y.shape}")
            print(f"  Sample labels: {sample_graph.y[:5]}...")  # Print first 5 labels

        # Additional statistics
        print("\nAdditional Statistics:")
        total_nodes = sum(sample_graph[nt].num_nodes for nt in sample_graph.node_types)
        total_edges = sum(sample_graph[et].num_edges for et in sample_graph.edge_types)
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total edges: {total_edges}")


