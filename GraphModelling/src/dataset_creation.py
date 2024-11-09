from torch_geometric.data import Dataset
import os
import pickle
from tqdm import tqdm
import traceback
from typing import Optional, Callable
from src.data_preprocessing_graph import type_features, heterodata_graph
from src.scaling import graph_scaling_params, scale_graph
from src.graph_creation import create_heterograph, visualize_heterograph
import numpy as np

class HeterogeneousGraphDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None, force_process: bool = False):
        """
        Initialize the heterogeneous graph dataset.
        
        Args:
            root: Root directory where the dataset should be saved
            transform: Optional transform to be applied on each data object
            pre_transform: Optional transform to be applied on each data object before saving
            force_process: If True, process the data even if it exists in processed_dir
        """
        self.root = root
        self.force_process = force_process
        self.edge_types = []
        self.graphs = []  # Raw NetworkX graphs
        self.scaled_graphs = []  # Scaled NetworkX graphs
        self.hetero_data_list = []  # List of HeteroData objects
        self.node_scalers = {}
        self.edge_scalers = {}
        
        super(HeterogeneousGraphDataset, self).__init__(root, transform, pre_transform)
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        if self.force_process or not self._processed_file_exists():
            self.graph_creation()
        else:
            self.graphs = self._load_graphs()
        
        self.graph_scaling()
        self.tensor_graphs()

    @property
    def raw_file_names(self):
        """Get list of raw file names."""
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.xlsx')]

    @property
    def processed_file_names(self):
        """Get list of processed file names."""
        return [f'data_{i}.gpickle' for i in range(len(self.raw_file_names))]

    def _processed_file_exists(self):
        """Check if processed files exist."""
        return all(os.path.exists(os.path.join(self.processed_dir, f)) 
                  for f in self.processed_file_names)

    def _load_graphs(self):
        """Load processed NetworkX graphs."""
        graphs = []
        for file_name in self.processed_file_names:
            file_path = os.path.join(self.processed_dir, file_name)
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
            graphs.append(graph)
        return graphs

    def graph_creation(self):
        """Create and save NetworkX graphs from raw data."""
        print("Starting graph creation")
        self.graphs = []  # Initialize graphs list
        
        y2 = np.load('./data/TabularDataETA.npy')
        
        with tqdm(total=len(self.raw_file_names), desc="Processing files") as pbar:
            for i, raw_path in enumerate(self.raw_paths):
                try:
                    G = create_heterograph(raw_path)  # Your existing function
                    G.graph['eta']=y2[i, :, :]
                    self.graphs.append(G)  # Store graph in memory
                    
                    # Save the graph to processed directory
                    nx_file_path = os.path.join(self.processed_dir, f'data_{i}.gpickle')
                    with open(nx_file_path, 'wb') as f:
                        pickle.dump(G, f)
                        
                except Exception as e:
                    print(f"\nError creating graph file {os.path.basename(raw_path)}: {str(e)}")
                    traceback.print_exc()
                pbar.update(1)
                
        print("Finished graph creation")
        if self.graphs:  # If we have at least one graph
            print('Visualizing sample heterograph')
            visualize_heterograph(self.graphs[0])  # Your existing function
            

    def graph_scaling(self):
        """Scale features across all graphs."""
        print("Starting graph scaling")
        self.scaled_graphs = []  # Initialize scaled graphs list
        
        try:
            # Collect features for scaling
            node_features_by_type, edge_features_by_type = type_features(self.graphs)
            
            # Compute scaling parameters
            self.node_scalers = graph_scaling_params(node_features_by_type)
            self.edge_scalers = graph_scaling_params(edge_features_by_type)
            
            # Scale each graph
            for i, G in enumerate(self.graphs):
                try:
                    scaled_G = scale_graph(G, self.node_scalers, self.edge_scalers)
                    self.scaled_graphs.append(scaled_G)
                except Exception as e:
                    print(f"Error scaling graph {i+1}:")
                    print(str(e))
                    raise e
                    
            print("Finished graph scaling")
            return self.scaled_graphs, self.node_scalers, self.edge_scalers
            
        except Exception as e:
            print("Error in graph scaling:")
            print(str(e))
            raise e

    def tensor_graphs(self):
        """Convert scaled NetworkX graphs to PyTorch Geometric HeteroData objects."""
        print("Starting tensor conversion")
        self.hetero_data_list = []  # Initialize HeteroData list
        
        try:
            for i, graph in enumerate(self.scaled_graphs):
                try:
                    hetero_data = heterodata_graph(graph)  # Your existing function
                    if self.pre_transform is not None:
                        hetero_data = self.pre_transform(hetero_data)
                    self.hetero_data_list.append(hetero_data)
                except Exception as e:
                    print(f"Error converting graph {i+1} to tensor:")
                    print(str(e))
                    raise e
                    
            print("Finished tensor conversion")
            
        except Exception as e:
            print("Error in tensor conversion:")
            print(str(e))
            raise e

    def len(self):
        """Return the number of graphs in the dataset."""
        return len(self.hetero_data_list)

    def get(self, idx):
        """Get a graph from the dataset.
        
        Args:
            idx (int): Index of the graph to get
            
        Returns:
            HeteroData: The requested graph
        """
        data = self.hetero_data_list[idx]
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data
    