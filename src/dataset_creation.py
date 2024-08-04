import os
import torch
from torch_geometric.data import Dataset, HeteroData
import pandas as pd
from tqdm import tqdm
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

