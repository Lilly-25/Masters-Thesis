from src.dataset_creation import HeterogeneousGraphDataset
import os
# Use force_process as False else it takes too much time to process the files again
print("Creating dataset...")
directory = os.path.join(os.getcwd(), 'data')
dataset = HeterogeneousGraphDataset(root=directory, force_process=True)
print("Dataset creation completed.")