from src.dataset_creation import HeterogeneousGraphDataset
# Use force_process as False else it takes too much time to process the files again
print("Creating dataset...")
dataset = HeterogeneousGraphDataset(root='/home/k64889/Masters-Thesis/data', force_process=True)
print("Dataset creation completed.")