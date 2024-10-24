import pandas as pd
from src.table_creation import create_tabular_data
from src.data_preprocessing_tabular import data_prep_eta_grid
from src.utils import remove_faulty_files
import os
from tqdm import tqdm

directory = '/home/k64889/Masters-Thesis/data/raw'
# directory = '/home/k64889/Masters-Thesis/data/Tests'

remove_faulty_files(directory)

df_inputs=pd.DataFrame()
df_targets=pd.DataFrame()

for filename in tqdm(os.listdir(directory)):
    file_path = os.path.join(directory, filename)
    df_partial_inputs, df_partial_targets = create_tabular_data(file_path, purpose='train')
    
    if df_partial_inputs is not None:
        df_inputs = pd.concat([df_inputs, df_partial_inputs])
        
    if df_partial_targets is not None:
        df_targets = pd.concat([df_targets, df_partial_targets])


df_inputs.to_csv('./data/TabularDataInputs.csv', index=True)
df_targets.to_csv('./data/TabularDataY1Targets.csv', index=True)


# print(f"Total input rows: {df_inputs.shape[0]}")
# print(f"Total target rows: {df_targets.shape[0]}")

# data_prep_eta_grid()

# print("Data preprocessing complete")

