from src.table_creation import create_tabular_data
import os
import pandas as pd
from tqdm import tqdm

directory = '/home/k64889/Masters-Thesis/data/temp'

df_inputs=pd.DataFrame()
df_targets=pd.DataFrame()

for filename in tqdm(os.listdir(directory)):
    file_path = os.path.join(directory, filename)
    df_partial_inputs, df_partial_targets = create_tabular_data(file_path)
    
    if df_partial_inputs is not None:
        df_inputs = pd.concat([df_inputs, df_partial_inputs])
        
    if df_partial_targets is not None:
        df_targets = pd.concat([df_targets, df_partial_targets])
        
df_inputs.to_csv('./data/TempTabularData2VInputs.csv', index=True)
df_targets.to_csv('./data/TempTabularData2VTargets.csv', index=True)