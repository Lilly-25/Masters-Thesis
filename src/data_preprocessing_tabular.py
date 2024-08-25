import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split

def dataset_creation():
    df_inputs=pd.read_csv('./data/TabularData2VInputs.csv')
    df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    df_targets=pd.read_csv('./data/TabularData2VTargets.csv')
    df_targets.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    filenames = df_inputs['filename'].values##incase we need to refer the index later on
    x = df_inputs.drop('filename', axis=1).values
    y1 = df_targets.drop('filename', axis=1).values
    
    # # Create scalers
    # scaler_x = MinMaxScaler()##incase we need t do different scaling..std makes more sense for y1, 
    # scaler_y1 = MinMaxScaler()##incase we need t do different scaling..min max makes more sense for y1, 

    # # Normalize the data
    # x_normalized = scaler_x.fit_transform(x)
    # y1_normalized = scaler_y1.fit_transform(y1)

    # # Perform train-test split
    # x_train, x_temp, y1_train, y1_temp = train_test_split(
    #     x_normalized, y1_normalized, test_size=0.2, random_state=42
    # )

    # x_val, x_test, y1_val, y1_test = train_test_split(
    #     x_temp, y1_temp, test_size=0.2, random_state=42
    # )

    # Perform train-test split
    x_train, x_temp, y1_train, y1_temp, filenames_train, filenames_temp = train_test_split(
        x, y1, filenames, test_size=0.2, random_state=42
    )
    x_val, x_test, y1_val, y1_test, filenames_val, filenames_test = train_test_split(
        x_temp, y1_temp, filenames_temp, test_size=0.2, random_state=42
    )

    # Convert test sets back to DataFrames
    df_x_test = pd.DataFrame(x_test, columns=df_inputs.columns[1:], index=filenames_test)
    df_y1_test = pd.DataFrame(y1_test, columns=df_targets.columns[1:], index=filenames_test)
    # Create scalers
    scaler_x = MinMaxScaler()##incase we need t do different scaling..std makes more sense for y1, 
    scaler_y1 = MinMaxScaler()##incase we need t do different scaling..min max makes more sense for y1, 

    # Normalize the data
    x_train = scaler_x.fit_transform(x_train)
    y1_train = scaler_y1.fit_transform(y1_train)
    x_val = scaler_x.fit_transform(x_val)
    y1_val = scaler_y1.fit_transform(y1_val)

    #
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(torch.FloatTensor(x_train), 
                                  torch.FloatTensor(y1_train))#TODO test y1 to be integer and not float
    val_dataset = TensorDataset(torch.FloatTensor(x_val), 
                                  torch.FloatTensor(y1_val))#TODO test y1 to be integer and not float
    # test_dataset = TensorDataset(torch.FloatTensor(x_test), 
    #                              torch.FloatTensor(y1_test))#TODO test y1 to be integer and not float

    input_size = x_train.shape[1]
    output_y1_size = y1_train.shape[1]
  
    return train_dataset, val_dataset, input_size, output_y1_size, df_x_test, df_y1_test

# def dataset_creation():
#     df_inputs=pd.read_csv('./data/TempTabularData2VInputs.csv')
#     df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

#     df_targets=pd.read_csv('./data/TempTabularData2VTargets.csv')
#     df_targets.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

#     filenames = df_inputs['filename'].values
#     x = df_inputs.drop('filename', axis=1).values
#     y1 = df_targets.drop('filename', axis=1).values
    
#     y2 = []
#     max_rows, max_cols = 0, 0
    
#     for filename in filenames:
#         y2_file = pd.read_csv(f'./data/ETAgrid/{filename}.csv')
#         max_rows = max(max_rows, y2_file.shape[0])
#         max_cols = max(max_cols, y2_file.shape[1])##really not needded coz it always gonna be 191
    
#     # Second pass: pad arrays to the same size
#     for filename in filenames:
#         y2_file = pd.read_csv(f'./data/ETAgrid/{filename}.csv')
#         padded = np.zeros((max_rows, max_cols))
#         padded[:y2_file.shape[0], :y2_file.shape[1]] = y2_file.values
#         y2.append(padded)
    
#     y2 = np.array(y2)

  
#     # Create scalers
#     scaler_x = MinMaxScaler()##incase we need t do different scaling..std makes more sense for y1, 
#     scaler_y1 = MinMaxScaler()##incase we need t do different scaling..min max makes more sense for y1, 
#     scaler_y2 = MinMaxScaler()

#     # Normalize the data
#     x_normalized = scaler_x.fit_transform(x)
#     y1_normalized = scaler_y1.fit_transform(y1)

#     # For y2, we need to reshape it to 2D before scaling, then reshape back to 3D
#     y2_reshaped = y2.reshape(y2.shape[0], -1)  # Flatten to 2D
#     y2_normalized = scaler_y2.fit_transform(y2_reshaped)
#     y2_normalized = y2_normalized.reshape(y2.shape)  # Reshape back to 3D


#     # Perform train-test split
#     x_train, x_temp, y1_train, y1_temp, y2_train, y2_temp = train_test_split(
#         x_normalized, y1_normalized, y2_normalized, test_size=0.2, random_state=42
#     )

#     x_val, x_test, y1_val, y1_test, y2_val, y2_test = train_test_split(
#         x_temp, y1_temp, y2_temp, test_size=0.2, random_state=42
#     )
    
#     # Create PyTorch datasets
#     train_dataset = TensorDataset(torch.FloatTensor(x_train), 
#                                   torch.FloatTensor(y1_train), 
#                                   torch.FloatTensor(y2_train))
#     val_dataset = TensorDataset(torch.FloatTensor(x_val), 
#                                   torch.FloatTensor(y1_val), 
#                                   torch.FloatTensor(y2_val))
#     test_dataset = TensorDataset(torch.FloatTensor(x_test), 
#                                  torch.FloatTensor(y1_test), 
#                                  torch.FloatTensor(y2_test))



#     return train_dataset, val_dataset, test_dataset