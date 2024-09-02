import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

def data_prep(test_size):
    
    df_inputs=pd.read_csv('./data/TabularData2VInputs.csv')
    df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    df_targets=pd.read_csv('./data/TabularData2VTargets.csv')
    df_targets.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    filenames = df_inputs['filename'].values##incase we need to refer the index later on
    a=df_inputs.drop('filename', axis=1).values
    b=df_targets.drop('filename', axis=1).values
    df_inputs.drop('filename', axis=1, inplace=True)
    df_targets.drop('filename', axis=1, inplace=True)
    
    filenames_test= filenames[-test_size:]
    
    df_x_test = pd.DataFrame(a[-test_size:], columns=df_inputs.columns, index=filenames_test)
    df_y1_test = pd.DataFrame(b[-test_size:], columns=df_targets.columns, index=filenames_test)
    
    df_inputs_train_val = df_inputs[:-test_size]
    df_targets_train_val = df_targets[:-test_size]
    
    x=df_inputs_train_val.values
    y1=df_targets_train_val.values
    
    scaler_x=StandardScaler().fit(x)##Using different scalers for input and outputs, ..is that okay?
    scaler_y1=MinMaxScaler().fit(y1)
    
    ##Also should we fit intially for test but makes no sense coz thats cheating we are not supposed to know the test in real life
    
    ##I want NAN values which are already replaced as 0 to remain 0 should i change em to 1000 after scaling?..
    ##maybe change to a value near the scaled values max or so, coz changing it drasticallly is resulting in worse predictions and not penalising the model as expected
    
    x_normalized=scaler_x.transform(x)
    y1_normalized=scaler_y1.transform(y1)
    
    input_size = x_normalized.shape[1]
    output_y1_size = y1_normalized.shape[1]
  
    return x_normalized, y1_normalized, scaler_x, scaler_y1, input_size, output_y1_size, df_x_test, df_y1_test

# def dataset_creation():
#     df_inputs=pd.read_csv('./data/TabularData2VInputs.csv')
#     df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

#     df_targets=pd.read_csv('./data/TabularData2VTargets.csv')
#     df_targets.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

#     filenames = df_inputs['filename'].values##incase we need to refer the index later on
#     x = df_inputs.drop('filename', axis=1).values
#     y1 = df_targets.drop('filename', axis=1).values

#     #TODO Use cross validation instead of train-test split
#     x_train, x_temp, y1_train, y1_temp, filenames_train, filenames_temp = train_test_split(
#         x, y1, filenames, test_size=0.2, random_state=42
#     )
#     x_val, x_test, y1_val, y1_test, filenames_val, filenames_test = train_test_split(
#         x_temp, y1_temp, filenames_temp, test_size=0.2, random_state=42
#     )

#     # Convert test sets back to DataFrames, coz we dont want it scaled and simply want to remember what was not used for train/validation
#     df_x_test = pd.DataFrame(x_test, columns=df_inputs.columns[1:], index=filenames_test)
#     df_y1_test = pd.DataFrame(y1_test, columns=df_targets.columns[1:], index=filenames_test)
    
#     # Normalisation
#     scaler_x = MinMaxScaler()##incase we need t do different scaling..std makes more sense for x, 
#     scaler_y1 = MinMaxScaler()##incase we need t do different scaling..min max makes more sense for y1, 

#     # Normalize the data
#     x_train = scaler_x.fit_transform(x_train)
#     y1_train = scaler_y1.fit_transform(y1_train)
#     x_val = scaler_x.fit_transform(x_val)
#     y1_val = scaler_y1.fit_transform(y1_val)

    
#     # Create PyTorch datasets
#     train_dataset = TensorDataset(torch.FloatTensor(x_train), 
#                                   torch.FloatTensor(y1_train))#TODO test y1 to be integer and not float..No need coz it makes it a classification problem and not a regression problem
#     val_dataset = TensorDataset(torch.FloatTensor(x_val), 
#                                   torch.FloatTensor(y1_val))#TODO test y1 to be integer and not float..No need coz it makes it a classification problem and not a regression problem
    
#     input_size = x_train.shape[1]
#     output_y1_size = y1_train.shape[1]
  
#     return train_dataset, val_dataset, input_size, output_y1_size, df_x_test, df_y1_test

def data_prep_eta_grid():
    df_inputs=pd.read_csv('./data/CompleteTabularDataInputs.csv')
    df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    df_targets=pd.read_csv('./data/CompleteTabularDataTargets.csv')
    df_targets.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    filenames = df_inputs['filename'].values
    
    df_inputs=df_inputs.drop('filename', axis=1)
    df_targets=df_targets.drop('filename', axis=1)
      
    x=df_inputs.values
    y1=df_targets.values
    
    ##I want NAN values which are already replaced as 0 to remain 0 should i change em to 1000 after scaling?
    
    ##############
    
    max_mgrenz=np.max(y1)
    
    max_rows=(max_mgrenz*2)+1
    print(max_rows)
    y2 = []
    # Padding
    for filename in filenames:
        y2_file = pd.read_csv(f'./data/CompleteTabularDataETAgrid/{filename}.csv')
        # Convert DataFrame to NumPy array and replace NaNs with 1000
        #y2_values = np.nan_to_num(y2_file.values, nan=1000)
        y2_values = y2_file.values
        # Replace 0s with nans
        #y2_values[y2_values == 0] = 1000
        y2_values[y2_values == 0] = np.nan
        padded = np.full((max_rows, 191),  np.nan, dtype=float)
        #padded = np.full((max_rows, 191), 1000, dtype=y2_values.dtype)
        padded[:y2_file.shape[0], :y2_file.shape[1]] = y2_values
        y2.append(padded)
    
    y2 = np.array(y2)
    
    np.save('./data/CompleteTabularDataETA.npy', y2)

def data_prep_complete(test_size):
    # df_inputs=pd.read_csv('./data/400TabularData2VInputs.csv')
    # df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    # df_targets=pd.read_csv('./data/400TabularData2VTargets.csv')
    # df_targets.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    # filenames = df_inputs['filename'].values
    
    # df_inputs=df_inputs.drop('filename', axis=1)
    # df_targets=df_targets.drop('filename', axis=1)
    
    # df_x_test = df_inputs[-test_size:]
    # df_y1_test = df_targets[-test_size:]
    
    df_inputs=pd.read_csv('./data/TabularData2VInputs.csv')
    df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    df_targets=pd.read_csv('./data/TabularData2VTargets.csv')
    df_targets.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    filenames = df_inputs['filename'].values##incase we need to refer the index later on
    a=df_inputs.drop('filename', axis=1).values
    b=df_targets.drop('filename', axis=1).values
    df_inputs.drop('filename', axis=1, inplace=True)
    df_targets.drop('filename', axis=1, inplace=True)
    
    filenames_test= filenames[-test_size:]
    
    df_x_test = pd.DataFrame(a[-test_size:], columns=df_inputs.columns, index=filenames_test)
    df_y1_test = pd.DataFrame(b[-test_size:], columns=df_targets.columns, index=filenames_test)
    
    df_inputs_train_val = df_inputs[:-test_size]
    df_targets_train_val = df_targets[:-test_size]
    
    filenames_test = filenames[-test_size:]
    filenames_train = filenames[:-test_size]
    
    x=df_inputs_train_val.values
    y1=df_targets_train_val.values
    
    ##I want NAN values which are already replaced as 0 to remain 0 should i change em to 1000 after scaling?
    
    ##############
    
    # max_mgrenz=np.max(y1)
    
    # max_rows=(max_mgrenz*2)+1
    # print(max_rows)
    # y2 = []
    # # Padding
    # for filename in filenames_train:
    #     y2_file = pd.read_csv(f'./data/ETAgrid/{filename}.csv')
    #     # Convert DataFrame to NumPy array and replace NaNs with 1000
    #     #y2_values = np.nan_to_num(y2_file.values, nan=1000)
    #     y2_values = y2_file.values
    #     # Replace 0s with nans
    #     #y2_values[y2_values == 0] = 1000
    #     y2_values[y2_values == 0] = np.nan
    #     padded = np.full((max_rows, 191),  np.nan, dtype=float)
    #     #padded = np.full((max_rows, 191), 1000, dtype=y2_values.dtype)
    #     padded[:y2_file.shape[0], :y2_file.shape[1]] = y2_values
    #     y2.append(padded)
    
    # y2 = np.array(y2)
    
    # np.save('./data/AllTopologiesTabularData2VETA.npy', y2)
    
    y2_complete = np.load('./data/400TabularData2VETA.npy')
    y2=y2_complete[:-test_size, :, :]###only 1st dimension need to be considered..test
    ##############
    #Reshape it to 2D before scaling, then reshape back to 3D
    y2_reshaped = y2.reshape(y2.shape[0], -1)  # Flatten to 2D
    
    
    # Count NaN values
    # nan_count = np.isnan(y2_reshaped).sum()
    # print(f"Number of NaN values: {nan_count}")

    # Count non-NaN values
    # nonnan_count = (~np.isnan(y2_reshaped)).sum()
    # print(f"Number of non-NaN values: {nonnan_count}")
    
    scaler_x=StandardScaler().fit(x)
    scaler_y1=MinMaxScaler().fit(y1) 
    scaler_y2 = StandardScaler().fit(y2_reshaped)

    x_normalized=scaler_x.transform(x)
    y1_normalized=scaler_y1.transform(y1)
    y2_normalized=scaler_y2.transform(y2_reshaped)

    y2_normalized = y2_normalized.reshape(y2.shape)  # Reshape back to 3D
    
    # Replace nans with 1000
    y2_normalized = np.nan_to_num(y2_normalized, nan=3)
    
    print(np.nanmax(y2_normalized))
    print(np.nanmin(y2_normalized))
    # # Count NaN values
    # nan_count = np.isnan(y2_normalized).sum()
    # print(f"Number of NaN values: {nan_count}")

    # #Count non-NaN values
    # nonnan_count = (~np.isnan(y2_normalized)).sum()
    # print(f"Number of non-NaN values: {nonnan_count}")
    
    input_size = x_normalized.shape[1]
    output_y1_size = y1_normalized.shape[1]
    output_y2_size = y2_normalized.shape[1]
  
    return x_normalized, y1_normalized, y2_normalized, scaler_x, scaler_y1, scaler_y2, input_size, output_y1_size,output_y2_size, df_x_test, df_y1_test, filenames_test
