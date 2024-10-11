import pandas as pd
from src.scaling import StdScaler
import numpy as np

def data_prep_eta_grid():##Give the directory path as input so the same function can be used for test
    
    df_inputs=pd.read_csv('./data/TabularDataInputs.csv')
    df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    df_targets=pd.read_csv('./data/TabularDataTargets.csv')
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
        y2_file = pd.read_csv(f'./data/TabularDataETAgrid/{filename}.csv')
        # Convert DataFrame to NumPy array and replace NaNs with 1000
        #y2_values = np.nan_to_num(y2_file.values, nan=1000)
        y2_values = y2_file.values
        # Replace 0s with nans
        #y2_values[y2_values == 0] = np.nan ###TODO GOTO remove this
        padded = np.full((max_rows, 191),  np.nan, dtype=float)
        #padded = np.full((max_rows, 191), 1000, dtype=y2_values.dtype)#1000 if we gave as a value will result in predictions close to this range even if the rest of input is scaled, so default value given close to scaling range after scaling
        # padded[:y2_file.shape[0], :y2_file.shape[1]] = y2_values
        padded[max_rows//2 - y2_file.shape[0]//2 : max_rows//2 + y2_file.shape[0]//2, :] = y2_values
        y2.append(padded)
    
    y2 = np.array(y2)
    
    np.save('./data/TabularDataETA.npy', y2)

def data_prep(test_size):

    df_inputs=pd.read_csv('./data/TabularDataInputs.csv')
    df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)
    
    # print(max(df_inputs['rad_phiv1']))
    # print(max(df_inputs['rad_phi3b']))

    df_targets=pd.read_csv('./data/TabularDataTargets.csv')
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
    
    print(y1.max())##probably return this and use it to calculate the max_rows in test
    max_mgrenz=y1.max() 

    y1_flat = y1.flatten() # We donot need to scale based on dimensions and so we flatten the array
    
    y2_complete = np.load('./data/TabularDataETA.npy')
    y2=y2_complete[:-test_size, :, :]###only 1st dimension need to be considered..test
    ##############
    #Reshape it to 2D before scaling, then reshape back to 3D
    y2_flat = y2.reshape(-1) # We donot need to scale based on dimensions and so we flatten the array to 1 D
    
    # Count NaN values
    # nan_count = np.isnan(y2_reshaped).sum()
    # print(f"Number of NaN values: {nan_count}")

    # Count non-NaN values
    # nonnan_count = (~np.isnan(y2_reshaped)).sum()
    # print(f"Number of non-NaN values: {nonnan_count}")
    
    x_min, x_max=StdScaler().fit(x, flatten=False)
    y1_min, y1_max=StdScaler().fit(y1_flat, flatten=True) 
    y2_min, y2_max = StdScaler().fit(y2_flat, flatten=True)

    x_normalized=StdScaler().transform(x, x_min, x_max)
    y1_normalized=StdScaler().transform(y1_flat, y1_min, y1_max)
    y2_normalized=StdScaler().transform(y2_flat, y2_min, y2_max)
    
    y1_normalized = y1_normalized.reshape(y1.shape)

    y2_normalized = y2_normalized.reshape(y2.shape)  # Reshape back to 3D
    
    # Replace nans with -1..or maybe try -.1...maybe predictions might not be so skewed
    y2_normalized = np.nan_to_num(y2_normalized, nan=-1) # If we give nan values to model, although it doesnt enter the network, loss completely turns nan--TODO check
    # Count NaN values
    nan_count = np.isnan(x_normalized).sum()
    print(f"Number of NaN values for x: {nan_count}")
    nan_count = np.isnan(y1_normalized).sum()
    print(f"Number of NaN values for x: {nan_count}")
    nan_count = np.isnan(y2_normalized).sum()
    print(f"Number of NaN values for x: {nan_count}")



    # #Count non-NaN values
    # nonnan_count = (~np.isnan(y2_normalized)).sum()
    # print(f"Number of non-NaN values: {nonnan_count}")
    
    input_size = x_normalized.shape[1]
  
    return x_normalized, y1_normalized, y2_normalized, x_min, x_max, y1_min, y1_max, y2_min, y2_max, input_size, df_x_test, df_y1_test, max_mgrenz
