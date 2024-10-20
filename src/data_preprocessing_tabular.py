import pandas as pd
from src.scaling import StdScaler
import numpy as np

def data_prep_eta_grid():##Give the directory path as input so the same function can be used for test
    
    df_inputs=pd.read_csv('./data/TabularDataInputs.csv')
    df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    df_targets=pd.read_csv('./data/TabularDataY1Targets.csv')
    df_targets.rename(columns={'Unnamed: 0':'filename'}, inplace=True)

    filenames = df_inputs['filename'].values
    
    df_targets=df_targets.drop('filename', axis=1)
      
    y1=df_targets.values
    
    max_mgrenz=np.max(y1) # Max mgrenz here decided based on max value from our dataset should be overriddenable by user

    max_rows=max_mgrenz+1 # Positive grid including 0
    
    print(max_rows)
    y2 = []
    # Padding
    for filename in filenames:
        y2_file = pd.read_csv(f'./data/TabularDataETAgrid/{filename}.csv')
        y2_values = y2_file.values
        padded = np.full((max_rows, 191),  np.nan, dtype=float)
        padded[0 : y2_file.shape[0], :] = y2_values
        y2.append(padded)
    
    y2 = np.array(y2)
    
    np.save('./data/TabularDataETA.npy', y2)

def data_prep(test_size):

    df_inputs=pd.read_csv('./data/TabularDataInputs.csv')
    df_inputs.rename(columns={'Unnamed: 0':'filename'}, inplace=True)
    
    # print(max(df_inputs['rad_phiv1']))
    # print(max(df_inputs['rad_phi3b']))

    df_targets=pd.read_csv('./data/TabularDataY1Targets.csv')
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

    max_mgrenz=y1.max() 

    y2_complete = np.load('./data/TabularDataETA.npy')
    y2=y2_complete[:-test_size, :, :]###only 1st dimension need to be considered..test
    
    x_mean, x_stddev=StdScaler().fit(x)
    x_normalized=StdScaler().transform(x, x_mean, x_stddev)
    
    return x_normalized, y1, y2, x_mean, x_stddev, df_x_test, df_y1_test, max_mgrenz
