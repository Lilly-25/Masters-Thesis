from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd

def generate_predictions(model_path, df_inputs_test, df_targets_test, x_scaler, y_scaler):
    
    model = torch.load(model_path) # Load the trained model saved locally
    model.eval()  # Set the model to evaluation mode
    
    index = df_inputs_test.index#Index containing filename

    x_test = df_inputs_test.values
    
    x_test_normalized = x_scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_normalized, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(x_test_tensor) # Generate predictions

    predictions_np = predictions.numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_np.shape}")

    target_columns = df_targets_test.columns

    # Inverse transform the predictions to bring back to original scale
    predictions_original_scale = y_scaler.inverse_transform(predictions_np)
   
    ##predictions_rounded = np.round(predictions_original_scale).astype(int)  # Round predictions to nearest integer coz targets are always integers . TODO change tensor type to integer instead of float for y1 labels
    ##Client to take the call
    
    df_predictions = pd.DataFrame(predictions_original_scale, columns=target_columns, index=index)

    return df_predictions

def generate_all_predictions(model_path, df_inputs_test, df_targets_test, x_scaler, y1_scaler, y2_scaler, device):
    
    model = torch.load(model_path) # Load the trained model saved locally
    model=model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    index = df_inputs_test.index#Index containing filename

    x_test = df_inputs_test.values
    
    x_test_normalized = x_scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_normalized, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(x_test_tensor) # Generate predictions
        
    predictions_np_y1 = predictions[0].to('cpu').numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_np_y1.shape}")
    
    predictions_np_y2 = predictions[1].to('cpu').numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_np_y2.shape}")

    target_columns = df_targets_test.columns

    # Inverse transform the predictions to bring back to original scale
    predictions_original_scale_y1 = y1_scaler.inverse_transform(predictions_np_y1)
   
    ##predictions_rounded = np.round(predictions_original_scale).astype(int)  # Round predictions to nearest integer coz targets are always integers . TODO change tensor type to integer instead of float for y1 labels
    ##Client to take the call
    
    df_predictions_y1 = pd.DataFrame(predictions_original_scale_y1, columns=target_columns, index=index)
    
    ##y2
    

    predictions_np_y2[predictions_np_y2 == -1] = np.nan
    
    flattened_predictions_np_y2 = predictions_np_y2.reshape(predictions_np_y2.shape[0], -1) 
    predictions_original_scale_y2 = y2_scaler.inverse_transform(flattened_predictions_np_y2)  
    predictions_original_scale_y2 = predictions_original_scale_y2.reshape(predictions_np_y2.shape) 

    return df_predictions_y1, predictions_original_scale_y2

def generate_all_predictions(model_path, df_inputs_test, df_targets_test, x_scaler, y1_scaler, y2_scaler, device):
    
    model = torch.load(model_path) # Load the trained model saved locally
    model=model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    index = df_inputs_test.index#Index containing filename

    x_test = df_inputs_test.values
    
    x_test_normalized = x_scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_normalized, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(x_test_tensor) # Generate predictions
        
    predictions_np_y1 = predictions[0].to('cpu').numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_np_y1.shape}")
    
    predictions_np_y2 = predictions[1].to('cpu').numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_np_y2.shape}")

    target_columns = df_targets_test.columns

    # Inverse transform the predictions to bring back to original scale
    predictions_original_scale_y1 = y1_scaler.inverse_transform(predictions_np_y1)
   
    ##predictions_rounded = np.round(predictions_original_scale).astype(int)  # Round predictions to nearest integer coz targets are always integers . TODO change tensor type to integer instead of float for y1 labels
    ##Client to take the call
    
    df_predictions_y1 = pd.DataFrame(predictions_original_scale_y1, columns=target_columns, index=index)
    
    ##y2
    

    predictions_np_y2[predictions_np_y2 == -1] = np.nan
    
    flattened_predictions_np_y2 = predictions_np_y2.reshape(predictions_np_y2.shape[0], -1) 
    predictions_original_scale_y2 = y2_scaler.inverse_transform(flattened_predictions_np_y2)  
    predictions_original_scale_y2 = predictions_original_scale_y2.reshape(predictions_np_y2.shape) 

    return df_predictions_y1, predictions_original_scale_y2