from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd

def generate_predictions(model_path, df_inputs_test, df_targets_test):
    
    model = torch.load(model_path) # Load the trained model saved locally
    model.eval()  # Set the model to evaluation mode
    
    index = df_inputs_test.index#Index containing filename

    x_test = df_inputs_test.values
    
    #print(f"Input shape: {x_test.shape}")

    scaler = MinMaxScaler()#Scale with same scaler used for training TODO check if its better to refer the one used for training directly..
    scaler.fit(x_test)
    
    x_test_normalized = scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_normalized, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(x_test_tensor) # Generate predictions

    predictions_np = predictions.numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_np.shape}")

    target_columns = df_targets_test.columns

    y_scaler = MinMaxScaler()#Scale with same scaler used for training TODO check if its better to refer the one used for training directly..
    y_scaler.fit(df_targets_test)

    # Inverse transform the predictions to bring back to original scale
    try:
        predictions_original_scale = y_scaler.inverse_transform(predictions_np)
    except ValueError as e:
        print(f"Error during inverse transform: {e}")
        print(f"y_scaler.scale_ shape: {y_scaler.scale_.shape}")
        print(f"y_scaler.min_ shape: {y_scaler.min_.shape}")
        # If shapes don't match, we might need to transpose the predictions
        if predictions_np.shape[1] != len(y_scaler.scale_):
            predictions_np = predictions_np.T
            print(f"Transposed predictions shape: {predictions_np.shape}")
        predictions_original_scale = y_scaler.inverse_transform(predictions_np)

   
    predictions_rounded = np.round(predictions_original_scale).astype(int)  # Round predictions to nearest integer coz targets are always integers . TODO change tensor type to integer instead of float for y1 labels

    df_predictions = pd.DataFrame(predictions_rounded, columns=target_columns, index=index)

    return df_predictions