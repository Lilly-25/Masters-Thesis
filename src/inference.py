import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.interpolate import griddata

def generate_predictions_mm(model_path, df_inputs_test, df_targets_test, x_scaler, y1_scaler, y2_scaler, device):
    
    model = torch.load(model_path) # Load the trained model saved locally
    model=model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    index = df_inputs_test.index#Index containing filename

    x_test = df_inputs_test.values
    
    x_test_normalized = x_scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_normalized, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(x_test_tensor) # Generate predictions
        
    predictions_y1 = predictions[0].to('cpu').numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_y1.shape}")
    
    predictions_y2 = predictions[1].to('cpu').numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_y2.shape}")

    target_columns = df_targets_test.columns

    # Inverse transform the predictions to bring back to original scale
    y1 = y1_scaler.inverse_transform(predictions_y1)
   
    ##predictions_rounded = np.round(predictions_original_scale).astype(int)  # Round predictions to nearest integer coz targets are always integers . TODO change tensor type to integer instead of float for y1 labels
    ##Client to take the call
    
    df_y1 = pd.DataFrame(y1, columns=target_columns, index=index)
    
    ##y2

    predictions_y2[predictions_y2 == -1] = np.nan
    
    flattened_predictions_y2 = predictions_y2.reshape(predictions_y2.shape[0], -1) 
    y2 = y2_scaler.inverse_transform(flattened_predictions_y2)  
    y2 = y2.reshape(predictions_y2.shape) 
    
    mm_matrix= []
    eta_matrix=[]
    
    for i in range(y2.shape[0]):
        
        max_mgrenz = df_y1.iloc[i].max()
        max_mgrenz_rounded = np.round(max_mgrenz).astype(int)
        #max_rows = (max_mgrenz_rounded * 2) + 1
        # nnkpi2darray = np.array(nn_kpi_2d)
        # nn_kpi_2d_matrix = np.tile(nnkpi2darray, (max_rows, 1))
        mm_values = np.arange(-max_mgrenz_rounded, max_mgrenz_rounded + 1)
        mm_kpi_2d_matrix = np.tile(mm_values.reshape(-1, 1), (1, 191))

        y2_shape = y2.shape[1:]

        #Padded MM matrix with NAN values
        padded_mm_kpi_2d_matrix = np.full(y2_shape, np.nan)
        #padded_nn_kpi_2d_matrix = np.full(desired_shape, np.nan)

        # Estimate which parts of the matrix is relevant for each engine
        start_row = (y2_shape[0] - mm_kpi_2d_matrix.shape[0]) // 2
        end_row = start_row + mm_kpi_2d_matrix.shape[0]
        
        padded_mm_kpi_2d_matrix[start_row:end_row, :] = mm_kpi_2d_matrix
        
        #The padded MM matrix is used as a mask to filter out the relevant ETA values
        
        eta_predicted = y2[i]
        mask = np.isfinite(padded_mm_kpi_2d_matrix)
        eta_predicted=eta_predicted[mask]
        eta_predicted_reshaped = eta_predicted.reshape(-1, 191)#;Masking changes the shape of the matrix hence reshape
        
        # padded_mm_kpi_2d_matrix=padded_mm_kpi_2d_matrix[mask]
        # padded_mm_kpi_2d_matrix_reshaped = padded_mm_kpi_2d_matrix.reshape(-1, 191)
        
        mm_matrix.append(list(range(-max_mgrenz_rounded, max_mgrenz_rounded + 1)))
        eta_matrix.append(eta_predicted_reshaped)

    return df_y1, y2, mm_matrix, eta_matrix

def generate_predictions(model_path, df_inputs_test, df_targets_test, x_scaler, y1_scaler, y2_scaler, device):
    
    model = torch.load(model_path) # Load the trained model saved locally
    model=model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    index = df_inputs_test.index#Index containing filename

    x_test = df_inputs_test.values
    
    x_test_normalized = x_scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_normalized, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(x_test_tensor) # Generate predictions
        
    predictions_y1 = predictions[0].to('cpu').numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_y1.shape}")
    
    predictions_y2 = predictions[1].to('cpu').numpy() # convert to numpy array
    print(f"Predictions shape: {predictions_y2.shape}")

    target_columns = df_targets_test.columns

    # Inverse transform the predictions to bring back to original scale
    y1 = y1_scaler.inverse_transform(predictions_y1)
   
    ##predictions_rounded = np.round(predictions_original_scale).astype(int)  # Round predictions to nearest integer coz targets are always integers . TODO change tensor type to integer instead of float for y1 labels
    ##Client to take the call
    
    df_y1 = pd.DataFrame(y1, columns=target_columns, index=index)
    
    ##y2

    predictions_y2[predictions_y2 == -1] = np.nan
    
    flattened_predictions_y2 = predictions_y2.reshape(predictions_y2.shape[0], -1) 
    y2 = y2_scaler.inverse_transform(flattened_predictions_y2)  
    y2 = y2.reshape(predictions_y2.shape) 
    
    mm_matrix= []
    eta_matrix=[]
    
    for i in range(y2.shape[0]):
        
        max_mgrenz = df_y1.iloc[i].max()
        max_mgrenz_rounded = np.round(max_mgrenz).astype(int)
        eta_predicted = y2[i]
        mm_matrix.append(list(range(-max_mgrenz_rounded, max_mgrenz_rounded + 1)))
        eta_matrix.append(eta_predicted)

    return df_y1, y2, mm_matrix, eta_matrix

def plot_testdataset_kpi2d(df_targets, df_predictions,start,end, cols):
    
    nn_kpi_2d = list(range(0, 19100, 100)) # NN values alyways range from 0 to 19000 rpm
    
    index=df_targets.index

    # cols=2
    row_height=5 
    col_width=5

    num_plots = end - start

    rows = (num_plots + cols - 1) // cols  # ceiling division to get the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(cols * col_width, rows * row_height))  # Adjust fig size based on rows and columns
    fig.suptitle(f'Plots of {start} to {end} Examples from the Test Dataset', fontsize=10)

    for j in range(rows * cols):
        row = j // cols
        col = j % cols
        current_index = (start-1) + j
        
        if current_index <= end and current_index < len(df_targets):
            axs[row, col].plot(nn_kpi_2d, df_targets.loc[index[current_index]].tolist(), label='Target', color='blue')
            axs[row, col].plot(nn_kpi_2d, df_predictions.loc[index[current_index]].tolist(), label='Predictions', color='red')
            axs[row, col].set_xlabel('NN')
            axs[row, col].set_ylabel('Mgrenz')
            axs[row, col].set_title(f'Mgrenz(Torque Curve) KPI for\n{index[current_index]}', fontsize=7)
            axs[row, col].legend()
        else:
            axs[row, col].axis('off')  
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()
    
def eval_plot_kpi2d(df_targets, df_predictions,start,end, cols):
    
    nn_kpi_2d = list(range(0, 19100, 100)) # NN values alyways range from 0 to 19000 rpm
        
    index=df_targets.index

    # cols=2
    row_height=5 
    col_width=5

    num_plots = end - start

    rows = (num_plots + cols - 1) // cols  # ceiling division to get the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(cols * col_width, rows * row_height))  # Adjust fig size based on rows and columns
    fig.suptitle(f'Plots of {start} to {end} Examples from the Test Dataset', fontsize=10)

    for j in range(rows * cols):
        row = j // cols
        col = j % cols
        current_index = start + j  # Change index calculation

        if current_index <= end and current_index < len(df_targets):
            target_values = df_targets.loc[index[current_index]].tolist()
            prediction_values = df_predictions.loc[index[current_index]].tolist()

            deviations = np.array(prediction_values) - np.array(target_values)
            variance = np.mean(deviations ** 2)
            std_dev = np.sqrt(variance)
            
            # deviations = np.array(prediction_values) - np.array(target_values)
            # #print('Difference', deviations)
            # std_dev = np.mean(abs(deviations))
                
            percentage_diff = 100 * abs(deviations) / target_values 
            #print('Percentage diff', percentage_diff)
            
            axs[row, col].plot(nn_kpi_2d, target_values, label='Target', color='blue')
            axs[row, col].plot(nn_kpi_2d, prediction_values, label='Predictions', color='red')
            axs[row, col].fill_between(nn_kpi_2d, np.array(target_values) - std_dev, np.array(target_values) + std_dev, 
                                        alpha=0.2, color='red', label='Prediction Std Dev')
            axs[row, col].set_xlabel('NN')
            axs[row, col].set_ylabel('Mgrenz')
            axs[row, col].set_title(f'Mgrenz(Torque Curve) KPI for\n{index[current_index]}', fontsize=10)
            axs[row, col].legend()
            
            # Plot differences
            # Create twin y-axis for differences and percentage differences
            ax2 = axs[row, col].twinx()
            ax2.plot(nn_kpi_2d, deviations, label='Difference', color='green', linestyle='--')
            ax2.plot(nn_kpi_2d, percentage_diff, label='Percentage Diff', color='purple', linestyle='--')
            ax2.set_ylabel('Difference & Percentage Differences')

            # Combine the legends of both y-axes
            lines_1, labels_1 = axs[row, col].get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
            ax2.axhline(0, color='gray', linestyle=':', linewidth=1.5)
            ax2.set_ylim(-20, 60)  # Scale of the freaky y axis
        else:
            axs[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def plot_kpi3d_dual_mm(nn, mm1, eta1, mm2, eta2, filename):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    #fig.suptitle(f'Motor Efficiency - {filename}', fontsize=16)#only for client
    fig.suptitle(f'Motor Efficiency', fontsize=16)#better for reporting
    # sns.set_theme(style="whitegrid")

    #Global min and max efficiency values for consistent colormap scaling
    Z_global_min = 0.00
    Z_global_max = 100.00

    #Norm object with global min and max from both the grids
    norm = mcolors.Normalize(vmin=Z_global_min, vmax=Z_global_max)
    
    #Use the same scale based on the global min and max
    levels = np.linspace(Z_global_min, Z_global_max, 1000)

    for ax, mm, eta, title in zip([ax1, ax2], [mm1, mm2], [eta1, eta2], ['Original', 'Predicted']):
        
        X = nn
        Y = mm
        Z = eta

        X, Y = np.meshgrid(X, Y)
        #Create a mask based on ETA which is already aligned with MM grid and use this mask on the other axis
        mask = np.isfinite(Z)
        X, Y, Z = X[mask], Y[mask], Z[mask]

        #Norm object plots the grids with the same color scale
        contour = ax.tricontourf(X.ravel(), Y.ravel(), Z.ravel(), levels=levels, cmap='jet', norm=norm)

        ax.set_xlabel('Angular Velocity [rpm]', fontsize=12)
        ax.set_ylabel('Torque [Nm]', fontsize=12)
        ax.set_title(f'{title} Efficiency', fontsize=14)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Efficiency', fontsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45, ha='right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()
    
def plot_kpi3d_dual(nn, mm1, eta1, mm2, eta2, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    #fig.suptitle(f'Motor Efficiency - {filename}', fontsize=16)#only for client
    fig.suptitle(f'Motor Efficiency', fontsize=16)#better for reporting
    # sns.set_theme(style="whitegrid")
    
    Z_global_min = 0.00
    Z_global_max = 100.00
    norm = mcolors.Normalize(vmin=Z_global_min, vmax=Z_global_max)
    levels = np.linspace(Z_global_min, Z_global_max, 1000)
    
    for ax, mm, eta, title in zip([ax1, ax2], [mm1, mm2], [eta1, eta2], ['Original', 'Predicted']):
        
        X, Y = np.meshgrid(nn, mm)
        Z = eta
        
        # Ensures the number of rows across all 3 axis is the same, the number of columns is always constant going to be 191
        min_rows = min(X.shape[0], Y.shape[0], Z.shape[0])
        X = X[:min_rows, :]
        Y = Y[:min_rows, :]
        Z = Z[:min_rows, :]
        
        mask = np.isfinite(Z)
        X, Y, Z = X[mask], Y[mask], Z[mask]
        
        contour = ax.tricontourf(X.ravel(), Y.ravel(), Z.ravel(), levels=levels, cmap='jet', norm=norm)
        ax.set_xlabel('Angular Velocity [rpm]', fontsize=12)
        ax.set_ylabel('Torque [Nm]', fontsize=12)
        ax.set_title(f'{title} Efficiency', fontsize=14)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Efficiency', fontsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()
    
def plot_mean_std_kpi2d(df_targets, df_predictions):
    nn_kpi_2d = list(range(0, 19100, 100))  # NN values always range from 0 to 19000 rpm
    
    #for each of the 191 NN values, calculate the mean and standard deviation of the target and prediction values
    target_mean = df_targets.mean().values
    target_std = df_targets.std().values
    pred_mean = df_predictions.mean().values
    pred_std = df_predictions.std().values

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean
    ax.plot(nn_kpi_2d, target_mean, label='Target Mean', color='blue')
    ax.plot(nn_kpi_2d, pred_mean, label='Prediction Mean', color='red')
    
    # Plot standard deviation as shaded area
    ax.fill_between(nn_kpi_2d, target_mean - target_std, target_mean + target_std, 
                    alpha=0.2, color='blue', label='Target Std Dev')
    ax.fill_between(nn_kpi_2d, pred_mean - pred_std, pred_mean + pred_std, 
                    alpha=0.2, color='red', label='Prediction Std Dev')

    ax.set_xlabel('NN (rpm)')
    ax.set_ylabel('Mgrenz')
    ax.set_title('Mean and Standard Deviation of Mgrenz(Torque Curve) KPI')
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_std_kpi2d(df_targets, df_predictions):
    nn_kpi_2d = list(range(0, 19100, 100))  # NN values always range from 0 to 19000 rpm
    
    #for each of the 191 NN values, calculate the mean values
    target_mean = df_targets.mean().values
    pred_mean = df_predictions.mean().values
    
    pred_std = []

    # Calculate standard deviation of predictions based on the means of the target columns
    for col in df_predictions.columns:
        deviations = df_predictions[col].values - df_targets[col].mean()
        variance = np.mean(deviations ** 2)
        std_dev = np.sqrt(variance)
        pred_std.append(std_dev)

    pred_std = np.array(pred_std)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean
    ax.plot(nn_kpi_2d, target_mean, label='Target Mean', color='blue')
    ax.plot(nn_kpi_2d, pred_mean, label='Prediction Mean', color='red', linestyle='dashed')
    
    ax.fill_between(nn_kpi_2d, target_mean - pred_std, target_mean + pred_std, 
                    alpha=0.2, color='red', label='Prediction Std Dev')

    ax.set_xlabel('NN (rpm)')
    ax.set_ylabel('Mgrenz')
    ax.set_title('Standard Deviation of Mgrenz(Torque Curve) KPI from the Target Mean')
    ax.legend()

    plt.tight_layout()
    plt.show()
    
def eval_plot_kpi3d_mm(nn, mm1, eta1, mm2, eta2, filename):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 10))
    #fig.suptitle(f'Motor Efficiency - {filename}', fontsize=16)#only for client
    fig.suptitle(f'Motor Efficiency', fontsize=16)#better for reporting
    # sns.set_theme(style="whitegrid")

    # Global min and max efficiency values for consistent colormap scaling
    Z_global_min, Z_global_max = 0.00, 100.00
    norm = mcolors.Normalize(vmin=Z_global_min, vmax=Z_global_max)
    levels = np.linspace(Z_global_min, Z_global_max, 1000)

    # Create a common grid
    x_min, x_max = 0, 19000
    y_min = min(np.min(mm1), np.min(mm2))
    y_max = max(np.max(mm1), np.max(mm2))
    
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 2000)##can change drastically for untrained files..need to then put max_mgrenz from it
    #Assuming 600 as it ranges from -max_mgrenz to max_mgrenz in steps of 1 and max_mgrenz we came across is *300
    xi, yi = np.meshgrid(xi, yi)

    for ax, mm, eta, title in zip([ax1, ax2], [mm1, mm2], [eta1, eta2], ['Original', 'Predicted']):
        X = nn
        Y = mm
        Z = eta

        # Ensure X, Y, and Z have compatible shapes
        X, Y = np.meshgrid(X, Y[:Z.shape[0]])
        
        # Flatten the arrays for griddata
        points = np.column_stack((X.ravel(), Y.ravel()))
        
        # Interpolate data onto the common grid
        zi = griddata(points, Z.ravel(), (xi, yi), method='linear')

        contour = ax.contourf(xi, yi, zi, levels=levels, cmap='jet', norm=norm)
        ax.set_xlabel('Angular Velocity [rpm]', fontsize=12)
        ax.set_ylabel('Torque [Nm]', fontsize=12)
        ax.set_title(f'{title} Efficiency', fontsize=14)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Efficiency', fontsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45, ha='right')

    # Calculate and plot the difference
    X1, Y1 = np.meshgrid(nn, mm1[:eta1.shape[0]])
    X2, Y2 = np.meshgrid(nn, mm2[:eta2.shape[0]])
    
    points1 = np.column_stack((X1.ravel(), Y1.ravel()))
    points2 = np.column_stack((X2.ravel(), Y2.ravel()))
    
    Z1 = griddata(points1, eta1.ravel(), (xi, yi), method='linear')
    Z2 = griddata(points2, eta2.ravel(), (xi, yi), method='linear')
    Z_diff = Z2 - Z1

    diff_norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(Z_diff), vcenter=0, vmax=np.nanmax(Z_diff))
    diff_contour = ax3.contourf(xi, yi, Z_diff, levels=100, cmap='RdBu_r', norm=diff_norm)
    ax3.set_xlabel('Angular Velocity [rpm]', fontsize=12)
    ax3.set_ylabel('Torque [Nm]', fontsize=12)
    ax3.set_title('Difference', fontsize=14)
    cbar_diff = fig.colorbar(diff_contour, ax=ax3)
    cbar_diff.set_label('Efficiency Difference', fontsize=12)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(10))
    x_ticks = ax3.get_xticks()
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_ticks, rotation=45, ha='right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

def eval_plot_kpi3d(nn, mm1, eta1, mm2, eta2, filename):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 10))
    #fig.suptitle(f'Motor Efficiency - {filename}', fontsize=16)#only for client
    fig.suptitle(f'Motor Efficiency', fontsize=16)#better for reporting
    # sns.set_theme(style="whitegrid")
    
    Z_global_min, Z_global_max = 0.00, 100.00
    norm = mcolors.Normalize(vmin=Z_global_min, vmax=Z_global_max)
    levels = np.linspace(Z_global_min, Z_global_max, 1000)
    
    x_min, x_max = 0, 19000
    y_min = min(np.min(mm1), np.min(mm2))
    y_max = max(np.max(mm1), np.max(mm2))
    
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 2000)##can change drastically for untrained files..need to then put max_mgrenz from it
    #Assuming 600 to be ideal as it ranges from -max_mgrenz to max_mgrenz in steps of 1 and max_mgrenz we came across is *300
    xi, yi = np.meshgrid(xi, yi)
    
    for ax, mm, eta, title in zip([ax1, ax2], [mm1, mm2], [eta1, eta2], ['Original', 'Predicted']):

        # Ensure mm and eta have compatible shapes as eta is not masked as per MM grid
        min_rows = min(mm.shape[0], eta.shape[0])
        mm = mm[:min_rows]
        eta = eta[:min_rows, :]
        
        X, Y = np.meshgrid(nn, mm)
        Z = eta
        
        # Flatten and remove any NaN values
        mask = ~np.isnan(Z.ravel())
        points = np.column_stack((X.ravel()[mask], Y.ravel()[mask]))
        values = Z.ravel()[mask]
        
        # Interpolate data onto the common grid
        zi = griddata(points, values, (xi, yi), method='linear')
        
        contour = ax.contourf(xi, yi, zi, levels=levels, cmap='jet', norm=norm)
        ax.set_xlabel('Angular Velocity [rpm]', fontsize=12)
        ax.set_ylabel('Torque [Nm]', fontsize=12)
        ax.set_title(f'{title} Efficiency', fontsize=14)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Efficiency', fontsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45, ha='right')
    
    # Calculate and plot the difference
    min_rows1 = min(mm1.shape[0], eta1.shape[0])
    min_rows2 = min(mm2.shape[0], eta2.shape[0])
    
    X1, Y1 = np.meshgrid(nn, mm1[:min_rows1])
    X2, Y2 = np.meshgrid(nn, mm2[:min_rows2])
    
    mask1 = ~np.isnan(eta1[:min_rows1, :].ravel())
    mask2 = ~np.isnan(eta2[:min_rows2, :].ravel())
    
    points1 = np.column_stack((X1.ravel()[mask1], Y1.ravel()[mask1]))
    points2 = np.column_stack((X2.ravel()[mask2], Y2.ravel()[mask2]))
    
    Z1 = griddata(points1, eta1[:min_rows1, :].ravel()[mask1], (xi, yi), method='linear')
    Z2 = griddata(points2, eta2[:min_rows2, :].ravel()[mask2], (xi, yi), method='linear')
    
    Z_diff = Z2 - Z1
    diff_norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(Z_diff), vcenter=0, vmax=np.nanmax(Z_diff))
    diff_contour = ax3.contourf(xi, yi, Z_diff, levels=100, cmap='RdBu_r', norm=diff_norm)
    ax3.set_xlabel('Angular Velocity [rpm]', fontsize=12)
    ax3.set_ylabel('Torque [Nm]', fontsize=12)
    ax3.set_title('Difference', fontsize=14)
    cbar_diff = fig.colorbar(diff_contour, ax=ax3)
    cbar_diff.set_label('Efficiency Difference', fontsize=12)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(10))
    x_ticks = ax3.get_xticks()
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_ticks, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()
    
def y2_score_mm(nn, mm1, eta1, mm2, eta2):
    # Create a common grid
    x_min = 0
    x_max = 19000
    y_min = min(np.min(mm1), np.min(mm2))
    y_max = max(np.max(mm1), np.max(mm2))
   
    xi = np.linspace(x_min, x_max, 200)#Assuming 200 as it ranges from 0 to 19000 in steps of 100
    yi = np.linspace(y_min, y_max, 600)#Assuming 600 as it ranges from -max_mgrenz to max_mgrenz in steps of 1 and max_mgrenz we came across is *300
    xi, yi = np.meshgrid(xi, yi)
    
    # Create coordinates for both plots
    X1, Y1 = np.meshgrid(nn, mm1)
    X2, Y2 = np.meshgrid(nn, mm2)
   
    points1 = np.column_stack((X1.ravel(), Y1.ravel()))
    points2 = np.column_stack((X2.ravel(), Y2.ravel()))
   
    Z1 = griddata(points1, eta1.ravel(), (xi, yi), method='linear')
    Z2 = griddata(points2, eta2.ravel(), (xi, yi), method='linear')
    
    Z_diff = Z2 - Z1
    vaild_Z_diff = abs(Z_diff[~np.isnan(Z_diff)])
    variance = np.mean(vaild_Z_diff ** 2)
    score = np.sqrt(variance)
    
    return score

def y2_score(eta1, eta2):
    
    if eta1.shape[0] > eta2.shape[0]:
        larger_eta, smaller_eta = eta1, eta2
    else:
        larger_eta, smaller_eta = eta2, eta1
    
    rows, cols = larger_eta.shape
    smaller_rows, smaller_cols = smaller_eta.shape
    
    # Create the interpolated grid
    interpolated_smaller = np.full_like(larger_eta, np.nan)
    
    # Calculate the starting row for placing the smaller eta
    start_row = (rows - smaller_rows) // 2
    
    # Place the smaller eta directly into the middle of the larger grid
    interpolated_smaller[start_row:start_row+smaller_rows, :smaller_cols] = smaller_eta
    
    Z_diff = larger_eta - interpolated_smaller
    
    valid_diff = np.abs(Z_diff[~np.isnan(Z_diff)])
    variance = np.mean(valid_diff ** 2)
    score = np.sqrt(variance)

    return score

def y1_score(df_predictions_y1, df_test_y1_targets):
    index=df_test_y1_targets.index
    std_dev=0
    for index_no in range(len(index)):
        target_values = df_test_y1_targets.loc[index[index_no]].tolist()
        prediction_values = df_predictions_y1.loc[index[index_no]].tolist()

        deviations = np.array(prediction_values) - np.array(target_values)
        variance = np.mean(deviations ** 2)
        std_dev += np.sqrt(variance)
    score=std_dev/len(index)
    return score