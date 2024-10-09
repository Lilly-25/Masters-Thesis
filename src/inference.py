import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.interpolate import griddata

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
    
    ##y2 unmasking the nan values from the eta grid

    for em_id in range(predictions_y2.shape[0]):
        
        em = predictions_y2[em_id]
        mid_id = em.shape[0] // 2 # Find the middle row index of each ETa grid
        #print(predictions_y2[em_id,mid_id, 0:3])..Not exactly 0 but close to 0
        mask = (em == 0) # Create a mask for zeros
        mask[mid_id, :] = False  # Exclude the rows in the ETA grid where speed is 0
        em[mask] = np.nan # Replace zeros with np.nan everywhere except the middle row
        predictions_y2[em_id] = em

    flattened_predictions_y2 = predictions_y2.reshape(predictions_y2.shape[0], -1) 
    y2 = y2_scaler.inverse_transform(flattened_predictions_y2)  
    y2 = y2.reshape(predictions_y2.shape) 
    
    mm_matrix= []
    eta_matrix=[]
    
    for i in range(y2.shape[0]):
        
        max_mgrenz = df_y1.iloc[i].max()
        max_mgrenz_rounded = np.round(max_mgrenz).astype(int)
        if (2*max_mgrenz_rounded) + 1 > y2[i].shape[0]:
            print(f"Warning: The maximum torque value for {index[i]} is too high than expected")
            eta_predicted=y2[i]
        else:
            mid_eta= y2[i].shape[0] // 2 
            eta_predicted = y2[i, mid_eta-max_mgrenz_rounded:mid_eta+max_mgrenz_rounded+1 , :]
        #Probably need to add a check here when 2 * mgrenz values exceeds the eta grid size..error handling
        # eta_predicted = y2[i]
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

        i=1
        
        if mm.shape[0] != eta.shape[0]:
                    # For the scenario where we need to plot the topologies not trained on for  
            #We notice the difference especially for the Nabla topology
            #For the topology trained for, generally we have in post processing for the size of both MM and ETA to be the same hence this doesnt affect it
            
            if eta.shape[0] % 2 == 0:
                i=0

            min_rows = min(mm.shape[0], eta.shape[0])
            mm = mm[mm.shape[0]//2 - min_rows//2 : mm.shape[0]//2 + min_rows//2 + i]
            eta = eta[eta.shape[0]//2 - min_rows//2 : eta.shape[0]//2 + min_rows//2 + i, :]
            # eta[eta.shape[0]//2, :] = 0       #At 0 speed, known fact that efficiency is 0
            
        X, Y = np.meshgrid(nn, mm)
        Z = eta
            
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
    
def plot_mean_std_kpi2d(df_targets, df_predictions):#column wise  ##pretty useless  ##to deleteeeee
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

def plot_std_kpi2d(df_targets, df_predictions):#per sample row wise
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
    ax.plot(nn_kpi_2d, pred_mean, label='Standard deviation', color='yellow', linestyle='dotted')
    
    ax.fill_between(nn_kpi_2d, target_mean - pred_std, target_mean + pred_std, 
                    alpha=0.2, color='pink', label='Prediction Std Dev')

    ax.set_xlabel('NN (rpm)')
    ax.set_ylabel('Mgrenz')
    ax.set_title('Standard Deviation of Mgrenz(Torque Curve) KPI from the Target Mean')
    ax.legend()

    plt.tight_layout()
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
    #We are taking the min, max between prediced and true values here because predictions are rarely 100% accurate
    y_min = min(np.min(mm1), np.min(mm2))
    y_max = max(np.max(mm1), np.max(mm2))
    
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 2000)##can change drastically for untrained files..need to then put max_mgrenz from it
    #Assuming 600 to be ideal as it ranges from -max_mgrenz to max_mgrenz in steps of 1 and max_mgrenz we came across is *300
    xi, yi = np.meshgrid(xi, yi)
    
    for ax, mm, eta, title in zip([ax1, ax2], [mm1, mm2], [eta1, eta2], ['Original', 'Predicted']):
        
        i=1
        
        if mm.shape[0] != eta.shape[0]:
                    # For the scenario where we need to plot the topologies not trained on for  
            #We notice the difference especially for the Nabla topology
            #For the topology trained for, generally we have in post processing for the size of both MM and ETA to be the same hence this doesnt affect it
            
            if eta.shape[0] % 2 == 0:
                i=0

            min_rows = min(mm.shape[0], eta.shape[0])
            mm = mm[mm.shape[0]//2 - min_rows//2 : mm.shape[0]//2 + min_rows//2 + i]
            eta = eta[eta.shape[0]//2 - min_rows//2 : eta.shape[0]//2 + min_rows//2 + i, :]
            # eta[eta.shape[0]//2, :] = 0       #At 0 speed, known fact that efficiency is 0
        
        X, Y = np.meshgrid(nn, mm)
        Z = eta
        
        # Flatten and remove any NaN values
        mask = ~np.isnan(Z.ravel())
        # mask = np.isfinite(Z)
        
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
    
    # Calculate and plot the difference. Clcaulating min helps in case of untrained topologies
    min_rows1 = min(mm1.shape[0], eta1.shape[0])
    min_rows2 = min(mm2.shape[0], eta2.shape[0])

    X1, Y1 = np.meshgrid(nn, mm1[mm1.shape[0]//2 - min_rows1//2 : mm1.shape[0]//2 + min_rows1//2 + i])
    X2, Y2 = np.meshgrid(nn, mm2[mm2.shape[0]//2 - min_rows2//2 : mm2.shape[0]//2 + min_rows2//2 + i])
    
    mask1 = ~np.isnan(eta1[eta1.shape[0]//2 - min_rows1//2 : eta1.shape[0]//2 + min_rows1//2 + i].ravel())
    mask2 = ~np.isnan(eta2[eta2.shape[0]//2 - min_rows2//2 : eta2.shape[0]//2 + min_rows2//2 + i].ravel())
    
    points1 = np.column_stack((X1.ravel()[mask1], Y1.ravel()[mask1]))
    points2 = np.column_stack((X2.ravel()[mask2], Y2.ravel()[mask2]))
    
    Z1 = griddata(points1, eta1[eta1.shape[0]//2 - min_rows1//2 : eta1.shape[0]//2 + min_rows1//2 + i].ravel()[mask1], (xi, yi), method='linear')
    Z2 = griddata(points2, eta2[eta2.shape[0]//2 - min_rows2//2 : eta2.shape[0]//2 + min_rows2//2 + i].ravel()[mask2], (xi, yi), method='linear')
    
    Z_diff = np.abs(Z2 - Z1)
    
    # Use a standard colormap for absolute difference
    diff_contour = ax3.contourf(xi, yi, Z_diff, levels=1000, cmap='Reds')
    ax3.set_xlabel('Angular Velocity [rpm]', fontsize=12)
    ax3.set_ylabel('Torque [Nm]', fontsize=12)
    ax3.set_title('Absolute Difference', fontsize=14)
    cbar_diff = fig.colorbar(diff_contour, ax=ax3)
    cbar_diff.set_label('Absolute Efficiency Difference', fontsize=12)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(10))
    x_ticks = ax3.get_xticks()
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_ticks, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()
    

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
    score=0
    scores=[]
    for index_no in range(len(index)):
        target_values = df_test_y1_targets.loc[index[index_no]].tolist()
        prediction_values = df_predictions_y1.loc[index[index_no]].tolist()

        deviations = np.array(prediction_values) - np.array(target_values)
        variance = np.mean(deviations ** 2)
        std_dev = np.sqrt(variance)
        scores.append(std_dev)
        score+=std_dev
    
    y1_score=score/len(index)
    return y1_score, scores

def plot_kpi2d_stddev(df_y1_avg, df_test_y1_targets, model):
    nn_kpi_2d = list(range(0, 19100, 100))

    # Calculate mean RMSE to identify overlap wih target
    mean_rmse= ((df_y1_avg.iloc[0] - df_test_y1_targets)**2).mean(axis=0, skipna=True)**0.5

    # # Calculate element-wise RMSE
    # element_wise_rmse = ((df_y1_avg.iloc[0] - df_test_y1_targets)**2 / len(df_test_y1_targets))**0.5 # Nan not considered in numerator when taking subtraction but in denominator to find mean, it is conidered in len()

    squared_deviations = (df_y1_avg.iloc[0] - df_test_y1_targets)**2
    non_nan = squared_deviations.count()#Nan values if exists are not counted
    element_wise_rmse = (squared_deviations / non_nan)**0.5

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Plot average on the first y-axis
    ax1.plot(nn_kpi_2d, df_y1_avg.iloc[0], label=f'{model} Model', color='blue', linewidth=2)

    # Plot standard deviation as shaded area
    ax1.fill_between(nn_kpi_2d, 
                    df_y1_avg.iloc[0] - mean_rmse, 
                    df_y1_avg.iloc[0] + mean_rmse,
                    alpha=0.2, color='blue', label='± Average RMSE')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Speed (rpm)', fontsize=12)
    ax1.set_ylabel('Torque (N/m)', fontsize=12)

    # Create a twin axis for RMSE
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE', color='red', fontsize=12)

    # Plot element-wise RMSE for each test sample
    for i in range(len(element_wise_rmse)):
        ax2.plot(nn_kpi_2d, element_wise_rmse.iloc[i], 
                label=f'RMSE Sample {i+1}', 
                linestyle='--', alpha=0.7)

    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(f'Average RMSE and Element-wise RMSE of Test Dataset with {model} Model', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_kpi3d_stddev(y2_grid_avg, y2_grid, model):
    
    squared_deviations = (y2_grid_avg - y2_grid)**2
    rmse = (np.nanmean(squared_deviations, axis=0))**0.5

    plt.figure(figsize=(12, 5))

    x_min, x_max = 0, rmse.shape[1]
    y_min, y_max = -rmse.shape[0]//2, rmse.shape[0]//2

    im = plt.imshow(rmse, cmap='jet', extent=[x_min, x_max, y_min, y_max], aspect='auto', origin='lower')

    plt.colorbar(im, label='RMSE')
    plt.title(f'RMSE of Test Dataset Samples from {model} Model')
    plt.xlabel('Speed (rpm)/100')
    plt.ylabel('Torque (Nm)')

    x_ticks = np.arange(x_min, x_max, 20)  # x ticks are incorrect here and follows the actual speed /100
    plt.xticks(x_ticks)
    plt.show()
    
def plot_scores(scores, target):
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, kde=True, bins=20*3)
    plt.title(f'{target} Score Distribution')
    plt.xlabel(f'{target} Scores')
    plt.ylabel('Count')