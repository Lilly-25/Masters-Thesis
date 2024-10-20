import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from src.scaling import StdScaler
from src.utils import cumulative_counts as cumulative_col_count
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize

def generate_predictions(model_path, df_inputs_test, df_targets_test, x_mean, x_stddev, device):
    
    model = torch.load(model_path) # Load the trained model saved locally
    model=model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    index = df_inputs_test.index#Index containing filename

    x_test = df_inputs_test.values
    
    x_test_normalized = StdScaler().transform(x_test, x_mean, x_stddev)
    x_test_tensor = torch.tensor(x_test_normalized, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(x_test_tensor) # Generate predictions
        
    y1 = predictions[0].to('cpu').numpy() # convert to numpy array
    print(f"Predictions y1 shape: {y1.shape}")
    
    y2_half = predictions[1].to('cpu').numpy() # convert to numpy array
    print(f"Predictions y2 shape: {y2_half.shape}")

    y2 = np.zeros((y2_half.shape[0], 2* y2_half.shape[1]-1, y2_half.shape[2]))  #Mirroring negative grid of ETA to be same as the predicted positive grid

    for i in range(y2_half.shape[0]):
        y2[i, :y2_half.shape[1]-1, :] = y2_half[i, -1:0:-1, :]
        y2[i, y2_half.shape[1]-1:, :] = y2_half[i]

    target_columns = df_targets_test.columns

    ##predictions_rounded = np.round(y1).astype(int)  # Round predictions to nearest integer coz targets are always integers . TODO change tensor type to integer instead of float for y1 labels
    ##Client to take the call
    
    df_y1 = pd.DataFrame(y1, columns=target_columns, index=index)
    
    ##y2 unmasking the nan values from the eta grid

    for em_id in range(y2.shape[0]):
        
        em = y2[em_id]
        mid_id = em.shape[0] // 2 # Find the middle row index of each ETa grid
        #print(predictions_y2[em_id,mid_id, 0:3])..Not exactly 0 but close to 0
        mask = (em == 0) # Create a mask for zeros
        mask[mid_id, :] = False  # Exclude the rows in the ETA grid where speed is 0
        em[mask] = np.nan # Replace zeros with np.nan everywhere except the middle row
        y2[em_id] = em

    mm_matrix= []
    eta_matrix=[]
    
    for i in range(y2.shape[0]): # loop through each sasmple
        
        max_mgrenz = df_y1.iloc[i].values.max()
        max_mgrenz_rounded = np.round(max_mgrenz).astype(int)
        mgrenz_values = np.round(df_y1.iloc[i]).values.astype(int)
        # print('Mgrenz \n',mgrenz_values)
        eta_predicted=[]
        
        if (2*max_mgrenz_rounded) + 1 > y2[i].shape[0]: # If worse case scenario where we predict a torque higher than expected ETA
            print(f"Warning: The maximum torque value for {index[i]} is too high than expected")
            eta_predicted=y2[i]
            
        else:
            mid_eta= y2[i].shape[0] // 2
            y2_sliced = y2[i, mid_eta-max_mgrenz_rounded:mid_eta+max_mgrenz_rounded+1 , :]
            cumulative_counts = cumulative_col_count(mgrenz_values)
            # print('Cumulative counts', cumulative_counts)
            negative_eta=[]
            j = 0
            k=0
            # while i < y2_sliced.shape[0]//2 + 1:
            for i in range(y2_sliced.shape[0]//2 + 1): 
                if j < len(cumulative_counts):
                    negative_columns = cumulative_counts[j]
                else:
                    negative_columns = cumulative_counts[-1]
                
                padded_eta = np.full(191, np.nan)
                padded_eta[:negative_columns] = y2_sliced[i, :negative_columns]
                negative_eta.append(padded_eta)

                # Update j based on mgrenz_values
                if negative_columns < len(mgrenz_values) and k == 0:
                    k = abs(mgrenz_values[negative_columns-1] - mgrenz_values[negative_columns])
                if k == 1:
                    j+=1
                k -= 1
            eta_predicted.extend(negative_eta)
            positive_eta=[]
            j=0
            k=0
            for i in range(y2_sliced.shape[0] - 1, y2_sliced.shape[0]//2, -1): # Loop through each row in the sample
                if j < len(cumulative_counts):
                    positive_columns = cumulative_counts[j]
                else:
                    positive_columns = cumulative_counts[-1]
                padded_eta = np.full(191, np.nan)
                padded_eta[:positive_columns] = y2_sliced[i, :positive_columns]
                positive_eta.append(padded_eta)
                # Update j based on mgrenz_values
                if positive_columns < len(mgrenz_values) and k == 0:
                    k = abs(mgrenz_values[positive_columns-1] - mgrenz_values[positive_columns])
                if k == 1:
                    j+=1
                k -= 1
            eta_predicted.extend(reversed(positive_eta))

        mm_matrix.append(list(range(-max_mgrenz_rounded, max_mgrenz_rounded + 1)))
        eta_matrix.append(np.array(eta_predicted))
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
            #axs[row, col].set_title(f'Mgrenz(Torque Curve) KPI for\n{index[current_index]}', fontsize=10)
            axs[row, col].set_title(f'Mgrenz(Torque Curve) KPI', fontsize=10)
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
    fig.suptitle(f'Motor Efficiency', fontsize=16)

    Z_global_min = 0.00
    Z_global_max = 100.00
    norm = Normalize(vmin=Z_global_min, vmax=Z_global_max)

    for ax, mm, eta, title in zip([ax1, ax2], [mm1, mm2], [eta1, eta2], ['Original', 'Predicted']):
        if mm.shape[0] != eta.shape[0]:
            i = 1 if eta.shape[0] % 2 != 0 else 0
            min_rows = min(mm.shape[0], eta.shape[0])
            mm = mm[mm.shape[0]//2 - min_rows//2 : mm.shape[0]//2 + min_rows//2 + i]
            eta = eta[eta.shape[0]//2 - min_rows//2 : eta.shape[0]//2 + min_rows//2 + i, :]

        X, Y = np.meshgrid(nn, mm)
        Z = eta

        im = ax.pcolormesh(X, Y, Z, cmap='jet', norm=norm, shading='auto')
        ax.set_xlabel('Angular Velocity [rpm]', fontsize=12)
        ax.set_ylabel('Torque [Nm]', fontsize=12)
        ax.set_title(f'{title} Efficiency', fontsize=14)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Efficiency', fontsize=12)
        # Set x-axis limits
        ax.set_xlim(0, max(nn))

        # Improve x-axis tick labels
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

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
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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

    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
def eval_plot_kpi3d(nn, mm1, eta1, mm2, eta2, filename):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 10))
    fig.suptitle(f'Motor Efficiency', fontsize=16)

    Z_global_min, Z_global_max = 0.00, 100.00
    norm = Normalize(vmin=Z_global_min, vmax=Z_global_max)

    def plot_data(ax, X, Y, Z, title, cmap='jet'):
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto')
        ax.set_xlabel('Angular Velocity [rpm]', fontsize=12)
        ax.set_ylabel('Torque [Nm]', fontsize=12)
        ax.set_title(f'{title}', fontsize=14)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Efficiency' if title != 'Absolute Difference' else 'Absolute Efficiency Difference', fontsize=12)
        ax.set_xlim(0, max(nn))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def prepare_data(nn, mm, eta):
        min_rows = min(mm.shape[0], eta.shape[0])
        i = 1 if eta.shape[0] % 2 != 0 else 0
        eta = eta[eta.shape[0]//2 - min_rows//2 : eta.shape[0]//2 + min_rows//2 + i]
        mm = mm[mm.shape[0]//2 - min_rows//2 : mm.shape[0]//2 + min_rows//2 + i]
        
        X, Y = np.meshgrid(nn, mm)
        
        # Remove NaN and infinite values
        mask = np.isfinite(eta)
        X_valid = X[mask]
        Y_valid = Y[mask]
        eta_valid = eta[mask]
        return X_valid, Y_valid, eta_valid

    X1, Y1, Z1 = prepare_data(nn, mm1, eta1)
    X2, Y2, Z2 = prepare_data(nn, mm2, eta2)

    # Create a common grid for interpolation
    x_min, x_max = min(X1.min(), X2.min()), max(X1.max(), X2.max())
    y_min, y_max = min(Y1.min(), Y2.min()), max(Y1.max(), Y2.max())
    xi = np.linspace(x_min, x_max, len(nn))  # Use original nn length
    yi = np.linspace(y_min, y_max, min(len(mm1), len(mm2)))  # Use minimum mm length
    XI, YI = np.meshgrid(xi, yi)
    
    Z1_interp = griddata((X1, Y1), Z1, (XI, YI), fill_value=np.nan)
    Z2_interp = griddata((X2, Y2), Z2, (XI, YI), fill_value=np.nan)
    
    plot_data(ax1, XI, YI, Z1_interp, 'Original Efficiency')
    plot_data(ax2, XI, YI, Z2_interp, 'Predicted Efficiency')

    # Calculate and plot the difference
    Z_diff = np.abs(Z2_interp - Z1_interp)
    plot_data(ax3, XI, YI, Z_diff, 'Absolute Difference', cmap='Reds')

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

def plot_kpi2d_stddev(df_y1_avg, df_test_y1_targets, plot, model):
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
                    alpha=0.2, color='blue', label=f'± Average {plot}')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Speed (rpm)', fontsize=12)
    ax1.set_ylabel('Torque (N/m)', fontsize=12)

    # Create a twin axis for RMSE
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'{plot}', color='red', fontsize=12)

    # Plot element-wise RMSE for each test sample
    for i in range(len(element_wise_rmse)):
        ax2.plot(nn_kpi_2d, element_wise_rmse.iloc[i], 
                linestyle='--', alpha=0.7)

    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(f'Average {plot} and Element-wise {plot} of Test Dataset with {model} Model', fontsize=14)
    plt.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
def plot_kpi3d_stddev(y2_grid_avg, y2_grid, plot, model):
    
    squared_deviations = (y2_grid_avg - y2_grid)**2
    rmse = (np.nanmean(squared_deviations, axis=0))**0.5

    plt.figure(figsize=(12, 5))

    x_min, x_max = 0, rmse.shape[1]
    y_min, y_max = -rmse.shape[0]//2, rmse.shape[0]//2

    im = plt.imshow(rmse, cmap='Reds', extent=[x_min, x_max, y_min, y_max], aspect='auto', origin='lower')

    plt.colorbar(im, label='RMSE')
    plt.title(f'{plot} of Test Dataset Samples from {model} Model')
    plt.xlabel('Speed (rpm)/100')
    plt.ylabel('Torque (Nm)')

    x_ticks = np.arange(x_min, x_max, 20)  # x ticks are incorrect here and follows the actual speed /100
    plt.xticks(x_ticks)
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    
def plot_scores(scores, target, model):
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, kde=True, bins=20*3)
    plt.title(f'{target} Score Distribution for {model} Model')
    plt.xlabel(f'{target} Scores')
    plt.ylabel('Count')
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def plot_eta(etas, ax, n):
    for i, eta_array in enumerate(etas):
        x = np.linspace(-300, 300, len(eta_array))
        ax.plot(x, eta_array, label=f'Array {i+1}', linestyle='--')
    
    ax.set_xlabel('Torque (Nm)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title(f'Efficiency at Speed {n} rpm')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_eta_statistics(eta, speed_ranges, input):
    num_plots = len(eta)

    num_cols = min(3, num_plots)  
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows))
    fig.suptitle(f'Standard Deviation of {input} Efficiency across NN ranges', fontsize=16)

    if num_plots > 1:
        axs = axs.flatten()
        
    n=speed_ranges[0] * 100

    for i, etas in enumerate(eta):
        if num_plots > 1:
            ax = axs[i]
        else:
            ax = axs
        plot_eta(etas, ax, n)
        n+= 2000

    # Remove any unused subplots
    if num_plots > 1:
        for j in range(num_plots, len(axs)):
            fig.delaxes(axs[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
def plot_eta_mean_statistics(speed_ranges, mean_eta, std_eta):
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(speed_ranges, mean_eta, yerr=std_eta, fmt='o', capsize=5, label="Mean ± Std Dev", ecolor='red', linestyle='--', marker='s')
    plt.xlabel("Speed*100(rpm) ")
    plt.ylabel("Efficiency(%)")
    plt.title("Standard Deviation of ETA values ranging NN speed")
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.show()