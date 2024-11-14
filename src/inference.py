import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.scaling import StdScaler
from src.utils import cumulative_counts as cumulative_col_count, pdiff_scoring as pdiff_scores
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import joblib

def generate_predictions(model, df_inputs_test, df_targets_test, x_mean, x_stddev, device):
    
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
        mask[mid_id, :] = False  # Exclude the rows in the ETA grid where Angular Velocity is 0
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
        
        if (max_mgrenz_rounded) + 1 > y2[i].shape[0]: # If worse case scenario where we predict a torque higher than expected ETA
            print(f"Warning: The maximum torque value for {index[i]} is too high than expected")
            eta_predicted=y2[i]
            
        else:
            mid_eta= y2[i].shape[0] // 2
            y2_sliced = y2[i, mid_eta-max_mgrenz_rounded:mid_eta+max_mgrenz_rounded+1 , :]
            cumulative_counts = cumulative_col_count(mgrenz_values)
            # print('Cumulative counts', cumulative_counts)
            # negative_eta=[]
            # j = 0
            # k=0
            # # while i < y2_sliced.shape[0]//2:
            # for i in range(y2_sliced.shape[0]//2): 
            #     if j < len(cumulative_counts):
            #         negative_columns = cumulative_counts[j]
            #     else:
            #         negative_columns = cumulative_counts[-1]
                
            #     padded_eta = np.full(191, np.nan)
            #     padded_eta[:negative_columns] = y2_sliced[i, :negative_columns]
            #     negative_eta.append(padded_eta)

            #     # Update j based on mgrenz_values
            #     if negative_columns < len(mgrenz_values) and k == 0:
            #         k = abs(mgrenz_values[negative_columns-1] - mgrenz_values[negative_columns])
            #     if k == 1:
            #         j+=1
            #     k -= 1
            # eta_predicted.extend(negative_eta)
            positive_eta=[]
            j=0
            k=0
            for i in range(y2_sliced.shape[0] - 1, y2_sliced.shape[0]//2 - 1, -1): # Loop through each row in the sample
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

        # mm_matrix.append(list(range(-max_mgrenz_rounded, max_mgrenz_rounded + 1)))
        mm_matrix.append(list(range(0, max_mgrenz_rounded + 1)))
        eta_matrix.append(np.array(eta_predicted))
    return df_y1, mm_matrix, eta_matrix

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
            axs[row, col].set_xlabel('Angular Velocity (rpm)')
            axs[row, col].set_ylabel('Torque (Nm)')
            axs[row, col].set_title(f'Torque Curve KPI for\n{index[current_index]}', fontsize=7)
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
    # fig.suptitle(f'Plots of {start} to {end} Examples from the Test Dataset', fontsize=10)

    for j in range(rows * cols):
        row = j // cols
        col = j % cols
        current_index = start + j  # Change index calculation

        if current_index <= end and current_index < len(df_targets):
            target_values = df_targets.loc[index[current_index]].tolist()
            prediction_values = df_predictions.loc[index[current_index]].tolist()

            deviations = np.array(prediction_values) - np.array(target_values)
            variance = np.mean(deviations ** 2)
            rmse = np.sqrt(variance)

                
            percentage_diff = 100 * abs(deviations) / target_values 
            
            axs[row, col].plot(nn_kpi_2d, target_values, label='Target', color='blue')
            axs[row, col].plot(nn_kpi_2d, prediction_values, label='Predictions', color='red')
            axs[row, col].fill_between(nn_kpi_2d, np.array(target_values) - rmse, np.array(target_values) + rmse, 
                                        alpha=0.2, color='red', label='Prediction RMSE')
            axs[row, col].set_xlabel('Angular Velocity (rpm)')
            axs[row, col].set_ylabel('Torque (Nm)')
      
            # axs[row, col].set_title(f'Mgrenz(Torque Curve) KPI', fontsize=10)
            axs[row, col].legend()
            axs[row, col].spines['top'].set_visible(False)
            
            # Plot differences
            # Create twin y-axis for differences and percentage differences
            ax2 = axs[row, col].twinx()
            ax2.plot(nn_kpi_2d, deviations, label='Difference', color='green', linestyle='--')
            ax2.plot(nn_kpi_2d, percentage_diff, label='Percentage Diff', color='purple', linestyle='--')
            ax2.set_ylabel('Difference & Percentage Differences', color='red')
            ax2.tick_params(axis='y', colors='red')
            ax2.spines['right'].set_color('red')  # Color the right spine red
            ax2.spines['right'].set_linestyle('--')  
            ax2.spines['top'].set_visible(False)  # Hide the top spine for the twin axis

            # Combine the legends of both y-axes
            lines_1, labels_1 = axs[row, col].get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
            ax2.axhline(0, color='gray', linestyle=':', linewidth=1.5)
            ax2.set_ylim(-20, 60)  # Scale of the freaky y axis
        else:
            axs[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('./Manuscript/ReportImages/kpi2d_predictions.png', bbox_inches='tight')
    plt.show()
    

def plot_kpi3d_dual(nn, mm1, eta1, mm2, eta2, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    # fig.suptitle(f'Motor Efficiency', fontsize=16)

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
        ax.set_xlabel('Angular Velocity [rpm]', fontsize=14)
        ax.set_ylabel('Torque [Nm]', fontsize=14)
        ax.set_title(f'{title} Efficiency', fontsize=16)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Efficiency (%)', fontsize=14)
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
   
    
def clean_eta(eta_kpi3d):
    
    cleaned_eta_kpi3d=(np.nan_to_num(eta_kpi3d, nan=0.0))
    return cleaned_eta_kpi3d

def eta_difference(eta_kpi3d, eta_predicted):
    
    cleaned_eta_kpi3d=clean_eta(eta_kpi3d)
    cleaned_eta_predicted=clean_eta(eta_predicted)

    min_shape = min(cleaned_eta_kpi3d.shape[0], cleaned_eta_predicted.shape[0])
    eta_diff = cleaned_eta_kpi3d[:min_shape, :] - cleaned_eta_predicted[:min_shape, :] 
    
    return eta_diff    

def eval_plot_kpi3d(nn, mm1, eta1, mm2,  eta2, mm_diff, eta_diff, filename):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 10))
    # fig.suptitle(f'Motor Efficiency', fontsize=16)

    Z_global_min = 0.00
    Z_global_max = 100.00
    norm = Normalize(vmin=Z_global_min, vmax=Z_global_max)

    for ax, mm, eta, title in zip([ax1, ax2, ax3], [mm1, mm2, mm_diff], [eta1, eta2,eta_diff], ['Original', 'Predicted', 'Difference']):
        if mm.shape[0] != eta.shape[0]:
            i = 1 if eta.shape[0] % 2 != 0 else 0
            min_rows = min(mm.shape[0], eta.shape[0])
            mm = mm[mm.shape[0]//2 - min_rows//2 : mm.shape[0]//2 + min_rows//2 + i]
            eta = eta[eta.shape[0]//2 - min_rows//2 : eta.shape[0]//2 + min_rows//2 + i, :]

        X, Y = np.meshgrid(nn, mm)
        Z = eta
        cmap='jet'
        if title == 'Difference':
            cmap='Reds'
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto')
        ax.set_xlabel('Angular Velocity [rpm]', fontsize=16)
        ax.set_ylabel('Torque [Nm]', fontsize=16)
        ax.set_title(f'{title} Efficiency', fontsize=18)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Efficiency (%)', fontsize=16)
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

def y2_score(eta_diff):
    
    variance = np.mean(eta_diff ** 2)
    score = np.sqrt(variance)
    return score

def y1_score(df_predictions_y1, df_test_y1_targets, title):
    index=df_test_y1_targets.index
    score=0
    scores=[]
    for index_no in range(len(index)):
        target_values = df_test_y1_targets.loc[index[index_no]].tolist() 
        prediction_values =  df_predictions_y1.loc['Average'].tolist() if title == 'Baseline' else  df_predictions_y1.loc[index[index_no]].tolist()

        deviations = np.array(prediction_values) - np.array(target_values)
        variance = np.mean(deviations ** 2)
        std_dev = np.sqrt(variance)
        scores.append(std_dev)
        score+=std_dev
    
    y1_score=score/len(index)
    return y1_score, scores

def plot_mgrenz_statistics(df_y1_pred, df_test_y1_targets, plot, model):
    
    nn_kpi_2d = list(range(0, 19100, 100))
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Create a twin axis for RMSE
    ax2 = ax1.twinx()
    
    if model == 'Baseline':
        df_y1_pred_avg = df_y1_pred
        squared_deviations = (df_y1_pred_avg.iloc[0] - df_test_y1_targets)**2
        non_nan = squared_deviations.count()#Nan values if exists are not counted
        element_wise_rmse = (squared_deviations / non_nan)**0.5
        for i in range(len(element_wise_rmse)):
            ax2.plot(nn_kpi_2d, element_wise_rmse.iloc[i], 
                    linestyle='--', alpha=0.7)
        if plot == 'RMSE':
            ax2.set_ylabel(f'Y1 {model} {plot}', color='red', fontsize=12)
        else:
            ax2.set_ylabel(f'Y1 {plot}', color='red', fontsize=12)
        
    else:
        df_predictions_y1_avg = df_y1_pred.mean()
        df_y1_pred_avg = df_predictions_y1_avg.to_frame(name='Average MLP Prediction').transpose()
        element_wise_rmse=[]
        for i in range(len(df_y1_pred)):
            squared_deviations = (df_y1_pred.iloc[i] - df_test_y1_targets.iloc[i])**2
            non_nan = squared_deviations.count()
            rmse = (squared_deviations / non_nan)**0.5
            element_wise_rmse.append(rmse)
        # Plot element-wise RMSE for each test sample
        for i in range(len(element_wise_rmse)):
            ax2.plot(nn_kpi_2d, element_wise_rmse[i], 
                    linestyle='--', alpha=0.7) 
        ax2.set_ylabel(f'Y1 {plot}', color='red', fontsize=12)
        
        
    # Calculate mean RMSE to identify overlap wih target
    mean_rmse= ((df_y1_pred_avg.iloc[0] - df_test_y1_targets)**2).mean(axis=0, skipna=True)**0.5


    # Plot average on the first y-axis
    ax1.plot(nn_kpi_2d, df_y1_pred_avg.iloc[0], label=f'{model} Target Mean', color='blue', linewidth=2)

    # Plot standard deviation as shaded area
    ax1.fill_between(nn_kpi_2d, 
                    df_y1_pred_avg.iloc[0] - mean_rmse, 
                    df_y1_pred_avg.iloc[0] + mean_rmse,
                    
                    alpha=0.2, color='red', label=f'± Mean {plot}')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Angular Velocity (rpm)', fontsize=14)
    ax1.set_ylabel('Torque (Nm)', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=14)
    ax1.spines['top'].set_visible(False)
    
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.spines['right'].set_color('red')  # Color the right spine red
    ax2.spines['right'].set_linestyle('--')  
    ax2.spines['top'].set_visible(False)  # Hide the top spine for the twin axis

    #plt.title(f'Average {plot} and Element-wise {plot} of Test Dataset with {model} Model', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(f'./Manuscript/ReportImages/{plot}_{model}_y1.png', bbox_inches='tight')
    plt.show()
    
def plot_kpi3d_stddev(y2_grid_avg, y2_grid, plot, model):
    
    squared_deviations = (y2_grid_avg - y2_grid)**2
    rmse = (np.nanmean(squared_deviations, axis=0))**0.5

    plt.figure(figsize=(10, 6))

    x_min, x_max = 0, rmse.shape[1]
    y_min, y_max = 0, rmse.shape[0]

    im = plt.imshow(rmse, cmap='Reds', extent=[x_min, x_max, y_min, y_max], aspect='auto', origin='lower')

    plt.colorbar(im, label='Standard Deviation')
    # plt.title(f'{plot} of Random Samples from {model}')
    x_ticks = np.arange(x_min, x_max, 20)  # Create x-ticks at 20 interval
    plt.xticks(x_ticks, [str(int(val * 100)) for val in x_ticks]) 
    plt.xlabel('Angular Velocity (rpm)')
    plt.ylabel('Torque (Nm)')

    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig('./Manuscript/ReportImages/pos_stddev_y2.png', bbox_inches='tight')
    
    plt.show()
    
def plot_scores(scores, target, model):

    if model == 'Baseline':
        x_label = f'{target} {model}'
    else:
        x_label = f'{target}'
    
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, kde=True, bins=20*3)  # Plot histogram with KDE for scores
    plt.xlabel(f'{x_label} RMSE')
    plt.ylabel('Count')
    # plt.title(f'{model} {target} Score Distribution')
    
    # Hide top and right spines for the plot
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f'./Manuscript/ReportImages/score_{model}_{target}.png', bbox_inches='tight')
    plt.show()
    
    if target == 'Y1':
        min_value = joblib.load('./Intermediate/min_mgrenz.pkl')
        max_value = joblib.load('./Intermediate/max_mgrenz.pkl')
    elif target == 'Y2':
        min_value = 0
        max_value = 100
    percentage_diff = pdiff_scores(scores, min_value, max_value)  # Efficiency percentage range
    
    # Plot for percentage differences
    plt.figure(figsize=(10, 6))
    sns.histplot(percentage_diff, kde=True, bins=20*3, color='r')  # Plot histogram with KDE for percentage differences
    plt.xlabel(f'{x_label} Percentage Difference (%)')
    plt.ylabel('Count')
    # plt.title(f'{model} {target} Percentage Difference Distribution')
    
    ax2 = plt.gca()  # Get current axis for percentage diff plot
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig(f'./Manuscript/ReportImages/percentage_diff_{model}_{target}.png', bbox_inches='tight')
    plt.show()
    
    
def plot_eta_mean_statistics(speed_ranges, mean_eta, std_eta, title):
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(speed_ranges, mean_eta, yerr=std_eta, fmt='o', capsize=5, label="Mean ± Std Dev", ecolor='red', linestyle='--', marker='s')
    plt.xlabel("Angular Velocity (rpm) ")
    plt.ylabel("Efficiency(%)")
    # plt.title(f"Standard Deviation of {title} ETA values ranging NN Angular Velocity")
    plt.xticks(speed_ranges, [label * 100 for label in speed_ranges])
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper right', fontsize=14, bbox_to_anchor=(1.05, 1.05), borderaxespad=0.)
    plt.savefig(f'./Manuscript/ReportImages/stddev_y2_nn_{title}.png', bbox_inches='tight')
    plt.show()
    

def plot_eta(predicted_etas, target_etas, ax1, n, model):
    
    min_shape = min(predicted_etas.shape[0], min(target_eta.shape[0] for target_eta in target_etas)) if model == 'Baseline' else min(min(eta.shape[0] for eta in predicted_etas), min(target_eta.shape[0] for target_eta in target_etas))
    predicted_etas = clean_eta(predicted_etas[:min_shape]) if model == 'Baseline' else np.array([clean_eta(eta[:min_shape]) for eta in predicted_etas]) 
    mean_eta = np.array(predicted_etas) if model == 'Baseline' else np.mean(predicted_etas, axis=0)

    target_etas = np.array([clean_eta(eta[:min_shape]) for eta in target_etas])
    
    mm = np.linspace(0, len(mean_eta), len(mean_eta))

    # Create a twin axis for RMSE
    ax2 = ax1.twinx()
    
    if model == 'Baseline':
        element_wise_rmse=[]
        for i in range(target_etas.shape[0]):
            squared_deviations = (target_etas[i] - predicted_etas)**2
            non_nan = len(squared_deviations)
            rmse = (squared_deviations / non_nan)**0.5
            element_wise_rmse.append(rmse)
        for i in range(len(element_wise_rmse)):
            ax2.plot(mm, element_wise_rmse[i], 
                    linestyle='--', alpha=0.7) 
        ax2.set_ylabel(f'Y2 {model} RMSE', color='red', fontsize=12)
        
    else:
        element_wise_rmse=[]
        for i in range(target_etas.shape[0]):
            squared_deviations = (target_etas[i] - predicted_etas[i])**2
            non_nan = len(squared_deviations)
            rmse = (squared_deviations / non_nan)**0.5
            element_wise_rmse.append(rmse)
        # Plot element-wise RMSE for each test sample
        for i in range(len(element_wise_rmse)):
            ax2.plot(mm, element_wise_rmse[i], 
                    linestyle='--', alpha=0.7) 
        ax2.set_ylabel('Y2 RMSE', color='red', fontsize=12)
        
    # Calculate mean RMSE to identify overlap wih target
    mean_rmse= ((mean_eta - target_etas)**2).mean(axis=0)**0.5

    # Plot average on the first y-axis
    ax1.plot(mm, mean_eta, label=f'Mean', color='blue', linewidth=2)

    # Plot standard deviation as shaded area
    ax1.fill_between(mm, 
                    mean_eta - mean_rmse, 
                    mean_eta + mean_rmse,
                    
                    alpha=0.2, color='red', label=f'± Average RMSE')
    
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 10)  # Scale of the freaky y axis
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Torque (Nm)', fontsize=12)
    ax1.set_ylabel('Efficiency(%)', fontsize=12)
    ax1.set_title(f'Efficiency at Angular Velocity {n} rpm', fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.spines['right'].set_color('red')  # Color the right spine red
    ax2.spines['right'].set_linestyle('--')  
    ax2.spines['top'].set_visible(False)  # Hide the top spine for the twin axis

    
def plot_eta_statistics(eta, target_eta, speed_ranges, input):
    num_plots = len(eta)

    num_cols = min(3, num_plots)  
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows))
    # fig.suptitle(f'RMSE of {input} Positive Efficiency across NN ranges', fontsize=16)

    if num_plots > 1:
        axs = axs.flatten()
        
    n=speed_ranges[0] * 100
    for i in range(len(eta)):
        if num_plots > 1:
            ax = axs[i]
        else:
            ax = axs
        plot_eta(eta[i], target_eta[i], ax, n, input)
        n+= 2000

    # Remove any unused subplots
    if num_plots > 1:
        for j in range(num_plots, len(axs)):
            fig.delaxes(axs[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
  
    plt.savefig(f'./Manuscript/ReportImages/rmse_eta_{input}.png', bbox_inches='tight')
    plt.show()
    
def y1_folds_deviations(y1_cross_fold, model_name="MLP Model"):
    nn_kpi_2d = list(range(0, 19100, 100))
    plt.figure(figsize=(10, 6))

    # Calculate the mean prediction across folds
    df_y1_pred_avg = y1_cross_fold.mean()

    # Plot the mean curve
    plt.plot(nn_kpi_2d, df_y1_pred_avg, label='Mean Prediction', color='blue', linewidth=2)

    # Colors for each fold
    colors = sns.color_palette("husl", len(y1_cross_fold))

    # Plot each fold individually
    for i in range(len(y1_cross_fold)):
        plt.plot(nn_kpi_2d, y1_cross_fold.iloc[i], label=f'Fold {i+1} Prediction', color=colors[i], linestyle='--')

    # Plot settings
    plt.ylabel('Torque (Nm)', fontsize=12)
    plt.xlabel('Angular Velocity (rpm)', fontsize=12)
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig('./Manuscript/ReportImages/folds_dev_y1.png', bbox_inches='tight')
    plt.show()

def plot_y2_folds_deviations(predicted_etas, ax, n, colors):
    min_shape = min(eta.shape[0] for eta in predicted_etas)
    predicted_etas = clean_eta(np.array([clean_eta(eta[:min_shape]) for eta in predicted_etas])) 
    mean_eta = np.mean(predicted_etas, axis=0)

    mm = np.linspace(0, len(mean_eta), len(mean_eta))

    # Plot each fold without displaying individual legends
    for i in range(len(predicted_etas)):
        ax.plot(mm, predicted_etas[i], label=f'Fold {i+1} Prediction', color=colors[i], linestyle='--')
        
    # Plot average on the first y-axis
    ax.plot(mm, mean_eta, label='Mean', color='blue', linewidth=2)

    ax.set_ylim(0, 100)
    ax.set_xlabel('Torque (Nm)', fontsize=14)
    ax.set_ylabel('Efficiency (%)', fontsize=14)
    ax.set_title(f'Efficiency at Angular Velocity {n} rpm', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)

# Function to generate the subplots for different speed ranges
def y2_folds_deviations(eta, speed_ranges):
    num_plots = len(eta)
    num_cols = min(3, num_plots)
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows))
    colors = sns.color_palette("husl", len(eta[0]))  # Adjust colors to match the folds

    if num_plots > 1:
        axs = axs.flatten()

    n = speed_ranges[0] * 100
    for i in range(len(eta)):
        ax = axs[i] if num_plots > 1 else axs
        plot_y2_folds_deviations(eta[i], ax, n, colors)
        n += 2000

    # Remove any unused subplots
    if num_plots > 1:
        for j in range(num_plots, len(axs)):
            fig.delaxes(axs[j])

    # Add a single legend outside the subplots
    handles, labels = axs[0].get_legend_handles_labels() if num_plots > 1 else axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=14, ncol=len(labels), bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the legend
    plt.savefig('./Manuscript/ReportImages/folds_dev_y2.png', bbox_inches='tight')
    plt.show()