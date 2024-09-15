import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from tqdm import tqdm
import os
import pandas as pd

def read_files_kpi2d(file_path, kpi2):##can remove this TODO less likely necessary
    ## For 2d KPI we require only 1 row of NN to plot the 2d graph
    wb = openpyxl.load_workbook(file_path)

    sheet_mgrenz = wb[kpi2]
    mgrenz_values = [cell.value for cell in sheet_mgrenz[1] if cell.value is not None]

    sheet_nn = wb['NN']
    nn_values = [cell.value for cell in sheet_nn[1] if cell.value is not None]

    min_length = min(len(mgrenz_values), len(nn_values))
    mgrenz_values = mgrenz_values[:min_length]
    nn_values = nn_values[:min_length]
    
    return nn_values, mgrenz_values


def read_file_kpi3d(file_path, sheet_name):
    # For 3d KPI we need the whole matrix of all 3 dimensions
    wb = openpyxl.load_workbook(file_path)
    sheet = wb[sheet_name]
    data = [[cell.value if cell.value is not None else np.nan for cell in row] for row in sheet.iter_rows()]
    return np.array(data, dtype=float)

# def plot_kpi2d(nn_values, mgrenz_values):
#     plt.figure(figsize=(10, 6))
#     plt.plot(nn_values, mgrenz_values, marker='o')
#     plt.xlabel('NN Values')
#     plt.ylabel('Mgrenz Values')
#     plt.title(f'NN vs Mgrenz')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
    

def plot_kpi3d(nn, mm, eta):
    fig, ax = plt.subplots(figsize=(16, 10))

    # # Remove the first row and column if headers
    X = nn[0, 1:]
    Y = mm[1:, 0]
    Z = eta[1:, 1:]
    
    # X = nn
    # Y = mm
    # Z = eta[1:, 1:]

    X, Y = np.meshgrid(X, Y)

    mask = np.isfinite(Z)
    X, Y, Z = X[mask], Y[mask], Z[mask]

    contour = ax.tricontourf(X.ravel(), Y.ravel(), Z.ravel(), levels=20, cmap='jet')
    lines = ax.tricontour(X.ravel(), Y.ravel(), Z.ravel(), levels=10, colors='black', linewidths=0.5)
    
    ax.clabel(lines, inline=True, fontsize=8, fmt='%.2f', colors='white',
               inline_spacing=3, use_clabeltext=True)
    
    for text in ax.findobj(match=plt.Text):
        text.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])

    ax.set_xlabel('Angular Velocity [rpm]', fontsize=12)
    ax.set_ylabel('Torque [Nm]', fontsize=12)
    ax.set_title('Motor Efficiency', fontsize=14)

    cbar = fig.colorbar(contour)
    cbar.set_label('Efficiency', fontsize=12)

    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.15)

    plt.show()
    
def plot_testdataset_kpi2d(df_targets, df_predictions,start,end):
    
        nn_kpi_2d = list(range(0, 19100, 100)) # NN values alyways range from 0 to 19000 rpm
        
        index=df_targets.index

        cols=2
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
    
def plot_kpi2d(nn_values, mgrenz_values, predicted_mgrenz):
    plt.figure(figsize=(10, 6))
    plt.plot(nn_values, mgrenz_values, label='Target', color='blue')
    plt.plot(nn_values, predicted_mgrenz, label='Predictions', color='red')
    plt.xlabel('NN Values')
    plt.ylabel('Mgrenz Values')
    plt.title(f'NN vs Mgrenz')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def remove_faulty_files(directory):
    # Loop over all files to check if there are any faulty files and remove them
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory, filename)
            try:
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                if "ETA" not in sheet_names or "MM" not in sheet_names or "input_data" not in sheet_names or "NN" not in sheet_names or "Mgrenz" not in sheet_names:
                    print(f"{filename} is missing required sheets.")
                    os.remove(file_path)
                    print(f"{filename} removed.")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
def plot_kpi3d_dual(nn, mm1, eta1, mm2, eta2, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(f'Motor Efficiency - {filename}', fontsize=16)

    for ax, mm, eta, title in zip([ax1, ax2], [mm1, mm2], [eta1, eta2], ['Original', 'Predicted']):
        X = nn[0, 1:]
        Y = mm[1:, 0]
        Z = eta[1:, 1:]
   
        X, Y = np.meshgrid(X, Y)
        mask = np.isfinite(Z)
        X, Y, Z = X[mask], Y[mask], Z[mask]
        contour = ax.tricontourf(X.ravel(), Y.ravel(), Z.ravel(), levels=20, cmap='jet')
        lines = ax.tricontour(X.ravel(), Y.ravel(), Z.ravel(), levels=10, colors='black', linewidths=0.5)
   
        ax.clabel(lines, inline=True, fontsize=8, fmt='%.2f', colors='white',
                   inline_spacing=3, use_clabeltext=True)
   
        for text in ax.findobj(match=plt.Text):
            text.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])
    
        ax.set_xlabel('Angular Velocity [rpm]', fontsize=12)
        ax.set_ylabel('Torque [Nm]', fontsize=12)
        ax.set_title(f'{title} Efficiency', fontsize=14)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Efficiency', fontsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set_xticklabels(ax.get_xticks(), rotation=45, ha='right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()
