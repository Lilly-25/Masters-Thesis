import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

def read_files_kpi2d(file_path, sheet_name):##can remove this TODO less likely necessary
    ## For 2d KPI we require only 1 row of NN to plot the 2d graph
    wb = openpyxl.load_workbook(file_path)

    sheet_mgrenz = wb[sheet_name]
    mgrenz_values = [cell.value for cell in sheet_mgrenz[1] if cell.value is not None]

    sheet_nn = wb['NN']
    nn_values = [cell.value for cell in sheet_nn[1] if cell.value is not None]

    min_length = min(len(mgrenz_values), len(nn_values))
    mgrenz_values = mgrenz_values[:min_length]
    nn_values = nn_values[:min_length]
    
    return nn_values, mgrenz_values

def read_mm_nn(file_path, sheet_name):
    # We only need the 1st row from NN and 1st column from MM for plotting
    wb = openpyxl.load_workbook(file_path)
    sheet = wb[sheet_name]
    if sheet_name == 'MM':
        data = [cell.value for cell in sheet['A'] if cell.value is not None]
    else:
        data = [cell.value for cell in sheet[1] if cell.value is not None]
    return data

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
    
def plot_kpi2d(nn_values, mgrenz_values):
    plt.figure(figsize=(10, 6))
    plt.plot(nn_values, mgrenz_values, color='blue')
    # plt.plot(nn_values, mgrenz_values, label='Target', color='blue')
    # plt.plot(nn_values, predicted_mgrenz, label='Predictions', color='red')
    plt.xlabel('NN Values')
    plt.ylabel('Mgrenz Values')
    plt.title(f'NN vs Mgrenz')
    plt.grid(True)
    plt.tight_layout()
    # plt.legend()
    plt.show()

def plot_kpi3d(nn, mm, eta):
    fig, ax = plt.subplots(figsize=(16, 10))

    # # Remove the first row and column if headers
    X = nn[0, 1:]
    Y = mm[1:, 0]
    Z = eta[1:, 1:]

    X, Y = np.meshgrid(X, Y)

    mask = np.isfinite(Z)
    X, Y, Z = X[mask], Y[mask], Z[mask]

    contour = ax.tricontourf(X.ravel(), Y.ravel(), Z.ravel(), levels=100, cmap='jet')

    ax.set_xlabel('Angular Velocity [rpm]', fontsize=12)
    ax.set_ylabel('Torque [Nm]', fontsize=12)
    ax.set_title('Motor Efficiency', fontsize=14)

    cbar = fig.colorbar(contour)
    cbar.set_label('Efficiency', fontsize=12)

    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.15)

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
                

def artifact_deletion():
    wandb_folder = os.path.expanduser('~/.local/share/wandb')
    if os.path.exists(wandb_folder):
        for root, dirs, files in os.walk(wandb_folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(wandb_folder)
        print(f"Deleted the folder: {wandb_folder}")
    wandb_folder = os.path.expanduser('~/.cache/wandb')
    if os.path.exists(wandb_folder):
        for root, dirs, files in os.walk(wandb_folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(wandb_folder)
        print(f"Deleted the folder: {wandb_folder}")
    else:
        print(f"The folder {wandb_folder} does not exist.")
        
def cumulative_counts(arr):
    if len(arr) == 0:
        return []
    
    result = []
    count = 0
    for i in range(len(arr)):
        count += 1
        if i == len(arr) - 1 or arr[i] != arr[i + 1]:
            result.append(count)
    
    return result