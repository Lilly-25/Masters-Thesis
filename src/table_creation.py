import pandas as pd
import openpyxl
import re
import os
import csv
import numpy as np
from src.utils import cumulative_counts as cumulative_col_count

def create_tabular_data(file_path, purpose):
    try:
        df = pd.read_excel(file_path, sheet_name='input_data', header=None)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        params_dict = {}
        for _, row in df.iterrows():
            param, value = row[0], row[1]
            if pd.notna(param) and pd.notna(value):
                try:
                    params_dict[param] = float(value)
                except ValueError:
                    params_dict[param] = value
                    
        params_rotor_v=['lmsov', 'lth1v', 'lth2v', 'r1v', 'r11v', 'r2v', 'r3v', 'r4v', 'rmt1v', 'rmt4v', 'rlt1v', 'rlt4v', 
                      'hav', 'mbv', 'mhv', 'rmagv',
                        'dsmv', 'dsmuv', 'amtrv', 'dsrv', 'deg_phiv', 'lmav', 'lmiv', 'lmov', 'lmuv']
        
        params_rotor_delta=['lmsob','lthb', 'r2b', 'r3b', 'r4b', 'r5b', 'lgr3b', 'lgr4b',
                            'mbb', 'mhb', 'mtbb', 'rmagb',
                            'amtrb', 'dsr3b', 'dsr4b', 'deg_phi3b', 'deg_phi4b', 'lmob', 'lmub', 'lmsub']
        
        params_stator=['airgap', 'b_nng', 'b_nzk', 'b_s', 'h_n','h_s', 'r_sn', 'r_zk', 'r_ng', 'h_zk', 'bhp', 'hhp', 'rhp',
                'dhphp', 'dhpng']
        
        params_general=['N', 'simQ','r_a', 'r_i']
 
        df_features=pd.DataFrame()
        
        #V represent max double v magnets and b max delta magnets
        v=2
        b=1
        while v >= 1:
            for key in params_rotor_v:
                df_features.loc[file_name, f"{key}{v}"] = params_dict.get(f"{key}{v}", 0)
            v -= 1
        while b >= 1:
            for key in params_rotor_delta:
                df_features.loc[file_name, f"{key}{b}"] = params_dict.get(f"{key}{b}", 0)
            b -= 1
        for key in params_stator:
            df_features.loc[file_name, f"{key}"] = params_dict.get(f"{key}", 0)
        for key in params_general:
            df_features.loc[file_name, f"{key}"] = params_dict.get(f"{key}", 0)
        
        # Filter all 'deg_phi' columns and create new columns with their corresp radian values
        deg_columns = df_features.filter(regex='^deg_phi').columns

        for col in deg_columns:
            new_col_name = col.replace('deg_', 'rad_')
            df_features[new_col_name] = np.radians(df_features[col])
            
        df_features = df_features.drop(columns=deg_columns)
        #print(df_features.head())
        wb = openpyxl.load_workbook(file_path)
        sheet_mgrenz = wb['Mgrenz']
        mgrenz_values = [cell.value for cell in sheet_mgrenz[1] if cell.value is not None]
        max_mgrenz = max(mgrenz_values)
            
        columns = [f'Column_{i + 1}' for i in range(len(mgrenz_values))]
  
        df_targets = pd.DataFrame(columns=columns)
        df_targets.loc[file_name] = mgrenz_values
        
        max_cols=191
       
        if len(mgrenz_values)!= max_cols:
            print('We have a problem Houston!')
            
        if purpose=='train':
            ##Checking MM sheet to get the correct grid for KPI ETA    
            sheet_mm = wb['MM']

            #Finding out correct indices of the ETA grid
            def check_rows(sheet, start_row, end_row, mgrenz):
                for row in sheet.iter_rows(min_row=start_row, max_row=end_row):
                    first_cell_value = row[0].value  # Get the value of the first cell in the row
                    if first_cell_value == mgrenz:
                        index = row[0].row
                        return index
                

            # Check the first 5 rows
            min_index = check_rows(sheet_mm, 1, 5, -max_mgrenz)

            # Check the last 5 rows
            last_row = sheet_mm.max_row
            max_index = check_rows(sheet_mm, last_row - 4, last_row, max_mgrenz)

            cumulative_counts = cumulative_col_count(mgrenz_values)
            
            #load the eta grid
            sheet_eta = wb['ETA']
            eta_grid_folder = './data/TabularDataETAgrid/'
            if not os.path.exists(eta_grid_folder):
                os.makedirs(eta_grid_folder)
            eta_grid = os.path.join(eta_grid_folder, f"{file_name}.csv")##Instead of saving each grid into a file appeand to a numpy array and save as npy file can access test_size only ased on index
            ##think of an alternative way where we can also have filenames indexed into the numpy array or whatever pythonic object
            with open(eta_grid, mode='w', newline="") as file:
                writer = csv.writer(file)
                #Negative torque values
                for i, row in enumerate(sheet_eta.iter_rows(min_row=min_index, max_row=(min_index + max_index)//2 - 1, values_only=True)):
                    if i < len(cumulative_counts):
                        negative_columns = cumulative_counts[i]
                    else:
                        negative_columns = cumulative_counts[-1]
                    padded_eta = np.full(191, np.nan)
                    padded_eta[:negative_columns] = row[:negative_columns]
                    writer.writerow(padded_eta)
                    
                positive_eta_grid = []
                #Positive torque values..consider only this portion when mirroring
                for j, row in enumerate(reversed(list(sheet_eta.iter_rows(min_row=((min_index + max_index)//2), max_row=max_index, values_only=True)))):
                    if j < len(cumulative_counts):
                        positive_columns = cumulative_counts[j]
                    else:
                        positive_columns = cumulative_counts[-1]
                    padded_eta = np.full(191, np.nan)
                    padded_eta[:positive_columns] = row[:positive_columns]
                    positive_eta_grid.append(padded_eta)
                for row in reversed(positive_eta_grid): # Reverse the sliced grid back to the original file
                    writer.writerow(row)
        
        return df_features, df_targets
                
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return None
        