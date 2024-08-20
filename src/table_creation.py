import pandas as pd
import openpyxl
import re
import os
import csv

def create_tabular_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='input_data', header=None)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        #file_name = os.path.basename(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(file_name)
        params_dict = {}
        for _, row in df.iterrows():
            param, value = row[0], row[1]
            if pd.notna(param) and pd.notna(value):
                try:
                    params_dict[param] = float(value)
                except ValueError:
                    params_dict[param] = value
                    
        params_rotor=['lmsov', 'lth1v', 'lth2v', 'r1v', 'r11v', 'r2v', 'r3v', 'r4v', 'rmt1v', 'rmt4v', 'rlt1v', 'rlt4v', 'hav', 'mbv', 'mhv', 'rmagv',
        'dsm', 'dsmu', 'amtrv', 'dsrv', 'deg_phi', 'lmav', 'lmiv', 'lmov', 'lmuv']
        params_stator=['airgap', 'b_nng', 'b_nzk', 'b_s', 'h_n','h_s', 'r_sn', 'r_zk', 'r_ng', 'h_zk', 'bhp', 'hhp', 'rhp',
                'dhphp', 'dhpng']
        params_general=['N', 'simQ','r_a', 'r_i']
        
        match_doubleV = re.search(r"doubleV", file_name, re.IGNORECASE)##TODO replace this logic by checking the data in the file
        match_singleV = re.search(r"singleV", file_name, re.IGNORECASE)
        match_tripleV = re.search(r"tripleV", file_name, re.IGNORECASE)
        
        if match_singleV:
            v=1
        elif match_doubleV:
            v=2
        elif match_tripleV:
            v=3
            
        df_features=pd.DataFrame()
        
        while v >= 1:##logic needs to be changed TODO
            for key in params_rotor:
                df_features.loc[file_name, f"{key}{v}"] = params_dict.get(f"{key}{v}", 0)
            v -= 1
        for key in params_stator:
            df_features.loc[file_name, f"{key}"] = params_dict.get(f"{key}", 0)
        for key in params_general:
            df_features.loc[file_name, f"{key}"] = params_dict.get(f"{key}", 0)
        
        wb = openpyxl.load_workbook(file_path)
        sheet_mgrenz = wb['Mgrenz']
        mgrenz_values = [cell.value for cell in sheet_mgrenz[1] if cell.value is not None]
            
        columns = [f'Column_{i + 1}' for i in range(len(mgrenz_values))]
  
        df_targets = pd.DataFrame(columns=columns)
        df_targets.loc[file_name] = mgrenz_values
       
        if len(mgrenz_values)!= 191:
            print('We have a problem Houston!')
            
        
        return df_features, df_targets
                
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return None
        