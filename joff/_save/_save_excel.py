# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import os
import numpy as np
import pandas as pd
from joff._load._load_para import _find_file_in_path
from joff._save._print_style import _clickable_file_link, _cstr

def _save_excel(path, file, data, sheet_name = None):
    if not os.path.exists(path): os.makedirs(path)
    file_name = path + '/' + file

    if type(data) == list:
        for i in range(len(data)):
            if type(data[i]) == np.ndarray: data[i] = pd.DataFrame(data[i])
    elif type(data) == np.ndarray: data = [pd.DataFrame(data)]
    else: data = [data]

    _sheet_name = ['Sheet1'] if sheet_name is None else sheet_name
    
    writer = pd.ExcelWriter(file_name, engine='openpyxl')
    for i, _name in enumerate(_sheet_name):
        df = data[i]
        if_index = len(df.index) != 0 if hasattr(df, 'index') else False
        if_column = len(df.columns) != 0 if hasattr(df, 'columns') else False
        data[i].to_excel(excel_writer = writer, sheet_name = _name, \
                         index = if_index, header = if_column)
        # data[i].to_excel(excel_writer=writer, sheet_name=_name, encoding="utf-8", \
        #                  index=index, header=columns)
    # writer.save()
    writer.close()

    clickable_text = _clickable_file_link(file_name)
    # print("\nSave \033[4m{}\033[0m in '{}'".format(file, path))
    print(f"\nSave {_cstr( file,'草绿色','/')} in '{clickable_text}'")
    # print(f"Plot {clickable_text} in '../Result/DAE_HY/selected_fd_Act1/re-T2-kde'")

def _read_excel(file_path):
    data = pd.read_excel(file_path, sheet_name = None)
    if type(data) == dict:
        return [data[key].values[:,1:] for key in data.keys()]
    else:
        return data.values

def _concat_excel(path, file, df):
    if not os.path.exists(path): os.makedirs(path)
    file_name = path + '/' + file
    if _find_file_in_path(path, file) is not None:
        df0 = pd.read_excel(file_name, index_col = 0)
        df = pd.concat([df0, df], axis=0)

    writer = pd.ExcelWriter(file_name, engine='openpyxl')
    index, columns = len(df.index) != 0, len(df.columns) != 0
    df.to_excel(excel_writer=writer, index=index, header=columns)
    # df.to_excel(excel_writer=writer, encoding="utf-8", index=index, header=columns)
    # writer.save()
    writer.close()