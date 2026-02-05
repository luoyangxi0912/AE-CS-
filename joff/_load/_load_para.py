# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import os
import torch
import numpy as np
from joff._func._string import _get_suffixs

def _set_path_for_best(self, mode):
    if mode == 'mult': data = self.mult_df.values[:,len(self.train_loss_hd):]
    else: data = self.test_perf_df.values
    
    if self.task == 'cls': row = np.argmax(data[:,0])
    elif self.task == 'prd': row = np.argmin(data[:,0])
    elif self.task == 'fd':
        AMDR = data[:,1::2]
        index = np.argmin(AMDR)     # global min AMDR
        row, column = int( index / AMDR.shape[1] ), np.mod(index, AMDR.shape[1])    # find row and column
        i = int(column/2)
        sub_path = self.fd_pr_name[i]
        
    if mode == 'mult': self._save_path = self.save_path + _get_suffixs(row + 1)
    elif self.task == 'fd': self._save_path = self.save_path + '/' + sub_path

def _find_file_in_path(_path, _file):
    file = None
    if type(_file) == tuple: _path, _file = _file[0], _file[1]
    if os.path.exists(_path):
        # list all subfolder and file under the folder
        file_list = os.listdir(_path)
        for file_name in file_list:
            if _file in file_name:
                file_path = _path + '/' + file_name
                if os.path.isfile(file_path):
                    file = file_path
    return file

def _load_module_para(self, _type = 'last', sub = None, file_name = None, replace = None):
    # 'sub' only works for task = 'fd' and _type = 'best'
    _path = self._save_path + '/' + self.fd_pr_name[sub]\
        if sub is not None else self._save_path

    if replace is not None:
        _path = _path.replace(replace[0], replace[1])

    # first judge if file exist
    if file_name is None: file_name = _type.capitalize()
    file = _find_file_in_path(_path, file_name)
     
    if file is not None:
        self.load_state_dict(torch.load(file))
        print("\nLoad \033[4mmodel paras\033[0m from '{}'".format(file))

def _load_module(file):
    return torch.load(file)