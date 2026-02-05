# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 18:38:04 2022

@author: Fuzz4
"""

import torch
import numpy as np
from joff._func._dvc import _2dvc
from joff._func._np import _2np
from joff._func._msg import _init_msg, _update_msg, _record_msg

def batch_training(self, loader = 'train'):
    _2dvc(self, self.dvc)
    if not self.training: self.train()
    if loader == 'train': loader = self.train_loader
    self.cur_loader = loader
    
    _init_msg(self)
    for b_id, (input, label) in enumerate(loader):
        input, label = _2dvc(input, self.dvc), _2dvc(label, self.dvc)
        self._label = label
        self.cnt_batch += 1

        if hasattr(self, '_forward'):
            self._forward(input)    # custom train
        else:
            self.zero_grad()
            self.forward(input)
            self.loss.backward()
            self.opti.step()
        
        _update_msg(self, b_id, input.size(0), len(loader))
        
    _record_msg(self, 'train', loader.X.shape[0])
    
def batch_testing(self, loader = 'test'):
    _2dvc(self, self.dvc)
    if self.training: self.eval()
    _msg_phase = 'test'
    phase = loader if type(loader) == str else ''
    if loader == 'test':
        loader = self.test_loader
    elif loader == 'train':
        _msg_phase = 'no'
        loader = self.unshuf_train_loader if hasattr(self,'unshuf_train_loader') else self.train_loader

    self.cur_loader = loader
    In, Out, Latent, Label, Cust_MM, Cust_TS = [], [], [], [], [], []
    with torch.no_grad():
        _init_msg(self)
        for b_id, (input, label) in enumerate(loader):
            input, label = _2dvc(input, self.dvc), _2dvc(label, self.dvc)
            self._label = label
            
            # customize test
            output = self.forward(input)
            
            In.append(_2np(input))
            Out.append(_2np(output))
            Label.append(_2np(self._label))

            if hasattr(self, '_latent'): Latent.append(_2np(self._latent))
            if hasattr(self, '_cust_mm'): Cust_MM.append(_2np(self._cust_mm))
            if hasattr(self, '_cust_ts'): Cust_TS.append(_2np(self._cust_ts))
            
            _update_msg(self, b_id, input.size(0), len(loader))
            
        _record_msg(self, _msg_phase, loader.X.shape[0])
    In = np.concatenate(In, 0)
    Label = np.concatenate(Label, 0)
    try:
        Out = np.concatenate(Out, 0)
    except ValueError:
        Out = []
    if hasattr(self, '_latent'): Latent = np.concatenate(Latent, 0)
    if hasattr(self, '_cust_mm'): fd_dt2arr(self, Cust_MM, self._cust_mm, 'mm', phase)
    if hasattr(self, '_cust_ts'): fd_dt2arr(self, Cust_TS, self._cust_ts, 'ts', phase)
    return In, Out, Latent, Label

def fd_dt2arr(self, dict_list, dict_data, var, phase):
    fd_phase = 'offline' if phase == 'train' else 'online'
    if type(dict_data) != dict:
        return exec('self.' + fd_phase + '_cust_'+ var + '= np.concatenate(Data, 0)')
    for key in dict_data.keys():
        Data_key_list = []
        # 把 list[dict{key: array}] 转换为 list_key[array]
        for i in range(len(dict_list)):
            Data_key_list.append(_2np(dict_list[i][key]))
        # 记录 self.online_key = array
        exec('self.' + fd_phase + '_' + key + '= np.concatenate(Data_key_list, 0)')

if __name__ == '__main__':
    from joff._nn._fcnn_module import fcnn_example
    module = fcnn_example()