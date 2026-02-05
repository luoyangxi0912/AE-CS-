# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
import numpy as np
t_cpu = torch.FloatTensor 
t_cuda = torch.cuda.FloatTensor

def _Tensor(dvc):
    if type(dvc) == str: dvc = torch.device(dvc)
    return t_cuda if dvc == torch.device('cuda') else t_cpu

# move to device
def _2dvc(v, dvc):
    if type(dvc) == str: dvc = torch.device(dvc)
    # tensor
    if torch.is_tensor(v): cnt_dvc = v.device
    # module
    else: cnt_dvc = next(v.parameters()).device
    # move
    if cnt_dvc != dvc: return v.to(dvc)
    else: return v

# dvc for loader
def _dvc_dt(dvc):
    if type(dvc) == str: dvc = torch.device(dvc)
    if dvc == torch.device('cpu'): return {'pin_memory': False}
    else: return {'pin_memory': True, 'num_workers': 0}


if __name__ == '__main__':
    a = t_cuda(np.ones((1,2)))
    print(a, a.device)
    a = _2dvc(a, 'cpu')
    print(a, a.device)
    b = torch.nn.Linear(10, 20)
    print(b, next(b.parameters()).device)
    b = _2dvc(b, 'cuda')
    print(b, next(b.parameters()).device)