# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
import numpy as np

# convert to numpy
def _2np(v):
    # 获取实际的数据
    if hasattr(v, 'data'):
        v = v.data
    # 根据类型处理
    if type(v) == dict:
        for key in v.keys():
            v[key] = _2np(v[key])
        return v
    elif isinstance(v, torch.Tensor):
        return v.cpu().numpy()
    elif isinstance(v, (np.ndarray, memoryview)):
        return np.asarray(v)
    else:
        return np.asarray(v)

# element-wise search A in B
def _search_AinB_loc(B, A):
    loc = np.array(A).copy()
    loc[np.isin(loc, B, invert = True)] = -1
    for i, b in enumerate(B):
        loc[loc == b] = i
    return loc.astype(int)

def _concat_v(v):
    for i in range(len(v)):
        v[i] = v[i].reshape(-1,1)
    return np.concatenate(v, axis = 1)
        

if __name__ == '__main__':
    A = np.array([0, 1, 3, 3, 4, 2, 1, 5])
    B = [2, 1, 0]
    C = [1, 0, 2, 3]
    print(_search_AinB_loc(B, A))
    print(_search_AinB_loc(C, A))
    print(A)
