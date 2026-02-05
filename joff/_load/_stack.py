# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import numpy as np
from joff._nn._attr import _update_dict

def _stack_samples(D, p):
    D._check()
    D.stack = p['stack']
    if hasattr(D, 'u_dim'): p['u_dim'] = D.u_dim
    if type(D.train_X) == list:
        D.train_X, D.train_Y = _get_dynamic_list_data(D.train_X, D.train_Y, **p)
    else:
        D.train_X, D.train_Y = _get_dynamic_data(D.train_X, D.train_Y, p)

    if type(D.train_X) == list:
        D.test_X, D.test_Y = _get_dynamic_list_data(D.test_X, D.test_Y, **p)
    else:
        D.test_X, D.test_Y = _get_dynamic_data(D.test_X, D.test_Y, p)

def _get_dynamic_list_data(X, Y, **p):
    X_list, Y_list = [], []
    for i in range(len(X)):
        x, y = _get_dynamic_data(X[i], Y[i], p)
        if x is not None:
            X_list.append(x)
            Y_list.append(y)
    return X_list, Y_list

_dynamic_dt = {
    'stack': 12,
    'u_dim': None,
    'n_delay': 0,
    'stack_label': 'fd'
}
def _get_dynamic_data(X, L, p):
    if X.shape[0] < p['stack']: return None, None
    p = _update_dict(_dynamic_dt, p)
    if p['u_dim'] is not None and p['n_delay'] > 0:
        u_delay = X[:-p['n_delay'],:p['u_dim']]
        y_delay = X[p['n_delay']:,p['u_dim']:]
        X = np.concatenate([u_delay, y_delay], axis=-1)
        L = L[p['n_delay']:]

    N_stack = X.shape[0] - p['stack'] + 1
    dX = []
    # dX
    for i in range(p['stack']): dX.append(X[i:i + N_stack])
    dX = np.concatenate(dX, axis = 1)
    # dY
    if p['stack_label'] == 'prd': dL = L[p['stack'] - 1:] # take the last one as label
    elif p['stack_label'] == 'fd':                      # task the max one as label (有一个故障就认为故障)
        dL = []
        if_onehot = True if len(L.shape) > 1 and L.shape[1] > 1 else False
        if if_onehot:
            n_cate = L.shape[1]
            L = L.argmax(axis = 1)
        # 多个样本对应一个标签
        for i in range(N_stack): dL.append(np.max(L[i:i + p['stack']]))
        if if_onehot:
            dL = np.eye(n_cate)[np.array(dL)]           # get one-hot label
    return np.array(dX, dtype=float), \
        np.array(dL, dtype=int)
