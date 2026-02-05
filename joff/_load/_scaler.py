# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def _dataset_fit_transform(train_data, test_data = None, _scaler = 'st',
                           n_cate = None, data_path = None, data_char = 'X'):
    Scaler = None
    train_list, test_list = [], []
    train_data = train_data.copy()
    if test_data is None: test_data = train_data.copy()
    else: test_data = test_data.copy()
    if _scaler == 'oh':
        for i in range(len(train_data)): train_list.append(np.eye(n_cate)[train_data[i]])
        for i in range(len(test_data)): test_list.append(np.eye(n_cate)[test_data[i]])
    else:
        train_array = np.concatenate(train_data, axis=0)
        if _scaler == 'mm': Scaler = MinMaxScaler()
        elif _scaler == 'st': Scaler = StandardScaler()
        Scaler.name = _scaler
        Scaler.fit(train_array)
        if data_path is not None:
            _save_scaler(Scaler, data_path, data_char)

        for i in range(len(train_data)): train_list.append( Scaler.transform(train_data[i]) )
        for i in range(len(test_data)): test_list.append( Scaler.transform(test_data[i]) )

    return train_list, test_list, Scaler

def _scaler_fit_transform(data_char, D, p):
    Scaler = None
    if p['scaler'] is None: return Scaler

    if data_char == 'X':
        _scaler = p['scaler'][0]
        train_data, test_data = D.train_X0, D.test_X0
    else:
        _scaler = p['scaler'][1]
        train_data, test_data = D.train_Y0, D.test_Y0

    train_list, test_list, Scaler = _dataset_fit_transform(train_data, test_data, _scaler,
                                                           D.n_cate, p['data_path'], data_char)

    if _scaler is not None:
        if data_char == 'X':
            D.train_X0, D.test_X0 = train_list, test_list
        else:
            D.train_Y0, D.test_Y0 = train_list, test_list
    return Scaler

def _save_scaler(Scaler, path, data_char):
    if Scaler.name == 'st':
        np.savetxt(path + '/'+ data_char + '_scaler[{}].csv'.format(Scaler.name),
                   np.concatenate([Scaler.mean_.reshape(-1, 1), Scaler.var_.reshape(-1, 1)], 1),
                   fmt='%f', delimiter=',')
        # print('st:', '\nmean = ', scaler.mean_, '\nvar = ', scaler.var_)
    elif Scaler.name == 'mm':
        np.savetxt(path + '/'+ data_char + '_scaler[{}].csv'.format(Scaler.name),
                   np.concatenate([Scaler.scale_.reshape(-1, 1), Scaler.min_.reshape(-1, 1)], 1),
                   fmt='%f', delimiter=',')

def _read_scaler(path, scaler_type, data_char):
    data = np.loadtxt(path + '/'+ data_char + '_scaler[{}].csv'.format(scaler_type), delimiter=',')
    if scaler_type == 'st':
        Scaler = StandardScaler()
        Scaler.mean_ = data[:, 0]; Scaler.var_ = data[:, 1]
    elif scaler_type == 'mm':
        Scaler = MinMaxScaler()
        Scaler.scale_ = data[:, 0]; Scaler.min_ = data[:, 1]
    Scaler.name = scaler_type
    return Scaler

def _transform(Scaler, X):
    # 'st': (X - Scaler.mean_) / np.sqrt(Scaler.var_)
    # 'mm': X * Scaler.scale_ + Scaler.min_
    if type(X) != list: return Scaler.transform(X)
    _X = []
    for x in X: _X.append(Scaler.transform(x))
    return _X

def _inverse_transform(Scaler, X):
    # 'st': X * np.sqrt(Scaler.var_) + Scaler.mean_
    # 'mm': (X - Scaler.min_) / Scaler.scale_
    if type(X) != list: return Scaler.inverse_transform(X)
    _X = []
    for x in X: _X.append(Scaler.inverse_transform(x))
    return _X