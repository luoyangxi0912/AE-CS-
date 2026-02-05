# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import os
import numpy as np
import pandas as pd
import scipy.io as scio
from collections import Counter
from joff.customize import _load_dt
from joff._func._np import _search_AinB_loc
from joff._func._msg import _msg_code
from joff._load._scaler import _scaler_fit_transform
from joff._load._stack import _stack_samples
from joff._load._stats_data_attri import _stats_data_categories
from joff._load._relist_for_fd_test import reorganize_fault_dataset

class Data(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    # 复制 train_X0 等 到 train_X 等
    def _check(self):
        for key in ['train_X', 'train_Y', 'test_X', 'test_Y']:
            if not hasattr(self, key):
                setattr(self, key, eval('self.'+ key + '0.copy()'))
                
    def _length(self):
        if isinstance(self.train_X, list):
            train_length = [len(self.train_X[i]) for i in range(len(self.train_X))]
            test_length = [len(self.test_X[i]) for i in range(len(self.test_X))]
        else:
            train_length = [len(self.train_X)]
            test_length = [len(self.test_X)]
        return train_length, test_length
        
    def _array(self):
        array = []
        for data in [self.train_X, self.train_Y, self.test_X, self.test_Y]:
            if isinstance(data, list): array.append(np.concatenate(data))
            else: array.append(data)
        return tuple(array)

    # 获取分段点的索引
    def _split_p(self):
        if not hasattr(self, 'seg_len'): self.seg_len = tuple(self._length())
        if not hasattr(self, 'split_p'):
            self.split_p = []
            for seg_len in list(self.seg_len):
                seg_loc = []
                for i in range(len(seg_len)):
                    if i ==0: seg_loc.append(seg_len[i])
                    else: seg_loc.append(seg_loc[-1] + seg_len[i])
                self.split_p.append(seg_loc)
        # print(self.split_p)
        return tuple(self.split_p)

    def _print(self):
        if hasattr(self, 'array'): train_X, train_Y, test_X, test_Y = self.array
        else: train_X, train_Y, test_X, test_Y = self._array()

        _stats_data_categories(self.train_Y, "训练集")
        print('  总尺寸：Train_X:', train_X.shape, '， Train_Y:', train_Y.shape)

        _stats_data_categories(self.test_Y, "测试集")
        print('  总尺寸：Test_X:', test_X.shape, '， Test_Y:', test_Y.shape)


def _make_dataset(**p):
    # see '_load_dt'
    p = dict(_load_dt, **p)
    # create data class
    D = Data()

    if p['special'] is not None:
        p = _special_case(p)
        D.name = p['special']
    if not os.path.exists(p['data_path']):
        p['data_path'] = '../' + p['data_path']

    # get origin dataset (as list)
    _read_dataeset(D, p)
    # rename label, get n_categories
    _check_label_name(D, p)
    if p['u_indices'] is not None:
        _resort_uy(D, p)
    if p['switch_status'] is not None:
        _set_normal(D, p)
    if p['del_dim'] is not None:
        _del_dim(D, p)

    # scaler fit transform
    if p['scaler'] is not None:
        D.X_Scaler = _scaler_fit_transform('X', D, p)
        D.Y_Scaler = _scaler_fit_transform('Y', D, p)

    # 动态之前的原始数据用 train_X0 等记录
    D._check()
    # stack samples
    if p['stack'] > 1:
        _stack_samples(D, p)

    if p['if_reorganize']:
        # D._print()
        D.test_X, D.test_Y = reorganize_fault_dataset(D.test_X, D.test_Y)

    # to array
    D.length = D._length()
    D.array = D._array()
    D._print()
    return D

def _special_case(p):
    if 'TE' in p['special']:
        sub_path = p['special'][3:]
        default = {
            'data_path': '../Dataset/TE/' + sub_path,
            'seg_part': [1,3],
            'task': 'fd',
            'switch_status': 160,
            'del_dim': np.arange(22,41),
            'del_cate': [3, 9, 15]
            }
    if 'CSTR' in p['special']:
        sub_path = p['special'][5:]
        if sub_path in ['fd_close']: seg_start = 0
        elif sub_path in ['fd']: seg_start = 7
        default = {
            'data_path': '../Dataset/CSTR/' + sub_path,
            'u_indices': [0, 1, 2],
            'u_dim': 3,
            'y_dim': 7,
            'task': 'fd',
            'seg_part': [seg_start,'-1'],
            'switch_status': 200
            }
    if 'TTS' in p['special']:
        sub_path = p['special'][4:]
        default = {
            'data_path': '../Dataset/TTS/' + sub_path,
            'u_indices': [0, 1],
            'u_dim': 2,
            'y_dim': 3,
            'task': 'fd',
            'switch_status': 200
            }
    # 这里加裂数据是先用 _read_hy_data 生成 train 和 test 文件夹，再用该程序读取的
    if 'HY' in p['special']:
        sub_path = p['special'][3:]
        default = {
            'data_path': '../Dataset/HY/' + sub_path,
            'u_indices': [0, 2, 4, 5, 6, 39, 41, 43, 51],
            'u_dim': 9,
            'y_dim': 52-9,
            'task': 'fd',
            'if_reorganize': True
            # 'switch_status': 200
        }
    return dict(p, **default)

def _read_dataeset(D, p):
    # need to put the data into 'train' and 'test' folders, respectively
    data_path = p['data_path']
    Fault = []
    for folder in ['train','test']:
        print("Load {} data from '{}'".format(folder, data_path))
        data_dict = {}
        Data_X, Data_Y = [], []
        
        # read data from folder
        file_paths = []
        if os.path.exists(data_path+'/'+folder):
            # list all subfolder and file under the folder
            file_list = os.listdir(data_path+'/'+folder)
            for file_name in file_list:
                file_path = data_path+'/'+folder+'/'+file_name
                if os.path.isfile(file_path):
                    _data_dict = load_file(file_name, file_path)
                    file_paths.append(file_path)
                    # add '_data_dict' into 'data_dict'
                    data_dict.update(_data_dict)

        for _name, data in data_dict.items():
            if len(data) == 0: continue
            # 1) file_name 以 '_x' 或 '_X' 结尾
            if _name[-2:] in ['_x', '_X']:
                X = data
                y_name = _name[:-1] + chr(ord(_name[-1])+1)
                # 如果在 data_dict 中，数据 X 有同名的且以 '_y' 或 '_Y' 结尾的对应数据集，则将这个文件中读取的数据作为 label
                if y_name in data_dict.keys():
                    label = data_dict[y_name]
                # 否则，以 X 文件名作为 label
                else:
                    if p['task'] == 'fd':
                        label = [0]*X.shape[0]
                    else:
                        label = [_name[:-2].capitalize()]*X.shape[0]
                Y = np.array(label)
            # 2) file_name 以 '_col({int})' 结尾，则以数据的最后 col 列作为标签，前面的列作为 X
            elif '_col(' in _name and _name[-1] == ')':
                start = _name.find('_col(') + 5
                col = int(_name[start:-1])
                X = data[:,:-col]
                Y = data[:,-col:]
            # 3) file_name 不以 ['_x', '_X', '_y', '_Y', '_col({int})'] 中的任何一个结尾时
            elif _name[-2:] not in ['_y', '_Y']:
                X = data
                # 有 seg_part 时，截取部分文件名作为 label
                if p['seg_part'] is not None:
                    if p['seg_part'][1] == '-1': label_name = _name[p['seg_part'][0]:]
                    else: label_name = _name[p['seg_part'][0]:p['seg_part'][1]]
                    label = [label_name.capitalize()]*X.shape[0]
                # 否则将整个 file_name 作为 label
                else:
                    label = [_name.capitalize()]*X.shape[0]
                Y = np.array(label)
            else: continue
                
            print("->  from {}'{}'\033[0m\t-> X{}, Y{}".format(_msg_code('cy',False), _name, X.shape, Y.shape))
            if folder == 'test' and p['fe_last_dims'] != 0:
                Data_X.append(X[:p['fe_last_dims']])
                Fault.append(X[p['fe_last_dims']:])
            else:
                Data_X.append(X)
            Data_Y.append(Y)
            
        if folder == 'train': D.train_X0, D.train_Y0 = Data_X, Data_Y
        else: D.test_X0, D.test_Y0 = Data_X, Data_Y

def load_file(file_name, file_path):
    # return {name: key}
    suffix = file_name.split('.')[-1]            # file format
    _name = file_name[:int(-1*len(suffix))-1]    # file name without format
    if suffix in ['csv']:
        # 'header' = None when there is no tab head
        return {_name: pd.read_csv(file_path, header = None).values}
    elif suffix in ['dat']:
        return {_name: np.loadtxt(file_path)}
    elif suffix in ['xls','xlsx']:
        return {_name: pd.read_excel(file_path).values}
    elif suffix == 'mat':
        data_dict = scio.loadmat(file_path)
        data_dict.pop('__header__')
        data_dict.pop('__version__')
        data_dict.pop('__globals__')
        return data_dict
    elif suffix == 'npz':
        return dict(np.load(file_path))

def _check_label_name(D, p):
    # get unique 'label_name'
    L = np.concatenate(D.train_Y0 + D.test_Y0, axis = 0)
    L = list(np.unique(L))
    if 'Normal' in L:
        del L[L.index('Normal')]
        L.insert(0, 'Normal')
    
    D.label_name0 = L
    # match dataset with 'label_name' id
    _match_dataset_label_id(D)
    
    # rename 'label_name'
    D.label_name = D.label_name0.copy()
    if p['task'] is not None:
        if p['task'] == 'fd':
            length = len(D.label_name0)
            zeros = '%0'+ str(len(str(length))) + 'd'
            D.label_name = ['Normal'] + ['Fault '+ zeros % i for i in range(1, length)]
        else:
            D.label_name = p['task']
        
    # del part of 'label_name' and corresponding datasets
    if p['del_cate'] is not None:
        _del_cate(D, p)
    
    # find 'Y' in 'label_name0' (convert 'str' to 'int')
    for i in range(len(D.train_Y0)): D.train_Y0[i] = _search_AinB_loc(D.label_name0, D.train_Y0[i])
    for i in range(len(D.test_Y0)): D.test_Y0[i] = _search_AinB_loc(D.label_name0, D.test_Y0[i])
    
    # rename label
    print('\nOriginal label names are:\n{}'.format(D.label_name0))
    if p['task'] is not None: print("\nReplaced label names are:\n{}".format(D.label_name))
    D.n_cate = len(D.label_name)
    
    
def _match_dataset_label_id(D, debug = False):
    # match which label the i-th Data_Y belong to
    # D.train_cate[i] = index of Data_Y[i]'s most label in 'label_name'
    D.train_cate, D.test_cate = [], []
    for phase, Y in enumerate([D.train_Y0, D.test_Y0]):
        if phase == 0: D_cate = D.train_cate
        else: D_cate = D.test_cate
        for i in range(len(Y)):
            counter = Counter(Y[i])
            sorted(counter.items(), key = lambda item: item[0])
            label = list(counter.keys())[-1]
            _id = D.label_name0.index(label) if label in D.label_name0 else -1
            D_cate.append(_id)
            
    if debug:
        print('Train categories corresponding to label name is:\n{}'.format(D.train_cate))
        print('Test categories corresponding to label name is:\n{}'.format(D.test_cate))    

def _del_cate(D, p):
    del_cate = sorted(p['del_cate'], reverse = True)
    # del label name
    for cate_id in del_cate: del D.label_name0[cate_id]; del D.label_name[cate_id]
    # del dataset
    for phase, X_cate in enumerate([D.train_cate, D.test_cate]):
        del_loc = _search_AinB_loc(del_cate, X_cate)
        del_cate_id = np.where(del_loc != -1)[0]
        del_cate_id = sorted(del_cate_id, reverse = True)
        for i in del_cate_id:
            if phase == 0: del D.train_X0[i]; del D.train_Y0[i]
            else: del D.test_X0[i]; del D.test_Y0[i]

def _resort_uy(D, p):
    u_indices = p['u_indices']
    D.u_dim = len(u_indices)
    D.y_dim = D.train_X0[0].shape[0] - D.u_dim
    for phase, data_X0 in enumerate([D.train_X0, D.test_X0]):
        for i in range(len(data_X0)):
            X0 = data_X0[i]
            U0 = X0[:, u_indices]
            # 创建一个布尔掩码来选择剩余的索引
            mask = np.ones(X0.shape[-1], dtype = bool)
            mask[u_indices] = False
            # 提取剩余的数据作为 y
            Y0 = X0[:, mask]
            X = np.concatenate([U0, Y0], axis=-1)
            if phase == 0: D.train_X0[i] = X
            else: D.test_X0[i] = X

def _set_normal(D, p):
    switch_status = p['switch_status']
    if type(switch_status) != list:
        switch_status = [switch_status]*len(D.test_Y0)

    for i in range(len(D.test_Y0)):
        if type(switch_status[i]) == tuple: switch = list(switch_status[i])
        else: switch = [switch_status[i]]
        switch.insert(0,0)
        switch.append(D.test_Y0[i].shape[0])
        for j in range(len(switch)):
            if np.mod(j,2) == 1:
                D.test_Y0[i][switch[j-1]:switch[j]] = 0

def _del_dim(D, p):
    del_dim = np.array(p['del_dim'])
    # del dim
    for i in range(len(D.train_X0)): D.train_X0[i] = np.delete(D.train_X0[i], del_dim, axis=1)
    for i in range(len(D.test_X0)): D.test_X0[i] = np.delete(D.test_X0[i], del_dim, axis=1)

if __name__ == '__main__':
    D = _make_dataset(special = 'CSTR/fd')
    # D = _make_dataset(special = 'TE', stack = 40)