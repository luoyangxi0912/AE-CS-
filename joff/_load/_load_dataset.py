# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch

import torch.utils.data as torch_Data
from joff._nn._attr import _set_func_pr, _update_module_attr
from joff._func._dvc import _dvc_dt
from joff.customize import _load_dt
from joff._load._make_dataset import _make_dataset
from joff._load.RNN_DataLoader import RNNDataloader

def _load(self, **kwargs):
    p = _set_func_pr(self, _load_dt, **kwargs)
    _update_module_attr(self, p)
    if 'b' not in p.keys(): p['b'] = self.b
    D = _make_dataset(p) if p['dataset'] is None else p['dataset']
    self.D = D
    _make_loader(self, D, p)

def _load_data_from_file(**p):
    return _make_dataset(**p)

def TensorDataLoader(X, Y, batch_size, shuffle, drop_last, **dvc_dt):
    if not isinstance(X, torch.Tensor):
        X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    data_set = torch_Data.dataset.TensorDataset(X, Y)
    data_loader = torch_Data.DataLoader(data_set, batch_size=batch_size,
                                        shuffle=shuffle, drop_last=drop_last, **dvc_dt)
    data_loader.X, data_loader.Y = X.numpy(), Y.numpy()
    data_loader.name = 'TensorDataLoader'

    if not shuffle and not drop_last:
        unshuf_data_loader = data_loader
    else:
        unshuf_data_loader = torch_Data.DataLoader(data_set, batch_size=batch_size,
                                                   shuffle=False, drop_last=False, **dvc_dt)
        unshuf_data_loader.X, unshuf_data_loader.Y = X.numpy(), Y.numpy()
        unshuf_data_loader.name = 'TensorDataLoader'

    return data_loader, unshuf_data_loader

def _make_loader(self, D, p):
    train_X, train_Y, test_X, test_Y = D._array()

    if p['Loader'] == 'Tensor':
        self.train_loader, self.unshuf_train_loader = TensorDataLoader(train_X, train_Y,
                                                                       p['b'], p['if_shuf'], p['if_drop_last'],
                                                                       **_dvc_dt(self.dvc))
        self.test_loader, _ = TensorDataLoader(test_X, test_Y, p['b'],False, False, **_dvc_dt(self.dvc))
        self.test_loader.seg_len = D._length()[1]  # split point for each faulty dataset
    elif p['Loader'] == 'RNNLoader':
        self.train_loader = RNNDataloader(D.train_X, D.train_Y, D.stack, p['b'],
                                          p['if_shuf'], p['if_shuf'], True)
        self.unshuf_train_loader = RNNDataloader(D.train_X, D.train_Y, D.stack, D.stack,
                                                 False, False, True)
        self.test_loader = RNNDataloader(D.test_X, D.test_Y, D.stack, D.stack,
                                         False, False, True)
        # print(self.test_loader.seg_len)

    self.D = D
    self.cur_loader = self.train_loader
    self.label_name = D.label_name
    self.kwargs['label_name'] = self.label_name


if __name__ == '__main__':
    load_datas = [{'special': 'CSTR/fd', 'stack': 10},
                  {'special': 'HY/fd', 'stack': 10}]
    D = _make_dataset(**load_datas[1])
    p = { 'b': 16,
          'if_shuf': True,
          'Loader': 'RNNLoader'
        }
    _make_loader(D, D, p)
    