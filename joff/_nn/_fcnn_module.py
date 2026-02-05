# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""
import os
import sys

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from pandas import DataFrame

from joff.customize import FCNN_dt
from joff._nn._attr import _init_module_attr, _set_func_pr
from joff._nn._opti import _opti

from joff._load._load_dataset import _load
from joff._run._run_model import _run
from joff._test._test_model import _test

from joff._save._save_df import _save_epoch_data_df
from joff._plot._line import _plot
from joff.customize import _plot_dt
from joff._model.seq import Sequential

class FCNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = _init_module_attr(self, FCNN_dt, kwargs)
        if self.if_adap_alf and hasattr(self, 'alf'):
            self.adap_alf = nn.Parameter(torch.ones(len(list(self.alf)))).to(self.dvc)
            self._requires_grad(self.adap_alf)

    def _requires_grad(self, x):
        x._grad_fn = None           # 设置为叶节点
        x.requires_grad_(True)      # 设置需要梯度

    def get_drop_rate(self, input_dim, **p):
        if input_dim >= p['drop_thrd']:
            if p['auto_drop']: return np.round(input_dim / 100.0 / p['drop_thrd'], 4)
            elif p['drop_rate'] != 0: return p['drop_rate']
        return 0.

    def Seq(self, **kwargs):
        return Sequential(self, **kwargs)

    def weighted_loss(self, losses, alf = None):
        if alf is None: alf = self.alf
        if self.if_adap_alf: alf = torch.softmax(alf, dim = -1)
        return sum([losses[i] * alf[i] for i in range(len(losses)) ])

    def _save_alf(self, epoch):
        if self.if_adap_alf == False or not hasattr(self, 'adap_alf'): return
        _alf = self.adap_alf.data.cpu()
        _alf = torch.softmax(_alf, dim=-1).numpy().reshape(1, -1)
        _alf_df = _save_epoch_data_df(self, epoch, _alf, 'alf')
        if epoch == self.e:
            dt = _plot_dt.copy()
            dt['labels'] = ['alf '+ str(i) for i in range(_alf_df.values.shape[1])]
            _plot(_alf_df.values, path=self._save_path, file='Epoch-Alf', **dt)
    
    def opti(self, **kwargs):
        # see '_opti_dt'
        _opti(self, **kwargs)
    
    def load(self, **kwargs):
        # see '_load_dt'
        _load(self, **kwargs)
    
    def run(self, **kwargs):
        # see '_run_model_dict' and '_{task}_dict'
        _run(self, **kwargs)
        
    def test(self, **kwargs):
        # see '_test_dt' and '_{task}_dict'
        _test(self, **kwargs)
    
def fcnn_example():
    module = FCNN(struct = [50, '*2', 333, '/2', '/3', '*4'], 
                  act = ['g','s','a'],
                  drop_rate = 0.1,
                  task = 'prd')
    module.fcnn = module.Seq()
    print(module.kwargs)
    print(module)
    return module


if __name__ == '__main__':
    module = fcnn_example()
    print(type(module))
    print(next(module.parameters()).device == torch.device('cpu'))
    # print(dir(module))