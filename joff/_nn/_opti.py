# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
from joff._nn._attr import _set_func_pr
from joff._nn._para import _get_para
from joff.customize import _opti_dt

'''
    SGD,  Adam, RMSprop
    Adadelta, Adagrad, Adamax, SparseAdam, ASGD, Rprop, LBFGS
'''
def _opti(self, **kwargs):
    p = _set_func_pr(self, _opti_dt, **kwargs)
    # l2_norm
    if p['l2_norm'] != 0.:
        _init_normal, _init_one, _init_zero = _get_para(self)
        weights, biases = _init_normal, _init_one + _init_zero
        params = r"[{'params': weights, 'weight_decay': p['l2_norm']}," + \
            r"{'params': biases, 'weight_decay':0} ],"
    else:
        params = r"self.parameters(),"
    # lr
    lr = r"lr = p['lr'],"
    # alpha
    alpha = ''
    if p['opt'] == 'RMSprop': alpha = r"alpha=0.9, eps=1e-10,"
    
    params = (params + lr + alpha)[:-1]
    self.opti = eval(r"torch.optim."+p['opt']+'('+params+')')