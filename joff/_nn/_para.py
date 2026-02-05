# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
import torch.nn as nn
import numpy as np

def _get_para(self, para_name = 'weight'):
    _init_normal, _init_one, _init_zero = [], [], []
    for name, para in self.named_parameters():
        # print(name, ',', para.shape)
        if para.dim() > 1: _init_normal.append(para)
        elif para_name in name: _init_one.append(para)
        else: _init_zero.append(para)
    return _init_normal, _init_one, _init_zero

def _init_para(paras = None, init = 'xavier_normal_'):
    '''
        uniform_, normal_, constant_, ones_, zeros_, eye_, dirac_, 
        xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, orthogonal_, sparse_
    '''
    for para in paras:
        # print(para.shape)
        if type(init) == float:
            nn.init.constant_(para, init)
        else:
            eval('nn.init.'+init+'(para)')

def _init_module_paras(self):
    '''
        default:
        W: truncated_normal(stddev=np.sqrt(2 / (size(0) + size(1))))
        b: constant(0.0)
    '''
    _init_normal, _init_one, _init_zero = _get_para(self)
    _init_para(_init_normal)
    _init_para(_init_one, 1.)
    _init_para(_init_one, 0.)

if __name__ == '__main__':
    from joff._nn._fcnn_module import fcnn_example
    module = fcnn_example()
    _init_module_paras(module)
    # print(dir(module))        
