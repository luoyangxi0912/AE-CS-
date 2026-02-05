# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
import numpy as np
from joff._nn._act_module import *
from joff.customize import _act_dt

# expand act's length to match struct
def _expand_act(act, length):
    if len(act) == 1: act = act * 2
    hid_act = act[:-1].copy()
    if len(hid_act) < length:
        for i in range(length - len(hid_act)):
            act.insert(-1, hid_act[int(np.mod(i, len(hid_act)))])
    return act

# check str in struct
def _check_struct(struct):
    for i, m_l in enumerate(struct):
        if i > 0 and type(m_l) == str:
            struct[i] = int(eval( 'struct[i-1]' + m_l))
    return struct

# convert to standard format for constructing module
def _check_struct_and_act(kwargs):
    struct = kwargs['struct']
    act = kwargs['act'] if len(kwargs['act']) >0 else ['a']
    
    struct = _check_struct(struct)           # 检查是不是有字符运算
    act = _expand_act(act, len(struct)-2)    # 自动扩展 act 至与 struct 所需激活数匹配，取 act[:-1] 列表循环（最后一个作为输出层激活）
    kwargs['struct'], kwargs['act'] = struct, act
    # print(struct, func)
    # for unsup
    if 'if_unsup' in kwargs.keys() and kwargs['if_unsup']:
        de_struct, de_act = kwargs['de_struct'], kwargs['de_act']
        if 'de_struct' not in kwargs.keys() or kwargs['de_struct'] is None:
            de_struct = struct.copy()
            de_struct.reverse()
            kwargs['de_struct'] = de_struct
        if 'de_act' not in kwargs.keys() or kwargs['de_act'] is None:
            de_act = act[:-1].copy()
            de_act.reverse()
            de_act += [act[-1]]
            kwargs['de_act'] = de_act
        kwargs['de_act'] = _expand_act(de_act, len(de_struct)-2)
        # print(self.de_struct, self.de_func)
    return kwargs

# convert to standard activation function class
def _get_act(name):
    if name in _act_dt.keys(): name = _act_dt[name]
    if '(' in name: pass
    elif 'Softmax' in name: name += '(dim = 1)'
    else: name += '()'
    return eval(name)
