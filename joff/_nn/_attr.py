# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import os
import torch
import numpy as np
from joff._nn._seq import _check_struct_and_act
from joff._func._task import _get_fd_pr

# set attr for class (+ new key into 'self.kwargs', replaced by 'kwargs')
def _init_module_attr(self, default, kwargs):
    # update 'default' using 'kwargs'
    kwargs = dict(default, **kwargs)
    
    # 检查 struct、act、de_struct、de_act 是否符合要求
    if kwargs['struct'][0] == -1:
        x_dim = kwargs['D'].train_X[0].shape[-1] if type(kwargs['D'].train_X) == list else kwargs['D'].train_X.shape[-1]
        kwargs['struct'][0] = x_dim
    kwargs = _check_struct_and_act(kwargs)
    
    # convert str to torch obj (dvc, loss)
    if type(kwargs['dvc']) == str: kwargs['dvc'] = torch.device(kwargs['dvc'])
    if type(kwargs['loss_func']) == str: kwargs['loss_func'] = \
        eval('torch.nn.'+ kwargs['loss_func']+'Loss()')
    # set '_name' and 'save_path'
    kwargs['name'] = self.__class__.__name__
    kwargs['_name'] = kwargs['name'] + kwargs['_addi_name']
    if kwargs['save_path'] is None:
        folder_name = os.path.basename( os.path.dirname( os.getcwd() ) )
        kwargs['save_path'] = '../Result/' + kwargs['_name']
        if folder_name != 'joff': kwargs['save_path'] = '../' + kwargs['save_path']
    # '_save_path' = '../Result/_name' + '_run_id'
    kwargs['_save_path'] = kwargs['save_path']

    # fd_pr
    if kwargs['task'] == 'fd' and 'fd_pr' in kwargs.keys():
        kwargs['fd_pr_dt'], kwargs['fd_pr_name'] = _get_fd_pr(kwargs['fd_pr'])

    # veiw loss
    if 'view_addi_info' in kwargs.keys():
        for loss_key in kwargs['view_addi_info']:
            exec('self.' + loss_key + ' = 0.0')

    # set attribution
    for key in kwargs.keys():
        setattr(self, key, kwargs[key])
    
    return kwargs

# [func default <- kwargs or self.kwargs] (no new key into 'default', but replaced by 'kwargs' or 'self.kwargs')
def _set_func_pr(self, default, **kwargs):
    # 先用 self.kwargs 来更新覆盖 default
    default = _update_dict(default, self.kwargs)
    # 再用 kwargs 来更新覆盖 default
    default = _update_dict(default, dict(**kwargs))
    return default

# [self.kwargs <- p] (no new key into 'self.kwargs', but replaced by 'p')
def _update_module_attr(self, p):
    for key in p.keys():
        if not hasattr(self, key) or p[key] != self.kwargs[key]:
            self.kwargs[key] = p[key]
            setattr(self, key, p[key])
    return self.kwargs
            
# [dict1 <- dict2] (no new key into 'dict1', but replaced by 'dict2')
def _update_dict(dict1, dict2):
    dict1 = dict1.copy()
    for key in dict1.keys():
        if key in dict2.keys():
            dict1[key] = dict2[key]
    return dict1


if __name__ == '__main__':
    class A():
        def _A(_a = 1): return
    
    dict1 = {'a':1, 'c':3}
    dict2 = {'a':0, 'b':2}
    
    a = A()
    _init_module_attr(a, dict2, dict1)
    print(dir(a))
    
    print( dict(dict1, **dict2) )
    print( dict(dict2, **dict1) )