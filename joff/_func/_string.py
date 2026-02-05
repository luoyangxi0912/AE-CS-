# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import numpy as np

# convert to numpy
def _get_suffixs(r):
    suffixs = ['th', 'st', 'nd', 'rd'] + ['th'] * 6
    if r > 10 and r < 20: suffix = 'th'
    else: suffix = suffixs[int(np.mod(r, 10))]
    return '/' + str(int(r)) + suffix

def _del_prefix(_str):
    if type(_str) != list and type(_str) != np.ndarray: _str = [_str]
    _str = _str.copy()
    for i in range(len(_str)):
        loc = np.max( _str[i].find('train-'), _str[i].find('test-') )
        _str[i] = _str[i][loc+6:]
        _str[i] = _str[i].replace('_',' ')
        _str[i] = _str[i].capitalize()
    return _str