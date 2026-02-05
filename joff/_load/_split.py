# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
import numpy as np
from joff._load._stack import _get_dynamic_data

# convert vector to list arccording to seg_len
# if seg_len are the length of list element, set if_accu = True
def _v2l(v, seg_len, if_need_accu = True):
    if if_need_accu:
        seg_len = seg_len.copy()
        for i in range(1, len(seg_len)):
            seg_len[i] += seg_len[i-1]
        seg_len.insert(0, 0)
    l = []
    for i in range(1, len(seg_len)):
        l.append(v[seg_len[i-1]: seg_len[i]])
    return l

def _split_xy(X, Y, splits):
    X_list = _v2l(X, splits, False)
    Y_list = _v2l(Y, splits, False) if Y is not None else None
    return X_list, Y_list

class SegData():
    # X: list; Y: list
    def __init__(self, X, Y = None):
        self.X, self.Y = X, Y
        # array sizes
        self.sizes = [X[i].shape[0] for i in range(len(X))]
        # split points
        self.splits = [0]
        for i in range(len(self.sizes)):
            self.splits.append(self.splits[-1] + self.sizes[i])
        self.n_class = self._n_class()

    # X: list; Y: list
    def concat(self, X = None, Y = None):
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        X = np.concatenate(X, 0)
        Y = np.concatenate(Y, 0) if Y is not None else None
        return X, Y

    # X: array; Y: array
    def split(self, X, Y = None):
        return _split_xy(X, Y, self.splits)

    # Y: list
    def _n_class(self):
        Y = self.Y
        if Y is None: return None
        self.labels = np.zeros(len(Y))
        for i in range(len(Y)):
            Y_i = Y[i]
            if len(Y_i.shape) > 1 and Y_i.shape[1] > 1:
                Y_i = np.array(np.argmax(Y_i, axis=1).reshape(-1, 1), dtype=np.float32)
            self.labels[i] = np.max(Y_i)
        return np.max(self.labels)

    def dynamic(self, stack = 12, index = None):
        p = {'stack': stack}
        if index is not None:
            X, Y = self.X[index], self.Y[index]
            return _get_dynamic_data(X, Y, p)
        X_list, Y_list = [], []
        for i in range(len(self.X)):
            dX, dY = _get_dynamic_data(self.X[i], self.Y[i], p)
            X_list.append(dX)
            Y_list.append(dY)
        return X_list, Y_list

