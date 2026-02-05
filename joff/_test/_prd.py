# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import numpy as np

def _test_prd(self, Y, L, p):
    m_y = Y.shape[1] if Y.ndim == 2 else 1
    Y, L = Y.reshape(-1, m_y), L.reshape(-1, m_y)

    RMSE, R2 = np.zeros((m_y,)), np.zeros((m_y,))
    for j in range(m_y):
        RMSE[j], R2[j] = _get_rmse(Y[:,j], L[:,j]), _get_R2(Y[:,j], L[:,j])
    ARMSE, AR2 = _get_rmse(Y, L), _get_R2(Y, L)

    self._perf = ARMSE, np.round(AR2, 3), RMSE, np.round(R2, 3)
    self._single_perf = ARMSE

def _get_rmse(y, l):
    return np.sqrt(np.mean((y - l) ** 2))

def _get_R2(y, l):
    SS_res = np.sum((y - l) ** 2)
    SS_tot = np.sum((l - np.mean(l, 0)) ** 2)
    return 1 - SS_res / SS_tot