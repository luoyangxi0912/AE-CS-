# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
import torch.utils.data as t_Data
import numpy as np
from joff._nn._attr import _set_func_pr
from joff._func._dvc import _dvc_dt

_gene_fault_dt ={
    'gene_addi_f': [0.1, 8],
    'gene_mult_f': [0.5, 8],
    'p_addi_fault': 1
    }

# 完全在随机变量位置生成
def _make_faulty_dataset(self, b_s, **kwargs):
    p = _set_func_pr(self, _gene_fault_dt, **kwargs)
    X = self.train_loader.X
    F = np.copy(X)
    n, m = F.shape[0], F.shape[1]

    fault_type = np.random.rand(n)  # addi or multi
    addi_loc = np.where(fault_type <= p['p_addi_fault'])
    mult_loc = np.where(fault_type > p['p_addi_fault'])
    F_addi, F_mult = F[addi_loc].copy(), F[mult_loc].copy()

    for i, _F in enumerate([F_addi, F_mult]):
        # add fault to var
        n_i = _F.shape[0]
        dim_p = np.random.rand(n_i, m)
        indexs = np.where(dim_p <= 0.3)

        rd = np.random.rand(n_i, m)
        # fault size
        if i == 0:
            addi_f = np.sign(rd) * (np.abs(rd) * (p['gene_addi_f'][1] - p['gene_addi_f'][0]) + p['gene_addi_f'][0])
            F_addi[indexs] = _F[indexs] + addi_f[indexs]
        else:
            mult_f = np.sign(rd) * (np.abs(rd) * (p['gene_mult_f'][1] - p['gene_mult_f'][0]) + p['gene_mult_f'][0])
            F_mult[indexs] = _F[indexs] * (mult_f[indexs] + 1)

    F[addi_loc], F[mult_loc] = F_addi, F_mult
    dataset = t_Data.dataset.TensorDataset(torch.from_numpy(F).float(),
                                           torch.from_numpy(X).float())
    loader = t_Data.DataLoader(dataset, batch_size=self.b,
                               shuffle=True, drop_last=False, **_dvc_dt(self.dvc))
    loader.X, loader.Y = F, X
    return loader

# 批次同向，先生成方向后生成故障幅值
def _make_faulty_dataset2(self, b_s, **kwargs):
    p = _set_func_pr(self, _gene_fault_dt, **kwargs)
    # print(b_s)
    X = self.train_loader.X
    F = []
    n, m = X.shape[0], X.shape[1]

    for i in range(int(np.ceil(n/b_s))):
        X_b = X[int(i*b_s): int(min((i+1)*b_s, n))]
        _b = X_b.shape[0]
        if np.random.rand() < p['p_addi_fault']:
            direct = (np.random.rand(m) < 0.1).astype(int)
            if np.sum(direct) == 0: direct[int(np.random.randint(0,m))] = 1
            rd = np.random.rand(_b, m)
            addi_F = (rd * (p['gene_addi_f'][1] - p['gene_addi_f'][0]) + p['gene_addi_f'][0]) * direct.reshape(1,-1).astype('float')
            F.append(X_b + addi_F)
        else:
            direct = (np.random.rand(m) < 0.35).astype(int)
            if np.sum(direct) < 3:
                loc = np.random.choice(np.arange(m),3, replace=False).astype(int)
                direct = np.zeros(m)
                direct[loc] = 1
            rd = np.random.rand(_b, m)
            F_mult = (rd * (p['gene_addi_f'][1] - p['gene_addi_f'][0]) + p['gene_addi_f'][0]) * direct.reshape(1, -1).astype('float')
            F.append(X_b * (F_mult + 1))
        # print(direct)
    F = np.concatenate(F, 0)

    dataset = t_Data.dataset.TensorDataset(torch.from_numpy(F).float(), 
                                           torch.from_numpy(X).float())
    loader = t_Data.DataLoader(dataset, batch_size = self.b,
                               shuffle = True, drop_last = False, **_dvc_dt(self.dvc))
    loader.X, loader.Y = F, X
    return loader