# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:46:30 2022

@author: Fuzz4
"""

import torch
from joff._model.vae import VAE
from joff._model.dae import DAE
from scipy.linalg import toeplitz

from joff._nn._fcnn_module import FCNN

DeNet_default = {'inner_model': 'VAE',
                 'toeplitz_type': '0-1',   # '0-1' or 'decay'
                 'toeplitz_rate': 0.618
                 }

def _toeplitz(n, toeplitz_type, toeplitz_rate):
    row = [1]*n if toeplitz_type == '0-1' else [1/(2**i) for i in range(n)]
    if toeplitz_type == '0-1': row[int(toeplitz_rate*n):] = 0
    col = [1] + row.copy().reverse()[:-1]
    return torch.from_numpy( toeplitz(col, row) )


class DeNet(FCNN):
    def __init__(self, **kwargs):
        kwargs = dict(DeNet_default, **kwargs)
        FCNN.__init__(self, **kwargs)

        self._toeplitz = _toeplitz(self.struct[0], self.toeplitz_type, self.toeplitz_rate).to(self.dvc)
        self.inner = eval(self.inner_model+'(**kwargs)')
        self.opti()

    def _pre_pro(self, x):
        b, m = x.size(0), x.size(1)
        # b × m × m
        x_epd = x.view(b, m, 1) * self._toeplitz
        # b * m × m
        return x_epd.view(-1, m)

    def _post_pro(self, X):
        if self.toeplitz_type == '0-1': X*= self._toeplitz
        return torch.sum(X, -1)/ torch.sum(self._toeplitz, -1).view(1,-1)

    def forward(self, x):
        x_inner = self._pre_pro(x)
        recon_inner = self.inner.forward(x_inner)
        recon = self._post_pro(recon_inner.view(x.size(0), x.size(1), x.size(1)))
        self.loss = self.inner.loss
        return recon
