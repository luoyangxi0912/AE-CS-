# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

from joff._nn._fcnn_module import FCNN

DAE_default = {'if_unsup': True,
               'de_struct': None,      # FCNN 在初始化的时候会自动检测，如果为 None 则自动取 struct 的逆
               'de_act': None
               }

class DAE(FCNN):
    def __init__(self, **kwargs):
        kwargs = dict(DAE_default, **kwargs)
        FCNN.__init__(self, **kwargs)
        
        self.encoder = self.Seq()
        self.decoder = self.Seq(struct = self.de_struct, act = self.de_act)
        
        self.opti()
    
    def forward(self, x):
        self._latent = self.encoder(x)
        recon = self.decoder(self._latent)
        self.loss = self.loss_func(recon, x)
        return recon

    def forward_for_vmap(self, x):
        self._latent = self.encoder(x)
        recon = self.decoder(self._latent)
        if self._vmap_mm == 'lv': return self._latent
        if self._vmap_mm == 're': return recon