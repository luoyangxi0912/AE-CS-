# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
from joff._nn._fcnn_module import FCNN

VAE_default = {
            'if_unsup': True,
            'de_struct': None,      # FCNN 在初始化的时候会自动检测，如果为 None 则自动取 struct 的逆
            'de_act': None,
            'priori_v2': 1.,
            'sample_times': 3,
            'view_addi_info': ['recon_loss', 'kl_loss', 'loss'],
            'if_loss_mean': True,
            'if_output_mean': True,
            'alf': (1.,1.)
            }

class VAE(FCNN):
    def __init__(self, **kwargs):
        kwargs = dict(VAE_default, **kwargs)
        FCNN.__init__(self, **kwargs)
        
        self.encoder = self.Seq(struct = self.struct[:-1], act = self.act[:-1])
        self.u = self.Seq(struct = self.struct[-2:], act = ['a'])
        self.logv2 = self.Seq(struct = self.struct[-2:], act = ['a'])

        self.decoder = self.Seq(struct = self.de_struct, act = self.de_act)

        # sampling
        self.mv_normal = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.struct[-1]), torch.eye(self.struct[-1]))

        self.opti()

    def _get_z_recon_u_v2(self, x, rd = None):
        h = self.encoder(x)
        u, logv2 = self.u(h), self.logv2(h)
        v, v2 = torch.exp(logv2 / 2.), torch.exp(logv2)
        self._latent = torch.cat([u, logv2], dim=-1)

        if rd is None: rd = self.mv_normal.sample(torch.Size([self.sample_times, u.size(0)])).to(self.dvc)
        _z_list, _recon_list = [], []
        for k in range(self.sample_times):
            z = u + v * rd[k]
            _z_list.append(z)
            recon = self.decoder(z)
            _recon_list.append(recon)
        return _z_list, _recon_list, u, v2

    def forward(self, x):
        _, _recon_list, u, v2 = self._get_z_recon_u_v2(x)
        self.recon_loss = torch.sum(sum([(recon - x)**2 for recon in _recon_list])/self.sample_times)
        self.kl_loss = torch.sum( (u**2/self.priori_v2 + v2/self.priori_v2 - torch.log(v2/self.priori_v2) -1)/2 )
        if self.if_loss_mean: self.recon_loss, self.kl_loss = self.recon_loss/x.size(0), self.kl_loss/x.size(0)
        self.loss = self.weighted_loss([self.recon_loss, self.kl_loss])

        if self.if_output_mean: return sum(_recon_list)/self.sample_times
        return _recon_list[-1]
    
    def forward_for_vmap(self, x):
        _, _recon_list, _, _ = self._get_z_recon_u_v2(x, self._rd_test)
        if self._vmap_mm  == 'lv': return self._latent

        if self.if_output_mean: recon = sum(_recon_list)/self.sample_times
        else: recon = _recon_list[-1]
        if self._vmap_mm == 're': return recon