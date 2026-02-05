# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
from joff._nn._fcnn_module import FCNN
from joff._nn._seq import _check_struct, _expand_act

def _check_di(self):
    # check discriminator struct
    if self.di_struct is None: self.di_struct = self.struct
    else: self.di_struct = _check_struct(self.di_struct)
    
    # check discriminator act
    if self.di_act is None: self.di_act = self.act
    else: self.di_act = _expand_act(self.di_act, len(self.di_struct)-2)

GAN_default = {'di_struct': None,
               'di_act': None,
               'view_addi_info':['g_loss', 'd_loss_r', 'd_loss_f'],
               'n_critic': 1
               }

class GAN(FCNN):
    def __init__(self, **kwargs):

        kwargs = dict(GAN_default, **kwargs)
        FCNN.__init__(self, **kwargs)
        
        self.generator = self.Seq()
        
        _check_di(self)
        self.discriminator = self.Seq(struct = self.di_struct[:-1] + [1], act = self.di_act[:-1] + ['s'])
        
        self.g_opti = torch.optim.Adam(params = self.generator.parameters(), lr = self.lr)

        self.d_opti = torch.optim.SGD(params = self.discriminator.parameters(), lr = self.lr)
        
        # sampling
        self.mv_normal = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.struct[0]), torch.eye(self.struct[0]))

    # for testing
    def forward(self, x):
        z = self.mv_normal.sample(torch.Size([x.size(0)])).to(self.dvc)
        self.g_loss, self.d_loss_r, self.d_loss_f = None, None, None
        return self.generator(z)

    # for training
    def _forward(self, x):
        # Use 'real' sample to train dicriminator
        ones_label = (torch.rand(x.size(0)) * 0.3 + 0.7 ).to(self.dvc)
        self.d_opti.zero_grad()
        p = self.discriminator(x)
        self.d_loss_r = -torch.mean(ones_label * torch.log(p))
        self.d_loss_r.backward()
        self.d_opti.step()
        
        # Use 'fake' sample to train dicriminator
        z = self.mv_normal.sample(torch.Size([x.size(0)])).to(self.dvc)
        self.d_opti.zero_grad()
        fake = self.generator(z)
        self.d_loss_f = -torch.mean(torch.log(1-self.discriminator(fake.detach())))
        self.d_loss_f.backward()
        self.d_opti.step()
        
        # Use 'fake' sample to train generator
        self.g_loss = None
        if self.cnt_batch % self.n_critic == 0:
            self.g_opti.zero_grad()
            self.g_loss = -torch.mean(torch.log(self.discriminator(fake)))
            self.g_loss.backward()
            self.g_opti.step()