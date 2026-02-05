# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
from joff._nn._fcnn_module import FCNN
from joff._model.gan import _check_di
from joff._load._make_faulty_dataset import _make_faulty_dataset
from joff._run._epoch import batch_training

FS_GAN_default = {'if_unsup': True,
                   'de_struct': None,
                   'de_act': None,
                   'di_struct': None,
                   'di_act': None,
                   'view_addi_info':['recon_loss', 'g_loss', 'd_loss_n', 'd_loss_f'],
                   'alf': (10, 1),
                   'n_critic': 1
                   }

# ma zi
class FS_GAN(FCNN):
    def __init__(self, **kwargs):

        kwargs = dict(FS_GAN_default, **kwargs)
        FCNN.__init__(self, **kwargs)
        
        g_encoder = self.Seq()
        g_decoder = self.Seq(struct = self.de_struct, act = self.de_act[:-1] + ['a'])
        self.generator = torch.nn.Sequential(*[g_encoder, g_decoder])
        
        _check_di(self)
        self.discriminator = self.Seq(struct = self.di_struct[:-1] + [1], act = self.di_act[:-1] + ['s'])
        
        self.g_opti = torch.optim.Adam(params = self.generator.parameters(), lr = self.lr)

        self.d_opti = torch.optim.SGD(params = self.discriminator.parameters(), lr = self.lr)
    
    def forward(self, x):
        self.recon_loss, self.g_loss, self.d_loss_n, self.d_loss_f = None, None, None, None
        return self.generator(x)
    
    def _batch_training(self):
        loader = _make_faulty_dataset(self)
        batch_training(self, loader)
    
    def _forward(self, x):
        x_n, x_f = self._label, x
        # Use 'normal' sample to train dicriminator
        ones_label = (torch.rand(x.size(0)) * 0.3 + 0.7 ).to(self.dvc)
        self.d_opti.zero_grad()
        p = self.discriminator(x_n)
        self.d_loss_n = -torch.mean(ones_label * torch.log(p))
        self.d_loss_n.backward()
        self.d_opti.step()
        
        # Use 'fake' sample to train dicriminator
        fake_n = self.generator(x_n)
        fake_f = self.generator(x_f)
        self.d_opti.zero_grad()
        self.d_loss_f = -torch.mean(torch.log(1-self.discriminator(fake_n.detach()))) \
                        -torch.mean(torch.log(1-self.discriminator(fake_f.detach())))
        self.d_loss_f.backward()
        self.d_opti.step()
        
        # Use 'fake' sample to train generator
        self.g_loss, self.recon_loss = None, None
        if self.cnt_batch % self.n_critic == 0:
            self.g_opti.zero_grad()
            self.recon_loss = self.loss_func(fake_n, x_n)
            self.g_loss = -torch.mean(torch.log(self.discriminator(fake_n))) \
                          -torch.mean(torch.log(self.discriminator(fake_f)))
            loss = self.recon_loss * self.alf[0] + self.g_loss * self.alf[1]
            loss.backward()
            self.g_opti.step()