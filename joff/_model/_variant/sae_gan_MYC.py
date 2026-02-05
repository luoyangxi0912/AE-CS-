# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
import numpy as np
from joff._nn._fcnn_module import FCNN
from joff._model.gan import _check_di
from joff._load._make_faulty_dataset import _make_faulty_dataset
from joff._run._epoch import batch_training, batch_testing
from joff._func._dvc import _2dvc
from joff._func._np import _2np
import torch.autograd as autograd

SAE_GAN_default = {'if_unsup': True,
                   'de_struct': None,
                   'de_act': None,
                   'di_struct': None,
                   'di_act': None,
                   'view_addi_info':['recon_loss', 'g_loss', 'd_real_loss', 'd_fake_loss'],
                   'alf': [1.]*2,
                   'n_critic': 1,
                   'if_augment_data': False
                   }

# ma zi
class SAE_GAN(FCNN):
    def __init__(self, **kwargs):

        kwargs = dict(SAE_GAN_default, **kwargs)
        FCNN.__init__(self, **kwargs)
        
        g_encoder = self.Seq()
        g_decoder = self.Seq(struct = self.de_struct, act = self.de_act[:-1] + ['a'])
        self.generator = torch.nn.Sequential(*[g_encoder, g_decoder])
        
        _check_di(self)
        self.discriminator = self.Seq(struct = self.di_struct[:-1] + [1], act = self.di_act[:-1] + ['s'])

        self.d_rate = 0.1
        g_params = [{"params":self.generator.parameters()}, {"params":self.adap_alf}] if self.if_adap_alf \
            else self.generator.parameters()
        # self.g_opti = torch.optim.Adam(params = g_params, lr = self.lr)
        self.g_opti = torch.optim.RMSprop(params=g_params, lr=self.lr, alpha=0.9, eps=1e-10)

        d_params = [{"params": self.discriminator.parameters()}, {"params": self.adap_alf}] if self.if_adap_alf \
            else self.discriminator.parameters()
        # self.d_opti = torch.optim.Adam(params = d_params, lr = self.lr)
        self.d_opti = torch.optim.RMSprop(params=d_params, lr=self.lr, alpha=0.9, eps=1e-10)

    # 用于测试
    def _batch_testing(self, phase):
        if self.if_augment_data == False or phase == 'test': return batch_testing(self, phase)
        _2dvc(self, self.dvc)
        if self.training: self.eval()

        In, Out, Latent, Label = [], [], [], []
        loader = _make_faulty_dataset(self, self.b)
        with torch.no_grad():
            for b_id, (input, label) in enumerate(loader):
                input, label = _2dvc(input, self.dvc), _2dvc(label, self.dvc)
                self._label = label

                # customize test
                x_f, x_n = input, label
                fake_f = self.forward(x_f)
                fake_n = self.forward(x_n)

                In.append(np.concatenate([_2np(x_n), _2np(x_n)]))
                Out.append(np.concatenate([_2np(fake_f), _2np(fake_n)]))
                if hasattr(self, '_latent'): Latent.append(_2np(self._latent))

        In = np.concatenate(In, 0)
        Out = np.concatenate(Out, 0)
        if hasattr(self, '_latent'): Latent = np.concatenate(Latent, 0)
        return In, Out, Latent, Label

    def forward(self, x):
        self._latent = self.discriminator(x)
        # self._latent = - torch.log(self.discriminator(x))
        return self.generator(x)

    # 用于训练
    def _batch_training(self):
        loader = _make_faulty_dataset(self, self.b)
        batch_training(self, loader)
    
    def _forward(self, x):
        x_f, x_n = x, self._label
        ''' Use 'real' sample to train dicriminator '''
        rd_label = (torch.rand(x.size(0)) * self.d_rate + 1- self.d_rate).to(self.dvc)
        self.zero_grad()
        d_xn_loss = -torch.mean(rd_label * torch.log(self.discriminator(x_n)))
        d_xf_loss = -torch.mean(rd_label * torch.log(1-self.discriminator(x_f)))
        self.d_real_loss = d_xn_loss + d_xf_loss
        self.d_real_loss.backward()
        self.d_opti.step()

        # md_loss = torch.mean( 1 - torch.exp( -(self.discriminator(x_n) - self.discriminator(x_f))**2 ) )
        # _d_loss = self.d_real_loss + md_loss
        # # _d_loss = self.weighted_loss([self.d_real_loss, md_loss])
        # _d_loss.backward()
        # self.d_opti.step()
        
        ''' Use 'fake' sample to train dicriminator '''
        fake_gn = self.generator(x_n)
        fake_gf = self.generator(x_f)
        rd_label = (torch.rand(x.size(0)) * self.d_rate + 1- self.d_rate).to(self.dvc)
        self.zero_grad()
        d_gn_loss = -torch.mean(rd_label * torch.log(1-self.discriminator(fake_gn.detach())))
        d_gf_loss = -torch.mean(rd_label * torch.log(1-self.discriminator(fake_gf.detach())))
        self.d_fake_loss = d_gn_loss + d_gf_loss
        self.d_fake_loss.backward()
        self.d_opti.step()
        
        ''' Use 'fake' sample to train generator '''
        self.g_loss, self.recon_loss = None, None
        if self.cnt_batch % self.n_critic == 0:
            self.zero_grad()
            rd_label = (torch.rand(x.size(0)) * self.d_rate + 1 - self.d_rate).to(self.dvc)

            g_gn_loss = -torch.mean(rd_label * torch.log(self.discriminator(fake_gn)))
            g_gf_loss = -torch.mean(rd_label * torch.log(self.discriminator(fake_gf)))
            self.g_loss = g_gn_loss + g_gf_loss
            _g_loss = self.g_loss

            # r_gn_loss = self.loss_func(fake_gn, x_n)
            # r_gf_loss = self.loss_func(fake_gf, x_n)
            # self.recon_loss = r_gn_loss + r_gf_loss
            # _g_loss = self.weighted_loss([r_gn_loss, r_gf_loss, self.g_loss])

            _g_loss.backward()
            self.g_opti.step()
