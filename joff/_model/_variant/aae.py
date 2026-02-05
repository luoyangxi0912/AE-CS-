# -*- coding: utf-8 -*-       
"""
Created on 2022/11/24 19:49 
@Author: Yccc7

@Software: PyCharm
"""
import torch
from joff._model.vae import VAE, VAE_default
from joff._model.gan import _check_di

AAE_default = {'di_struct': None,
               'di_act': None,
               'view_addi_info':['recon_loss', 'g_loss', 'd_loss_r', 'd_loss_f'],
               'n_critic': 1
               }

class AAE(VAE):
    def __init__(self, **kwargs):
        kwargs_default = dict(VAE_default, **AAE_default)
        kwargs = dict(kwargs_default, **kwargs)
        kwargs['sample_times'] = 1
        VAE.__init__(self, **kwargs)

        _check_di(self)
        self.generator = torch.nn.Sequential(*[self.encoder, self.decoder])
        self.discriminator = self.Seq(struct=self.di_struct[:-1] + [1], act=self.di_act[:-1] + ['s'])

        self.g_opti = torch.optim.Adam(params=self.generator.parameters(), lr=self.lr)

        self.d_opti = torch.optim.SGD(params=self.discriminator.parameters(), lr=self.lr)

    # for testing
    def forward(self, x):
        _, _recon_list, _, _ = self._get_z_recon_u_v2(x)
        return _recon_list[-1]

    # for training
    def _forward(self, x):
        # Use 'real' sample to train dicriminator
        ones_label = torch.rand(x.size(0)).to(self.dvc)
        z = self.mv_normal.sample(torch.Size([x.size(0)])).to(self.dvc)
        self.d_opti.zero_grad()
        p = self.discriminator(z)
        self.d_loss_r = -torch.mean(ones_label * torch.log(p))
        self.d_loss_r.backward()
        self.d_opti.step()

        # Use 'fake' sample to train dicriminator
        _z_list, _recon_list, _, _ = self._get_z_recon_u_v2(x)
        self.d_opti.zero_grad()
        fake = _z_list[-1]
        self.d_loss_f = -torch.mean(torch.log(1 - self.discriminator(fake.detach())))
        self.d_loss_f.backward()
        self.d_opti.step()

        # Use 'fake' sample to train generator
        self.g_loss, self.recon_loss = None, None
        if self.cnt_batch % self.n_critic == 0:
            self.g_opti.zero_grad()
            self.recon_loss = torch.sum((x - _recon_list[-1])**2)/x.size(0)
            self.g_loss = -torch.mean(torch.log(self.discriminator(fake)))
            _g_loss = self.recon_loss + self.g_loss
            _g_loss.backward()
            self.g_opti.step()
        return _recon_list[-1]