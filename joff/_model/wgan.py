# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable
from joff._model.gan import GAN
from joff._func._dvc import _Tensor

def _wasserstein_loss(self, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    real_samples, fake_samples = real_samples.to(self.dvc), fake_samples.to(self.dvc)
    # Random weight term for interpolation between real and fake samples
    alpha = _Tensor(self.dvc)(np.random.random((real_samples.size(0), real_samples.size(1))))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = self.discriminator(interpolates)
    ones = Variable(_Tensor(self.dvc)(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    # ∂ outputs/ ∂ inputs * grad_outputs
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class WGAN(GAN):
    def __init__(self, **kwargs):
        default = {'var_msg':['d_loss', 'w_loss', 'g_loss'],
                   'alf': (1,10)
                   }
        kwargs = dict(default, **kwargs)
        GAN.__init__(self, **kwargs)
    
    def _forward(self, x):
        z = self.mv_normal.sample(torch.Size([x.size(0)])).to(self.dvc)
        
        # Use 'real' and 'fake' sample to train dicriminator
        self.d_opti.zero_grad()
        fake = self.generator(z)
        self.w_loss = _wasserstein_loss(self, x.data, fake.data)
        self.d_loss = -torch.mean(self.discriminator(x)) * self.alf[0] +\
                      torch.mean(self.discriminator(fake.detach())) * self.alf[0] +\
                      self.w_loss * self.alf[1]
        self.d_loss.backward()
        self.d_opti.step()
        
        # Use 'fake' sample to train generator
        if self.cnt_batch % self.n_critic == 0:
            self.g_opti.zero_grad()
            self.g_loss = -torch.mean(torch.log(self.discriminator(fake)))
            self.g_loss.backward()
            self.g_opti.step()