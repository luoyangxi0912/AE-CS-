# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from joff._nn._fcnn_module import FCNN

# sui zi
class ND_DAE(FCNN):
    def __init__(self, **kwargs):
        default = {
            'if_unsup': True,
            'de_struct': None,
            'de_act': None,
            'if_pdf': True
            }
        kwargs = dict(default, **kwargs)
        FCNN.__init__(self, **kwargs)
        
        self.encoder = self.Seq()
        self.decoder = self.Seq(struct = self.de_struct, act = self.de_act)
        self.if_init = False

        self.opti()

    def m_fit(self):

        self.train_data = torch.sqrt(torch.sum(torch.square(torch.from_numpy(self.dataset.train_X[0]).float()), 1)).to(
            self.dvc)
        self.max_2norm = torch.max(self.train_data)

        # f = Fitter(self.train_data.cpu().numpy(), distributions=['gamma', 'norm'], timeout=30)
        # f.fit()
        # print(f.df_errors)
        # print( f.get_best())
        # print( f.fitted_param['norm'])
        x = np.sort(self.train_data.cpu().numpy())
        self.kde = gaussian_kde(x)
        # 'np.unique' returns an array with no duplicate elements from smallest to largest
        # x = np.unique(np.sort(self.train_data.cpu().numpy()))
        # evaluate pdfs
        kde_pdf = self.kde.evaluate(x)
        self.max_pdf = np.max(kde_pdf)  # 0.3838
        self.min_z2 = np.min(x)
        self.max_z2 = np.max(x)
        self.max_pdf_x = x[np.argmax(kde_pdf)]
        a = self.min_z2 + 0.2 * (self.max_z2 - self.min_z2)

        # zoom = 0.03
        # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        # ax .plot(x, kde_pdf, c='r')
        # ax .set_title('PDF')
        # ax .set_xscale('symlog')
        # ymin, ymax = np.min(kde_pdf), np.max(kde_pdf)
        # ax .set_ylim(ymin - (ymax - ymin) * zoom, ymax + (ymax - ymin) * zoom)
        # plt.show()

    def forward(self, x):
        if not self.if_init:
            self.m_fit()
            self.if_init = True
        if self.if_pdf:
            try:
                # x_2 = torch.sqrt(torch.sum(torch.square(x),1))
                # # x_2 = torch.where(x_2>(self.min_z2+0.2*(self.max_z2-self.min_z2)),x_2,(self.min_z2+0.20*(self.max_z2-self.min_z2)))
                # x_2 = torch.where(x_2>self.max_pdf_x,x_2,self.max_pdf_x)
                # kde_pdf = self.kde.evaluate(x_2.cpu().numpy())
                # weights = torch.tensor(0.5*self.max_pdf / (kde_pdf + 0.001), dtype=torch.float32).to(self.dvc)#0.4
                # weights = torch.where(weights<50,weights,50)
                x_2 = torch.sqrt(torch.sum(torch.square(x), 1))
                x_2 = torch.where(x_2 > self.max_pdf_x, x_2, self.max_pdf_x)
                # x_2 = torch.where(x_2>(self.min_z2+0.9*(self.max_z2-self.min_z2)),x_2,self.max_pdf_x)
                kde_pdf = self.kde.evaluate(x_2.cpu().numpy())
                weights = torch.tensor(0.5 * self.max_pdf / (kde_pdf + 0.001), dtype=torch.float32).to(self.dvc)  # 0.4
                weights = torch.where(weights < 50, weights, 50)
                weights = len(weights) * weights / torch.sum(weights)
            except:
                weights = torch.ones(x.shape[0]).to(self.dvc)
            # weights = torch.where(weights>0.5,weights,0.5).to(self.dvc) 1/pdf+0.02-2
        else:
            x_2 = torch.sqrt(torch.sum(torch.square(x), 1))
            weights = x_2/ self.max_2norm

        self._latent = self.encoder(x)
        recon = self.decoder(self._latent)
        self.loss = torch.zeros(x.shape[0]).to(self.dvc)
        for index in range(x.shape[0]):
            self.loss[index] = self.loss_func(recon[index], x[index])
        self.loss = torch.mean(weights * self.loss)
        return recon