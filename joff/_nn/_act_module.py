# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch

class Gaussian(torch.nn.Module):
    def forward(self, x):
        return 1-torch.exp(-torch.pow(x,2))
    
class Affine(torch.nn.Module):
    def forward(self, x):
        return x * 1.0
    
class Square(torch.nn.Module):
    def forward(self, x):
        return x**2
