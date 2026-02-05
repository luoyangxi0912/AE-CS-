# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:32:21 2022

@author: Fuzz4
"""

import numpy as np
# W1: (2, 3)
W1 = np.array([[2, 1, 0],[-1, 0, 1]])
b1 = np.array([-1,-2])
# W2: (1, 2)
W2 = np.array([[1,-2]])
b2 = np.array([-1.1])


def _act(_z, act):
    _a = np.zeros_like(_z, dtype = np.float32)
    for i in range(_z.size):
        z = _z[i]
        if act == 'r': _a[i] = z if z>0 else 0
        if act == 's': _a[i] = 1 / (1 + np.exp(-z))
        if act == 'q': _a[i] = z**2
        if act == 'g': _a[i] = 1-np.exp(-z**2)
        if act == 't': _a[i] = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
        if act == 'a': _a[i] = z
    return _a

def _symb(z, h, act):
    if act == 'r': return np.sign(h).T
    if act in ['g', 't', 'a']: return np.sign(z)
    return 1

def relu(A):
    A[A<0] = 0
    return A

def attr(x, acts):
    # forward
    z1 = W1@x + b1
    h1 = _act(z1, acts[0])
    z2 = W2@h1 + b2
    h2 = _act(z2, acts[1])
    # attribution
    x, h1, h2 = x.reshape(-1,1), h1.reshape(-1,1), h2.reshape(-1,1)
    A12, A01 = W2.T*h1/(z2-b2), W1.T*x/(z1-b1)
    _A12, _A01 = relu(W2.T*h1*_symb(z2, h2, acts[1])), relu(W1.T*x*_symb(z1, h1, acts[0]))
    _A12_sum, _A01_sum = np.sum(_A12,0), np.sum(_A01,0)
    _A12_sum[_A12_sum == 0] = 1; _A01_sum[_A01_sum == 0] = 1
    _A12, _A01 = _A12/np.sum(_A12,0), _A01/np.sum(_A01,0)

    np.set_printoptions(precision=2)    
    # layer 0 -> 1
    print('Layer 0 -> 1: (3,) -> (2,)')
    print('x = {}, z1 = {}, h1 = {}, act = {}'.format(x.reshape(-1,), z1, h1.reshape(-1,), acts[0]))
    # A01: (3,2), h1: (2,1), C01: (3,2)
    C01, _C01 = A01 * h1.T, _A01 * h1.T
    print('A01 = {}, _A01 = {}'.format(A01, _A01))
    print('C01 = {}, _C01 = {}'.format(C01, _C01))
    
    # layer 1 -> 2
    print('Layer 1 -> 2: (2,) -> (1,)')
    print('h1 = {}, z2 = {}, h2 = {}, act = {}'.format(h1.reshape(-1,), z2, h2.reshape(-1,), acts[1]))
    # A12: (2,1), h2: (1,1), C01: (2,1)
    C12, _C12 = A12 * h2, _A12 * h2
    print('A12 = {}, _A12 = {}'.format(A12, _A12))
    print('C12 = {}, _C12 = {}'.format(C12, _C12))
    
    # layer 0 -> 2
    print('Layer 0 -> 2: (3,) -> (1,)')
    print('x = {}, h2 = {}, acts = {}'.format(x.reshape(-1,), h2.reshape(-1,), acts))
    # A02: (3,1), h2: (1,1), C02: (3,1)
    A02, _A02 = A01 @ A12, _A01 @ _A12
    C02, _C02 = A02 * h2, _A02 * h2
    print('A02 = {}, _A02 = {}'.format(A02, _A02))
    print('C02 = {}, _C02 = {}'.format(C02, _C02))
    print()

acts_list = [ ['r', 'g'],
              ['s', 'q'],
              ['r', 'r']
            ]

x = np.array([1., 1., 1.])
for acts in acts_list:
    print('\nacts:', acts)
    attr(x, acts)