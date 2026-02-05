# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:02:13 2022

@author: Fuzz4
"""

import numpy as np

'''
    Element-wise multiplication law
    1) available for (n, m) * (n, 1) and (n, m) * (1, m) 
    2) (n, m) * (d,) -> (n, m) * (1, d)
    3) (d1, 1) * (1, d2) = (1, d2) * (d1, 1) -> (d1, d2) <special>
    4) all satisfy commutative law if it is available
'''
# (3,) * (2, 3) -> (1, 3) * (2, 3) -> (2, 3)
l = np.ones(3,)
rd = np.random.rand(2,3)
print((rd*l).shape, (l*rd).shape)
l2 = np.ones((1,3))
print((rd*l2).shape, (l2*rd).shape)
print(rd*l == l2*rd)
# (2,) * (2, 3) -> (1, 2) * (2, 3) -> error; (2, 1) * (2, 3) -> (2, 3)
r = np.ones((2,1))
print((rd*r).shape, (r*rd).shape)
# (3, 1) * (3,) -> (3, 1) * (1, 3) -> (1, 3) * (3, 1) -> (3, 3) !!! <special>
r1 = np.ones((3,1))
r2 = np.ones((3,))
print((r1*r2).shape, (r2*r1).shape)
print((r1.T*r1).shape, (r1*r1.T).shape)
r1 = np.ones((3,1))
r3 = np.ones((2,1))
print((r1*r3.T).shape, (r3*r1.T).shape)
print((r1.T*r3).shape, (r3.T*r1).shape)
# (3,) * (3,) -> (3,); (3, 1) * (3, 1) -> (3, 1)
print((r2*r2).shape, (r1*r1).shape)