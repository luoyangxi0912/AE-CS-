# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:22:53 2022

@author: Yccc7
"""

import numpy as np

from joff._run._runner import Runner

p = {'models': ['DAE','VAE'],
     'structs': [[7,'*20', 20],
                 [7, '*20', 50, 10]],
     'acts': [
                (1, ['a', 's'], ['g', 'a']),
                (1, ['s', 's'], ['g', 'a']),
                (1, ['g', 's'], ['g', 'a']),
                (1, ['q', 's'], ['g', 'a']),
                
                (2, ['q', 'q', 's'], ['s', 's', 'a']),
                (2, ['q', 'a', 's'], ['s', 's', 'a']),
                (2, ['q', 's', 's'], ['s', 's','a']),
                (2, ['s', 's', 's'], ['s', 's','a'])
              ],
     'fd_prs':[['lv&re-T2-ineq']
               ],
     #'drop_rate': 0.3,
     'lr': 1e-4,
     'load_datas': [{'special': 'CSTR_fd_close',
                     'scaler': ['st','oh']
                     # 'del_cate': [4, 5, 6],
                     # 'del_dim': [0, 1, 2]
                     # 'del_dim': [3, 4, 8]
                     },
                    {'special': 'TE_cls', 'stack': 40}],
     '_addi_name':'[unsup_lv]',
     'alf':(1.,100.)
    }
# p['if_epoch_eval'] = [False, True]
# p['mm'] = 'lv'
p['if_vmap'] = True
R = Runner(**p)
# R.run(dataset_id = 1, e = 30, run_times = 1
#           #if_save_plot = False
#           )
          
model = R._get_model(model_id = 2, dataset_id = 1, act_id = 4)
model.run(e = 24, b = 16, run_times = 1)
         #if_save_plot = False
         # )