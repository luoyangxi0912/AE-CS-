# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import numpy as np
from joff._run._runner import Runner
# np.random.seed(1992)

p = {'models': ['DAE','VAE','KW_VAE','SAE_GAN'],
     # struct of encoder
     'structs': [[10,'*10','/2'],
                 [10, '*10', '/2', '/2']
                 ],
     # (model_id, act1, act2, act3) → ('struct', 'act', 'de_act', 'di_act'); act: hidden act loop + output act
     'acts': [(1, ['a'], ['a']),
              (1, ['s'], ['s', 'a']),
              (1, ['g', 's'], ['s', 'a']),
              (2, ['q', 's', 's'], ['q', 's', 'a']),
              (2, ['g', 'g', 's'], ['g', 'a', 'a']),
              (2, ['t', 't', 's'], ['t', 's', 'a']),

              (1, ['g', 'g'], ['g', 'a']),
              (1, ['t'], ['t', 'a']),
              (1, ['q', 'g'], ['q', 'a']),
              (2, ['q', 't', 'g'], ['g', 't', 'a']),
              (2, ['s', 't', 's'], ['t', 't', 'a']),
              (2, ['s', 's', 's'], ['s', 's', 'a'])
              ],
     'fd_prs':[['re-T2&Q-kde'],['re-T2&Q-kde'],
               ['cust_ts_recon-kde', 'cust_ts_kl-kde'],
               ['re-T2&Q-kde', r'lv-\phi_{\cal{D}}-kde']
               ],
     # 'drop_rate': 0.3,
     'dvc': 'cuda',
     'lr': 1e-3,
     'load_datas': [{'special': 'CSTR_fd', 'scaler': ['st','oh'] },
                    {'special': 'TE', 'stack': 40}
                    ],
     '_addi_name':'[unsup_res]',
     'if_minus_mean': True,
     'if_use_lstsq': False,
     # 'priori_v2': 0.01,
     # 'alld_error': 0.01,
     # 'alf':(0.01,1.),
     # 'if_adap_alf': True,
     'if_save_alf': True,
     'ts_post_op': [False, False, '-log'],
     'if_augment_data': False
     # 'if_augment_data': True
     }

model_id, act_id = 4, 2

# p['if_epoch_eval'] = [False, True]
# p['if_plot_mm_kde'] = True
# p['if_plot_score_hm'] = True

# p['if_vmap'] = True
# p['if_save_plot'] = False

R = Runner(**p)
# R.run(dataset_id = 1, e = 24, run_times = 5
#           # if_save_plot = False
#           )
model = R._get_model(model_id = model_id, dataset_id = 1, act_id = act_id, b = 16)
model.run(e = 24, run_times = 1)

# 111: aaaa,      loss = 0.0001,      FAR = 0.3,      MDR = 8.8,      M(re) = 0.0031,     J(re) = 0.0167,     H(re) = 0.0000
# 112: sssa,      loss = 0.0282,      FAR = 0.15,     MDR = 11.7,     M(re) = 0.1106,     J(re) = 0.1504,     H(re) = 0.0073
# 113: gssa,      loss = 0.0278,      FAR = 0.1,      MDR = 11.84,    M(re) = 0.1078,     J(re) = 0.1518,     H(re) = 0.0286
# 114: qssqsa,    loss = 0.0170,      FAR = 0.1,      MDR = 11.39,    M(re) = 0.1053,     J(re) = 0.1708,     H(re) = 0.0316
# 115: ggsgaa,    loss = 0.0198,      FAR = 0.1,      MDR = 9.92,     M(re) = 0.0964,     J(re) = 0.1723,     H(re) = 0.0439
# 116: ttstsa,    loss = 0.0116,      FAR = 0.15,     MDR = 12.88,    M(re) = 0.0703,     J(re) = 0.1495,     H(re) = 0.0271

# 117: ggga,      loss = 0.0106,      FAR = 0.3,      MDR = 11.07,    M(re) = 0.0615,     J(re) = 0.1584,     H(re) = 0.0409
# 118: ttta,      loss = 0.0023,      FAR = 0.15,     MDR = 12.91,    M(re) = 0.0216,     J(re) = 0.0542,     H(re) = 0.0255
# 119: qgqa,      loss = 0.0155,      FAR = 0.15,     MDR = 11.39,    M(re) = 0.0765,     J(re) = 0.1757,     H(re) = 0.0601
# 1110: qtggta,   loss = 0.0112,      FAR = 0.35,     MDR = 9.58,     M(re) = 0.0664,     J(re) = 0.1736,     H(re) = 0.0675
# 1111: ststta,   loss = 0.0105,      FAR = 0.15,     MDR = 11.25,    M(re) = 0.0748,     J(re) = 0.1356,     H(re) = 0.0061
# 1112: sssssa,   loss = 0.2226,      FAR = 0.1,      MDR = 11.05,    M(re) = 0.3172,     J(re) = 0.1710,     H(re) = 0.0146