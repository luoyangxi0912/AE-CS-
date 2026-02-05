# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import torch

# ------------------------ _nn ------------------------
# _fcnn_module.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FCNN_dt = {
    'dvc': device,                      # set dvc to train model in cpu/cuda 
    'struct': [],                       # network structure
    'act': [],                          # activation functions [loop: hidden + the last: output]
    'b': 16,                            # batch size
    'drop_thrd': 100,                   # if number of neurons >= drop_thrd then use dropout
    'auto_drop': False,                 # 满足 drop_thrd 时，按照神经元数目进行自动 drop（= 神经元数目除以1w）
    'drop_rate': 0,                     # dropout rate
    'use_bn': False,                    # 是否使用 BatchNorm 层
    'loss_func': 'MSE',                 # loss function
    'task': 'fd',                       # use for 'cls', 'prd', 'impu', 'fd', 'gnr', ... task
    'if_adap_alf': False,               # if use learnable alf
    'if_save_alf': False                # if record alf
    }
# _opti.py
_opti_dt = {
    'lr': 1e-4,                         # learning rate
    'opt': 'RMSprop',                   # optimazer
    'l2_norm': 0.                            # l2 regularization 的学习率（可以取1e-5）
    }
# attr.py
_save_dt = {
    '_addi_name': '',                   # 'name' + '_addi_name' = '_name'
    'save_path': None,                  # path to save results
    'save_mode': 'para'                 # save 'para' or 'model'
    }
# _fcnn_module.py
for _dict in [_opti_dt, _save_dt]:
    FCNN_dt = dict(FCNN_dt, **_dict)

# _seq.py
_torch_act_dt = {
    'r': 'ReLU',     'l': 'LeakyReLU',  'm': 'Mish',
    's': 'Sigmoid',  't': 'Tanh',       'x': 'Softmax',    'p': 'Softplus',
    'il': 'SiLU',    'el': 'SELU',      'gl': 'GELU'
    }
for key in _torch_act_dt.keys(): _torch_act_dt[key] = 'torch.nn.' + _torch_act_dt[key]
# _act_module.py
_cust_act_dt = {
    'a': 'Affine',   'g': 'Gaussian',   'q': 'Square'
    }
# _seq.py
_act_dt = dict(_torch_act_dt, **_cust_act_dt)

# ------------------------ _load ------------------------
# _make_dataset.py
_make_dataset_dt = {
    'data_path': '',                    # path to load dataset
    'seg_part': None,                   # intercept part of the file name as the label name
    'label_name': None,                 # the label name for each category
    'u_indices': None,                  # 根据u的编号重新排序数据维度
    'n_delay': 0,                       # 表示y相对于u的延时（做动态数据集时会考虑删除前n_delay个y和l，以及后n_delay个u以对齐）
    'switch_status': None,              # points to switch normal/faulty status
    'del_dim': None,                    # delete part of dim
    'del_cate': None,                   # not consider some categories
    'special': None,                    # special case: 'TE', 'CSTR', ...
    'scaler': ['st','oh'],              # preprocessing: 'mm', 'st', 'oh', ...
    'stack': 1,                         # stack samples to obtain dynamic dataset
    'stack_label': 'fd',                # select a single label w.r.t. a stacked sample
    'fe_last_dims': 0,                  # the last dims are fault signals
    'if_reorganize': False              # 是否重新构建 test_list，让 test_list 中的array数目 = 类别数目
    }
# _load_dataset.py
_load_dt = {
    'dataset': None,                    # a dataset with 'Data' type
    'Loader': 'Tensor',                 # Loader类，默认采用自带的Tensor
    'if_shuf': True,                    # if shuffle for training dataset
    'if_drop_last': False               # if drop last
    }
# _load_dataset.py
_load_dt = dict(_load_dt, **_make_dataset_dt)

# ------------------------ _run ------------------------
# _run_model.py
_run_dt = {
    'run_times': 1,                     # how many times to run the model
    'multirun_seed': 'r*2',             # random seed for each run
    '_run_id': '',                      # '_save_path' = '../Result/_name' + '_run_id'
    'e': 12,                            # number of iterative training
    
    'load_para_type': None,             # load which type of model parameters: 'best'/'last'
    'load_sub': None,                   # load 'ts' sub paras in 'fd'
    'load_file_name': None,             # load file name
    'load_path_replace': None,          # 将path中的部分字符串进行替换
    # 'if_load_train': True,            # if continue to train when 'load_type' is not None

    'view_addi_info': ['loss'],         # additional loss info to view in console/ save in file
    'if_log': True,                     # if run with log
    'if_epoch_eval': None,              # [if_eval_train, if_eval_test] after each epoch training (default equal to '_epoch_eval_dt')
    'if_vmap': False,                   # if calculate vmap of Jcb and Hess

    'if_save_excel': True,              # if save result to excel
    'if_save_plot': True                # if plot result to pdf
    }
# _run_model.py
_epoch_eval_dt = {                      # if eval [train, test] dataset after each epoch training;
                                        # 'train_loss (epoch)' & 'train/test_perf (last)' always calculate, but 'test_loss (epoch)' & 'train/test_perf (epoch)' according to '_epoch_eval_dt'
    'cls': [True, True], 
    'prd': [True, True], 
    'fd': [False, False]
    }
# _run_model.py
_perf_dt = {                            # performance evaluation indices in different task
    'cls': ('AFDR', 'AFPR'),            # average fault diagnosis rate (average accuracy), average false positive rate
    'prd': ('ARMSE', 'AR2'),              # root mean square error, R-Square
    'fd': ('AFAR', 'AMDR')              # average false alarm rate, average missed detection rate
    }

# _runner.py
Runner_dt = {
    'models': ['DAE', 'VAE'],           # list of models
    'structs': [[], []],                # list of structs
    'acts': [(1, ['a'], ['a']),         # list of (struct_id, act_list, act_2_list)
             (2, ['s'], ['a'])],
    'fd_prs': None,                     # for fd tasks
    'load_datas': [{}, {}],             # list of paras in '_make_dataset_dt' to load dataset
    'if_add_run_info': True             # if add '_addi_name' when multiple run
    }
    
# ------------------------ _test ------------------------
# _test_model.py
_best_perf_dt = {                       # best performance in all the epoches ('index name', 'initial', 'comparison')
    'cls': ('max_AFDR', "0.", '>'),           # the max 'AFDR' in 'cls' task
    'prd': ('min_RMSE', "float('inf')", '<'), # the min 'RMSE' in 'prd'/'impu' task
    'fd': ('min_AMDR', "100.", '<')           # the min 'MDR', s.t. 'FAR' < 'expt_FAR' in 'fd' task
    }

# _test_model.py
_test_dt = {
    'loader': 'test',                   # select a loader to eval model
    'if_simple_msg': False,             # 'last' (detailed msg), 'eval' or 'multi' (simple msg)
    'if_save_eval_df': False            # if update 'eval_df' (if True -> simple eval mode)
    }
# _thrd.py
_ineq_dt = {
    'alld_error': 0.005,                # allowed estimation error ('ineq' only)
    'cl': 1 - 0.005,                    # confidence level ('ineq' only)
    'if_intp': False                    # if calculate mm's interpolation to obtain thrd ('ineq' only)
    }
# _fd.py
_fd_dt = {
    'fd_pr': ['re-T2-kde', 'cust_ts-kde'], # para list
                                        # 'mm': 're',                         -> fault detection indices: 're': recon error, 'lv': latent variables
                                        # 'ts': ['T2', 'Q'],                  -> test stat: 'T2', 'SPE', 'GLR', ...
                                        # 'thrd_meth': ['ineq', 'kde'],       -> threshold learning methord: 'pdf', 'ineq', 'kde', ...
    'fd_pr_dt': None,
    'fd_pr_name': None,

    'expt_FAR':  0.005,                 # expected false alarm rate
    'ts_post_op': None,                 # if do post op for ts
    'if_use_lstsq': True,               # if use lstsq to calculate inv of cov ('T2' only)

    'if_plot_mm_kde': False,            # if plot mm kde
    'if_plot_score_hm': False,          # if plot score heatmap
    'if_plot_cov_inv': False,            # 是否画出协方差矩阵的逆的热图

    'if_save_ts': False                 # 保存测试统计量到txt，第一个值代表阈值
    }
# _fd.py
_fd_dt = dict(_fd_dt, **_ineq_dt)

# ------------------------ _save ------------------------
# _plot
_plot_pr_dt = {
    'linewidth': 4,
    'marker': None,
    'markersize': None,
    'color': None,
    'label': None
    }

_matplotlib_dt = {
    'legend': 'y',                      # legend
    'xlabel': None,
    'ylabel': None,
    'figsize': (27.7, 9),
    'font': (54, 49, 54),               # (label, ticks, legend)
    'yscale': 'linear',
    'if_markers': False,
    'if_grid': False,
    'print_color': '亮蓝色'
    }

_plot_dt = {
    'loss_perf': 'sepa',                # separate or combine plot epoch-loss and epoch-perf
    'plot_whole_fd': False,             # plot whole fd result
    'language': 'en',                   # language used in plot 'zh' or 'en'
    'thrd': None                        # threshold
    }

for _dict in [_plot_pr_dt, _matplotlib_dt]:
    _plot_dt = dict(_plot_dt, **_dict)