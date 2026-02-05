# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import os
import sys
import time
import torch
import traceback
import numpy as np
from pandas import DataFrame
from IPython.core.ultratb import VerboseTB, ColorTB

from joff._nn._attr import _set_func_pr, _update_module_attr
from joff._nn._para import _init_module_paras
from joff._nn._vmap import _get_jcb_hess
from joff._func._log import Logger
from joff._func._task import _get_task_dt
from joff._func._string import _get_suffixs
from joff._func._msg import _msg_code
from joff._run._epoch import batch_training
from joff._save._save_para import _save_module_para
from joff._load._load_para import _load_module_para, _set_path_for_best
from joff._test._test_model import _eval, _test_best_run
from joff._save._save_excel import _save_excel
from joff._save._save_df import _df_mean_var
from joff.customize import _run_dt, _epoch_eval_dt, _perf_dt, _best_perf_dt

# get pr before run
def _run(self, **kwargs):
    # update kwargs
    p = _set_func_pr(self, _get_task_dt(self.task, _run_dt), **kwargs)
    _update_module_attr(self, p)
    
    # check if multi-run
    if self.run_times > 1: self.save_path += ' ({} times)'.format(self.run_times)

    # check if load
    if self.load_para_type is not None:
        self.run_times = 1
        # create df
        if not hasattr(self, 'train_loss_hd'): init_df_hd(self)
        init_df(self)

        # load parameters
        _load_module_para(self, self.load_para_type, self.load_sub, self.load_file_name, self.load_path_replace)
        # test model performance
        self.test()
        return
    
    # select run with or without log
    if self.if_log: _run_with_log(self, p)
    else: _run_n(self, p)

# log
def _run_with_log(self, p):
    try:
        # log_path = '../Log/'+ self._name + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        log_file_name = self.save_path + '/log - ' + time.strftime("%Y.%m.%d - %Hh %Mm %Ss", time.localtime()) + '.log'

        self.logger = Logger(log_file_name)
        _run_n(self, p)
    except:
        # save Error to file
        tb = ColorTB()
        self.logger.to_file(traceback.format_exc())
        self.logger.reset()
        
        # show Error in console
        for i, _str in enumerate(tb.structured_traceback(*sys.exc_info())):
            # if i == 1: _str = _str[_str.find('Traceback'):]
            print(_str)
    finally:
        self.logger.reset()

# run n time, n >= 1
def _run_n(self, p):
    if 'if_simple_msg' not in p.keys() or not p['if_simple_msg']:
        print(); print('\033[1m{}\033[0m'.format(self))
        print('\nrun paras: {}{}\033[0m'.format(_msg_code('pp',False), p))
        # print('\nmodel paras:', self.kwargs)

    print('\nTrain '+self._name + ' via {}'.format(self.dvc) +':')
    
    # run 'self.run_times' times
    for r in range(1, self.run_times + 1):
        self.run_id = r
        if self.run_times > 1:
            if self.multirun_seed is not None:
                if type(self.multirun_seed) == str: seed = eval(self.multirun_seed)
                else: seed = self.multirun_seed[np.mod(r, len(self.multirun_seed))]
                torch.manual_seed(int(seed))
            self._run_id = _get_suffixs(r)
            self._save_path = self.save_path + self._run_id
            
            time.sleep(1.0)
            # init paramaters
            _init_module_paras(self)
            print("\n{}☆ Running {} for the {}<{}>{} time\033[0m".format(_msg_code('o',False), \
                  self._name, _msg_code('lo'), self._run_id[1:], _msg_code('o',False)))
        else:
            # init paramaters
            _init_module_paras(self)
        
        # train model
        train_model(self, **p)

        # load and test epoch 'best' (if 'eval')
        if self.if_epoch_eval[1]:
            _set_path_for_best(self, 'eval')
            _load_module_para(self, 'best')

        # test model performance
        self.test()

        # get jcb and hess
        if self.if_vmap: _get_jcb_hess(self)

    # load and test run_times 'best' (if 'multi')
    if self.run_times > 1:
        self.mult_mean_var_df = _df_mean_var(self.mult_df, start_column = len(self.train_loss_hd) )
        _save_excel(self.save_path, 'Run Times.xlsx', self.mult_mean_var_df)
        subset = self.mult_mean_var_df.loc[['mean', 'std'], self.test_perf_hd]
        print('{}{}\033[0m'.format(_msg_code('ly',False), subset))

        _set_path_for_best(self, 'mult')
        _load_module_para(self)
        init_df(self)
        self.train_loss_df = self.best_train_loss_df  # load 'best' result
        _test_best_run(self, **p)  # test 'best' and give detailed msg

    # # get jcb and hess
    # if self.if_vmap: _get_jcb_hess(self)

def init_df_hd(self):
    # tab head for record/print 'loss' info
    self.train_loss_hd, self.test_loss_hd = [], []
    # tab head for record/print 'eval' info
    self.train_perf_hd, self.test_perf_hd  = [], []
    
    # if epoch eval
    self.if_epoch_eval = _epoch_eval_dt[self.task] if self.if_epoch_eval is None else self.if_epoch_eval
    
    # create [train/test_loss_hd] according to 'self.view_addi_info'
    for phase in ['train', 'test']:
        if phase == 'test' and not self.if_epoch_eval[1]: break
        for key in self.view_addi_info:                       # 'train_loss (epoch)' always calculate
            if phase == 'train': self.train_loss_hd.append('train-'+ key)
            else: self.test_loss_hd.append('test-'+ key)
            
    # create [train/test_perf_hd] according to '_perf_dt'
    if self.task != 'fd':                                     # 'train/test_perf (last)' always calculate for not 'fd'
        if self.if_epoch_eval[1]: exec('self.'+ _best_perf_dt[self.task][0] + '=' + _best_perf_dt[self.task][1])
        for index in list(_perf_dt[self.task]): self.train_perf_hd.append('train-' + index)
        for index in list(_perf_dt[self.task]): self.test_perf_hd.append('test-' + index)
    else:
        if self.if_epoch_eval[1]: exec('self.'+ _best_perf_dt[self.task][0] + '= [' + _best_perf_dt[self.task][1] +'] * len(self.fd_pr_name)')
        for pr_name in self.fd_pr_name:                       # 'test_perf (last)' always calculate for 'fd'
            self.test_perf_hd.append('{}-AFAR'.format(pr_name))
            self.test_perf_hd.append('{}-AMDR'.format(pr_name))
    
    # create [mult_df] to record 'test_perf (last)' of multiple run and their mean and var
    if self.run_times > 1:
        # create multiple run df
        exec('self.best_mult_perf =' + _best_perf_dt[self.task][1])
        self.mult_df = DataFrame(columns = self.train_loss_hd + self.test_perf_hd)

def recreate_df_from_head(head):
    if len(head) != 0: return DataFrame(columns = head)
    else: return None

def init_df(self):
    # recreate for each run
    self.train_loss_df = recreate_df_from_head(self.train_loss_hd)
    self.test_loss_df = recreate_df_from_head(self.test_loss_hd)
    self.train_perf_df = recreate_df_from_head(self.train_perf_hd)
    self.test_perf_df = recreate_df_from_head(self.test_perf_hd)

def train_model(self, **p):
    start = time.perf_counter()

    if not hasattr(self, 'train_loss_hd'): init_df_hd(self)   # noly need exec once
    init_df(self)

    # train and eval model
    self.cnt_batch = 0
    if p['e'] == 0: return
    for epoch in range(1, p['e'] + 1):
        self.cnt_e = epoch
        self.pro_e = epoch / (p['e'] + 1)
        
        # epoch train
        if hasattr(self, '_batch_training'): self._batch_training()  # custom training
        else: batch_training(self)
        # epoch eval (load 'best' for test when finish p['e'])
        if self.if_epoch_eval[0] and len(self.train_perf_hd) != 0:
            _eval(self, loader='train')
        if self.if_epoch_eval[1]:
            _eval(self, loader='test')
        # record custo
        if self.if_save_alf: self._save_alf(epoch)

    end = time.perf_counter()
    print("\nFinish training~ Spend {} seconds".format(int(end - start)))
    
    print("\nMean training losses: {} = {}{}\033[0m".format(self.train_loss_df.columns.values, _msg_code('cy'),\
                                                            np.round(self.train_loss_df.values.astype(float)[-1], 4)))
    # save 'last' paras
    _save_module_para(self, 'last')
    