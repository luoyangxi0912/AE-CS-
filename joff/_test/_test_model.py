# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import numpy as np
from joff._nn._attr import _set_func_pr, _update_module_attr
from joff._func._task import _get_task_dt
from joff._func._msg import _show_simple_msg
from joff._run._epoch import batch_testing
from joff._test._cls import _test_cls
from joff._test._prd import _test_prd
from joff._test._fd import fault_evaluation
from joff._save._save_para import _save_module_para
from joff._save._save_result import _save_result
from joff._save._save_df import _save_epoch_perf_df, _save_mult_df
from joff._load._load_para import _load_module_para, _set_path_for_best
from joff.customize import _test_dt, _best_perf_dt, _run_dt

# test 'last' for 'single' or 'mult' run
def _test(self, **kwargs):
    p = _set_func_pr(self, _get_task_dt(self.task, _test_dt), **kwargs)
    _update_module_attr(self, p)
    if self.run_times > 1: p['if_simple_msg'] = True
    _test_model(self, p)

# test 'epoch' for 'eval'
def _eval(self, **p):
    p = _set_func_pr(self, _get_task_dt(self.task, _test_dt), **p)
    p['if_save_eval_df'] = True
    p['if_simple_msg'] = True
    _test_model(self, p)

# test 'best' for 'mult' run
def _test_best_run(self, **p):
    p = _set_func_pr(self, _get_task_dt(self.task, _test_dt), **p)
    p['run_times'] = 1
    p['if_simple_msg'] = False
    # test the best run
    _test_model(self, p)

def _test_model(self, p):
    p_run = _set_func_pr(self, _run_dt.copy(), **p)
    p = dict(p_run, **p)
    # print(p)

    # get 'self._single_perf' and 'self._perf'
    _batch_testing = self._batch_testing if hasattr(self, '_batch_testing') else batch_testing
    if self.task == 'fd':
        train_In, train_Out, train_Latent, _ = _batch_testing(self, 'train')
        test_In, test_Out, test_Latent, test_Label = _batch_testing(self, 'test')

        self._TS_off, self._TS_on, self._thrd, self._perf, self._single_perf= [], [], [], [], []
        for i in range(len(self.fd_pr_dt)):
            # fd_pr_dt[i] =
            p['pr_id'], p['pr_dt'], p['pr_name'] = i, self.fd_pr_dt[i], self.fd_pr_name[i]
            # print(p['pr_id'], p['pr_dt'], p['pr_name'])
            # 离线：获取训练集的测试统计量以及阈值
            fault_evaluation(self, train_In, train_Out, train_Latent, [], 'offline', p)
            # 在线：获取测试集的测试统计量以及检测结果
            fault_evaluation(self, test_In, test_Out, test_Latent, test_Label, 'online', p)   # get 'self._single_perf' and 'self._perf'
    else:
        _, Out, _, Label = _batch_testing(self, p['loader'])
        self._test_Out, self._test_Label = Out, Label
        if self.task == 'cls': _test_cls(self, Out, Label, p)  # get 'self._single_perf'
        if self.task == 'prd': _test_prd(self, Out, Label, p)  # get 'self._single_perf'
    
    # epoch eval (save 'best')
    if p['if_save_eval_df']:
        if p['loader'] == 'test': _best_epoch_perf(self)
        _save_epoch_perf_df(self, phase = p['loader'])
        _show_simple_msg(self)
    # multiple run's last run_times (update '_perf_array')
    if p['run_times'] > 1 and self.cnt_e == self.e:
        _save_mult_df(self)
        _show_simple_msg(self)
    # save plot and excel
    if not p['if_simple_msg']:
        _save_result(self, p)

# p['loader'] == 'test'
def _best_epoch_perf(self):
    perf_name = 'self.' + _best_perf_dt[self.task][0]
    best_perf = eval(perf_name)

    # compare 'self._single_perf' with 'best_perf'
    if self.task == 'fd':
        # best_perf = list[ fd_pr_1: min_AMDR, fd_pr_2: min_AMDR, ... ]
        for i, _perf in enumerate(best_perf):
            if self._single_perf[i] < _perf:
                best_perf[i] = self._single_perf[i]
                _save_module_para(self, 'best', sub = i)
    else:
        if_better = exec('self._single_perf' + _best_perf_dt[self.task][2] + 'best_perf')
        if if_better:
            best_perf = self._single_perf
            _save_module_para(self, 'best')
    exec(perf_name + '= best_perf')
    

if __name__ == '__main__':
    r = np.random.randint(0,3,20)
    print(r)
    print(r.size)
    print(r[r == 0])
    print((r[r == 0]).size)
    print((r == 0).astype(int))
    print((r > 1).astype(int))
    r = np.random.rand(4,5)
    # take rows of np.mod(row,2) == 0, columns of np.mod(column,3) == 0
    print(r)
    print(np.argmax(r))
    print(r[1::2,::3])