# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""
import torch
import numpy as np
import pandas as pd
from torch.nn import Parameter
from pandas import DataFrame
from joff._func._np import _concat_v
from joff._nn._attr import _set_func_pr
from joff._load._split import _v2l
from joff._save._save_df import _concat_dfs, _fd_perf2v
from joff._save._save_excel import _save_excel, _concat_excel
from joff._plot._line import _plot_loss_perf, _plot_cls_dist, _plot_prd_curve, _loop_plot_fd_curve
from joff.customize import _plot_dt

def _save_result(self, p):
    if p['if_save_excel']: _save_result2excel(self, p)
    if p['if_save_plot']: _save_result2plot(self, p)

def _save_result2excel(self, p):
    # save result to df to excel
    result_df = {'keys': list(self.kwargs.keys()), 'vaules': list(self.kwargs.values())}
    indices = [i for i in range(len(result_df['vaules'])) if not isinstance(result_df['vaules'][i], (torch.Tensor, Parameter))  ]
    result_df['keys'] = [result_df['keys'][id] for id in indices]
    result_df['vaules'] = [result_df['vaules'][id] for id in indices]
    kwargs_df = pd.DataFrame(result_df)
    
    # get epoch-(loss + perf)
    epoch_df = _concat_dfs([self.train_loss_df, self.test_loss_df, self.train_perf_df, self.test_perf_df])
    epoch_df.index = np.arange(1, epoch_df.values.shape[0] + 1)
    
    # take all perf (detailed + average) from 'self._perf'
    if self.task == 'prd': v = np.array(list(self._perf)).reshape((1,-1))
    
    if self.task == 'fd': 
        det_v = np.concatenate( _fd_perf2v(self, loc = 'detail'), axis = 1)
        ave_v = np.array( _fd_perf2v(self, loc = 'ave') ).reshape(1,-1)
        v = np.concatenate([det_v, ave_v])
        
    if self.task == 'cls': v = np.concatenate([_concat_v(list(self._perf)[2:]),\
                                              np.array(list(self._perf)[:2]).reshape(1,-1) ])
    index = None
    if self.task in ['fd','cls']: 
        index = self.label_name + ['Average']
        if self.task == 'fd' and 'Normal' in index: index = index[1:]

    perf_df = pd.DataFrame(v, columns= self.test_perf_hd, index = index)
    
    _save_excel(self._save_path, self.task.upper() + ' Result.xlsx',\
                [kwargs_df, epoch_df, perf_df], ['Kwargs', 'Epoch', 'Perf'])

def _save_result2plot(self, p):
    # plot result in 'self._save_path' (if 'self.if_plot')
    p = _set_func_pr(self, _plot_dt, **p)
    _plot_loss_perf(self._save_path, self.train_loss_df, self.test_perf_df, self.task, **p)
    if self.task == 'cls': _plot_cls_dist(self._save_path, self._cls_count, **p)
    if self.task == 'prd': _plot_prd_curve(self._save_path, np.argmax(self._test_Out, 1), self._test_Label, **p)
    if self.task == 'fd':
        label_name = self.label_name[1:] if 'Normal' in self.label_name else self.label_name
        for i in range(len(self.fd_pr_name)):

            if 'cust_ts' in self.fd_pr_dt[i]['ts']: ts_name = self.fd_pr_dt[i]['ts'].replace('cust_ts_', '')
            else: ts_name = self.fd_pr_dt[i]['ts']

            p['thrd'] = self._thrd[i]               # 阈值
            p['pr_name'] = self.fd_pr_name[i]       # mm-ts-thrd
            if 'cust_' in p['pr_name']: p['pr_name'] = p['pr_name'][8:]
            _loop_plot_fd_curve(self._save_path + '/' + self.fd_pr_name[i].replace('\\', ''),
                                ts_name,            # 统计量的名称，如T2、Q
                                label_name,         # 标签 Y 类别的名称
                                self._TS_off[i],    # 离线统计量
                                _v2l(self._TS_on[i], self.test_loader.seg_len),     # 在线统计量（列表）
                                _v2l(np.argmax(self.test_loader.Y, 1), self.test_loader.seg_len),   # 在线标签（列表）
                                **p)
    
def save_runner_result(model, path, file):
    # save result to excel
    pref = model.mult_mean_var_df.values[-2, len(model.train_loss_hd):].tolist() \
        if model.kwargs[ 'run_times'] > 1 else _fd_perf2v(model)

    # create tab head and empty excel
    df = DataFrame(np.array([model.act_str, model.name] + pref).reshape(1,-1), columns= ['Act', 'Model'] + model.test_perf_hd)
    if model.kwargs['if_vmap']:
        data = np.array([model.Frobenius_M, model.Frobenius_J, model.Frobenius_H])
        df2 = DataFrame(data.reshape(1,-1), columns=['E[F-norm(mm)]', 'E[F-norm(J(mm))]', 'E[F-norm(H(mm))]'])
        df = pd.concat([df, df2], axis=1, ignore_index = False)
    df.index = [model.runner_cnt]
    _concat_excel(path, file, df)


if __name__ == '__main__':
    r1 = np.random.rand(4,)
    r2 = np.random.rand(4,)
    print(type(r1), r1.shape)
    print(np.concatenate([r1,r2], axis = 1))
    