# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import numpy as np
import pandas as pd
import warnings
from joff._func._np import _concat_v
from joff._save._save_excel import _save_excel
from joff.customize import _best_perf_dt

# 按列连接多个 df
def _concat_dfs(dfs):
    _df_list = []
    for df in dfs:
        if df is not None and df.values.size != 0: _df_list.append(df)
    if len(_df_list) == 1: return _df_list[0]
    else: return pd.concat(_df_list, axis=1, ignore_index=False)

# 按行计算均值方差并将结果保存在最后一行
def _df_mean_var(df, start_column = 0):
    head = df.columns[start_column:]
    data = np.array(df.values[:,start_column:], dtype=np.float64)
    df.index = np.arange(1, df.values.shape[0] + 1)
    mean = np.mean(data, axis = 0).reshape((1,-1))
    std = np.std(data, axis = 0).reshape((1,-1))
    df2 = pd.DataFrame(np.round(np.concatenate([mean, std], axis = 0), 4),\
                       columns = head, index = ['mean', 'std'])
    return pd.concat([df, df2])

# fd 性能指标 转变为长 list
def _fd_perf2v(self, loc = 'ave'):
    v = []
    for i in range(len(self._perf)):
        if loc == 'ave': v += list(self._perf[i])[:2]
        # concat v to a long vector
        else: v += [ _concat_v( list(self._perf[i])[2:] ) ]
    return v

def _save_epoch_data_df(self, epoch, data, name):
    s_name, _name = 'self._'+ name + '_df', '_' + name + '_df'
    exec(_name + '= pd.DataFrame(data, index=[self.cnt_e])' )
    if epoch == 1: exec(s_name + '= ' + _name )
    else: exec(s_name + '= pd.concat([ '+ s_name + ', '+ _name + '], ignore_index=False)')
    if epoch == self.e:
        _data = eval(s_name)
        _save_excel(self._save_path, name + '.xlsx', _data)
    return eval(s_name)

# 将当前代记录的 loss 信息（_msg_df）加入 'self.train_loss_df' 或 'self.test_loss_df' 中
def _save_loss_df(self, phase, N):
    # 'self.train_loss_df' / 'self.test_loss_df' <- 'self._msg_sum/N'
    _msg_mean = {}
    for key in self._msg_sum.keys(): _msg_mean[phase + '-' + key] = self._msg_sum[key] / N
    _msg_df = pd.DataFrame(_msg_mean, index=[self.cnt_e])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        exec('self.' + phase + '_loss_df = pd.concat([ self.' + phase + '_loss_df, _msg_df], ignore_index=False)')
    
# update 'self.train_perf_df' / 'self.test_perf_df' when testing 'train' / 'test' dataset
def _save_epoch_perf_df(self, phase):
    # 'self.train_perf_df' / 'self.test_perf_df' <- 'self._perf'
    v = _fd_perf2v(self) if self.task == 'fd' else list(self._perf)
    head = self.train_perf_hd if phase == 'train' else self.test_perf_hd
    _msg_df = pd.DataFrame(np.array(v).reshape((1,-1)), columns = head, index=[self.cnt_e])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        exec('self.' + phase + '_perf_df = pd.concat([ self.' + phase + '_perf_df, _msg_df], ignore_index=False)')

# update 'self.mult_df'
def _save_mult_df(self):
    # head = 'self.train_loss_df' + 'self.test_perf_df' <- 'self.train_loss_df.iloc[-1]' + 'self._perf'
    
    L = self.train_loss_df.iloc[-1].to_frame().stack().unstack(0)
    L.index = [self.run_id]
    
    v = _fd_perf2v(self) if self.task == 'fd' else list(self._perf)
    R = pd.DataFrame(np.array(v).reshape((1,-1)), columns = self.test_perf_hd, index = [self.run_id])
    
    C = pd.concat([L, R], axis=1, ignore_index = False )
    self.mult_df = pd.concat([self.mult_df, C], ignore_index = False)
    
    best_v = np.array(self._single_perf)
    best_v = np.min(best_v) if self.task == 'fd' else best_v
    if_better = (best_v < self.best_mult_perf) if _best_perf_dt[self.task][2] == '<' else (best_v > self.best_mult_perf)

    if if_better:
        self.best_mult_perf = best_v
        self.best_train_loss_df = self.train_loss_df.copy(deep=True)
        
        
if __name__ == '__main__':
    # DataFrame and Series
    print('>>> create df from dict:')
    _d = {'x':[0,1], 'y': [2,3], 1:[3,4], 3:['4',0], '6': [4,'t']}
    _d[6] = ['d']; _d[1] = -1
    df = pd.DataFrame(_d, index=[0,1])
    print(df)
    print(len(df.index))
    
    print('\n>>> create a empty df:')
    data0 = pd.DataFrame(columns=['A','B','C','D','E'])
    print(data0)
    print(len(data0.index))
    data0 = pd.DataFrame(index=['A','B','C','D','E'])
    print(len(data0.columns))
    
    print('\n>>> create df from ndarray:')
    df = pd.DataFrame(np.arange(30).reshape((6,5)),
                      columns=['A','B','C','D','E'])
    print(df, end = '\n\n')
    print(df.ndim, df.shape, df.size)
    print(df.index); print(df.columns, end = '\n\n'); print(df.values, end = '\n\n')
    print(_df_mean_var(df, 2))
    
    print('\n>>> concat two df:')
    df2 = pd.DataFrame(np.arange(5).reshape((1,5)), columns = ['A','B','X','D','E'])
    print(pd.concat([df, df2], ignore_index=False), end = '\n\n')
    print(pd.concat([df, df2], ignore_index=True))
    
    print('\n>>> take a row from df:')
    d0 = df.iloc[-1]
    print(type(d0))
    
    print('\n>>> concat df with df(series):')
    print(pd.concat([df, pd.DataFrame(d0.values.reshape((1,5)), columns=d0.index) ], ignore_index=True))
    
    print('\n>>> convert series to df:')
    d0 = d0.to_frame().stack().unstack(0)
    print(type(d0))
    print(pd.concat([df, d0]), end = '\n\n')
    d0.columns = ['1', '2', '4', '3', '5']
    print(pd.concat([df, d0], axis=1))
    
    print('\n>>> take a column from df:')
    d1 = df['A']
    print(d1)