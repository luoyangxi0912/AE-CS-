# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import sys
import torch
import numpy as np
import pandas as pd
from pandas import DataFrame
from joff._func._np import _2np
from joff._save._save_df import _save_loss_df
from joff.customize import _perf_dt

# 淡黄 218 202 119；深黄 255 191 103；淡橙 255 162 69；橙 255 128 108；青 117 194 179；蓝 73 176 206；粉 249 123 173；紫 219 184 255；
_rgb_code_dt = {'r': '31m', 'lb': '94m',
                'ly': (218, 202, 119),  # light yellow
                'dy': (255, 191, 103),  # dark yellow
                'lo': (255, 162, 69),   # light orange
                'o': (255, 128, 108),   # orange
                'cy': (117, 194, 179),  # cyan
                'b': (73, 176, 206),    # blue
                'pi': (249, 123, 173),  # pink
                'pp':(219, 184, 255)}    # purple
def _msg_code(color, bold = True):
    def _get_rgb_code(rgb_item):
        if type(rgb_item) == str: return rgb_item
        return '38;2;{};{};{}m'.format(rgb_item[0],rgb_item[1],rgb_item[2])

    bold_str = '' if not bold else '1;'
    return '\033[' + bold_str + _get_rgb_code(_rgb_code_dt[color])

# init msg to record training info
def _init_msg(self):
    self._msg, self._msg_sum = {}, {}
    for key in self.view_addi_info:
        self._msg[key] = 0.
        self._msg_sum[key] = 0.

# update msg in training phase
def _update_msg(self, b_id, n, batchs):
    if not hasattr(self, 'cnt_e'): self.cnt_e = 1
    _info = 'Epoch: {}{}\033[0m - {}{}\033[0m/{}{}\033[0m |'.format(_msg_code('pp', False),\
            self.cnt_e, _msg_code('pp', False), b_id+1, _msg_code('pp', False), batchs)

    for key in self.view_addi_info:
        if hasattr(self, key)==False or eval('self.'+ key) is None: continue
        loss = _2np(eval('self.'+ key))
        loss_n = loss * n
        loss_sum = loss_n + self._msg_sum[key]
        self._msg[key], self._msg_sum[key] = loss_n, loss_sum
        _info += ' ' + key + ' = {}{:.4f}\033[0m,'.format(_msg_code('pp', False),loss)
            
    if self.training and ((b_id+1) % 10 == 0 or (b_id+1) == batchs):
        sys.stdout.write('\r'+ _info[:-1] + '\033[0m                ')
        sys.stdout.flush()

# record msg in 'train/test_df'
def _record_msg(self, _msg_phase, N):
    '''
        batch_training: update self.train_df
        batch_testing (train): update self.eval_df
        batch_testing (test): update self.test_df and self.eval_df
    '''
    # when 'batch_testing (train)', _msg_phase = 'none'
    if _msg_phase == 'no': return
    _save_loss_df(self, _msg_phase, N)

# msg_dict
full_name_dict = {'cls': ('fault diagnosis rate', 'false positive rate'),
                  'prd': ('root mean square error', 'R-Square'),
                  'fd': ('false alarm rate', 'missed detection rate')}

# get perf index msg
def _perf_msg(task, loc, item = None, addi = ''):
    _full = full_name_dict[task][loc]
    _abbr = _perf_dt[task][loc]
    if task in ['fd', 'cls']:  
        if addi == 's': return _full + 's (' +_abbr[1:] + 's) are:\n{}{}\033[0m(%)'.format(_msg_code('ly'), item)
        else: return _abbr + " of '{}' is {}{}\033[0m(%)".format(addi, _msg_code('dy'), item)
    return _full + ' (' +_abbr + ') is:\n{}{}\033[0m'.format(_msg_code('dy'), item)

# show eval msg
def _show_simple_msg(self):
    length = len(self.fd_pr_name) if self.task == 'fd' else 1
    
    for i in range(length):
        sub_name = self.fd_pr_name[i]
        _perf = self._perf[i] if self.task == 'fd' else self._perf

        # 'simple_msg' for multiple runs (test 'last' paras)
        if self.run_times > 1:
            print("\nThe test " + _perf_msg(self.task, 0, _perf[0], sub_name ))
            print("The test " + _perf_msg(self.task, 1, _perf[1], sub_name ))
            
        # 'simple_msg' for epoch eval 'self._perf[0] and self._perf[1]', and also save to 'self.eval_df'
        else:
            if i == 0: print()
            _front_str = sub_name + ': ' if self.task == 'fd' else ''
            per_str = r'(%)' if self.task in ['cls', 'fd'] else ''
            print('\t>>> ' + _front_str + _perf_dt[self.task][0] + ' = {}{}, '.format(_perf[0], per_str) + \
                  _perf_dt[self.task][1] + ' = {}{}.'.format(_perf[1], per_str))

def _show_detailed_msg(self, pr_id = None):
    _perf = self._perf[pr_id] if self.task == 'fd' else self._perf

    print('\nThe test '+ _perf_msg(self.task, 0, _perf[-2], 's'))
    print('\nThe test '+ _perf_msg(self.task, 1, _perf[-1], 's'))
    if self.task in ['cls','fd']:
        print('\nThe test ' + _perf_dt[self.task][0] + ' is {}{}\033[0m(%).'.format(_msg_code('dy'), _perf[0]))
        print('The test ' + _perf_dt[self.task][1] + ' is {}{}\033[0m(%).'.format(_msg_code('dy'), _perf[1]))