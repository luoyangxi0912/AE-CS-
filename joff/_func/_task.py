# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

from joff.customize import _fd_dt, _test_dt, _run_dt

# get task dict
def _get_task_dt(task, _dt):
    if task == 'fd': 
        return dict(_dt, **_fd_dt)
    else:
        return _dt

# decode fd_pr
def _get_fd_pr(fd_pr):
    def _traversal(i, f_str):
        for str in PR_LIST[i]:
            if i < len(PR_LIST) - 1:
                # print(i+1, f_str + '-' + str, len(PR_LIST[i]))
                _traversal(i+1, f_str + '-' + str)
            else:
                fd_pr_tral.append(f_str[1:] + '-' + str)

    # fd_pr: ['re&lv-Q&T2-kde&ineq', 'cust_mm-T2-ineq', 'cust_ts-kde']
    fd_pr_dt, fd_pr_name = [], []
    for pr in fd_pr:            # 're&lv-Q&T2-kde&ineq'
        pr_sp = pr.split('-')
        PR_LIST = []
        cnt = 1.
        for sp in pr_sp:        # [['re,lv'], ['Q','T2'], ['kde','ineq']]
            sp2 = sp.split('&') if '&' in sp else [sp]
            cnt *= len(sp2)
            PR_LIST.append(sp2)
        # PR_LIST.reverse()
        fd_pr_tral = []
        _traversal(0, '')
        # print('attr:', fd_pr_tral)
        fd_pr_name += fd_pr_tral
        for pr_tral in fd_pr_tral:
            pr_dt = {}
            pr_split = tuple(pr_tral.split('-'))
            pr_dt['mm'] = None if len(pr_split) == 2 else pr_split[0]
            pr_dt['ts'], pr_dt['thrd_meth'] = pr_split[-2], pr_split[-1]
            fd_pr_dt += [pr_dt]

    # print('\nfd_pr_name:', fd_pr_name)
    return fd_pr_dt, fd_pr_name

if __name__ == '__main__':
    print('test_dict:\n', _get_task_dt('fd', _test_dt), '\n')
    print('run_dict:\n', _get_task_dt('fd', _run_dt))