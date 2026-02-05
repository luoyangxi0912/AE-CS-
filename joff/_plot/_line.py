# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from joff._nn._attr import _update_dict
from joff._func._string import _del_prefix
from joff.customize import _matplotlib_dt, _plot_pr_dt
from joff._plot._basic_plot_func import _gradient_lines, _generate_colors
from joff._save._print_style import _clickable_file_link, _cstr
from joff._plot._basic_plot_func import _generate_fd_colors

plt.rcParams['font.family'] = 'DejaVu Sans'

_task_dt = {'fd': (r'AMDR (%)', r'Average missed detection rate (%)'),
              'cls': (r'AFDR (%)', r'Average fault diagnosis rate (%)'),
              'prd': (r'RMSE', r'Root mean square error')}

def _plot_loss_perf(path, train_loss_df, test_perf_df, task, **p):
    p['print_color'] = '1赭石'
    # take loss data
    loss, loss_key = train_loss_df.values, _del_prefix(train_loss_df.columns.values)
    # print(loss_key)
    if len(loss_key) == 1: loss_key = loss_key[0]
    if test_perf_df is not None and test_perf_df.values.size != 0:
        perf, perf_key = test_perf_df.values, test_perf_df.columns.values
        # take signle perf data
        if task == 'fd': perf, perf_key = perf[:,1::2], perf_key[1::2]
        else: perf, perf_key = perf[:,0], _del_prefix(perf_key[0])
        if p['loss_perf'] == 'sepa':
            perf_basic = {'labels': perf_key, 'xlabel': r'Epoch',
                          'ylabel': _task_dt[task][0], 'figsize': (27.7, 12)}
            p = dict(p, **perf_basic)
            _plot(perf, path = path, file = 'Epoch-Perf',**p)
        else:
            comb_basic = {'labels': loss_key, 'labelt': perf_key,
                          'xlabel': r'Epoch', 'figsize': (27.7, 12),
                          'ylabel': r'Loss', 'ylabel2': _task_dt[task][0]}
            p = dict(p, **comb_basic)
            _plot(loss, perf, path = path, file = 'Epoch-Loss_Perf', **p)
            
    loss_basic = {'labels': loss_key, 'xlabel': r'Epoch',
                  'ylabel': r'Loss', 'yscale': 'log', 'figsize': (27.7, 12)}
    p = dict(p, **loss_basic)
    p['if_markers'], p['if_grid'] = True, True
    _plot(loss, path = path, file = 'Epoch-Loss', **p)

def _plot_cls_dist(path, cls_count, **p):
    pass

def _plot_prd_curve(path, Y, L, **p):
    pass

def _loop_plot_fd_curve(path, _ts_name, _label_name, _TS_off, _TS_on, _Lable_on, **p):
    # 画训练正常+测试正常
    p['print_color'], p['seg_color'] = '1淡蓝', ['steelblue','b']
    test_normal =np.concatenate( [_TS_on[i][_Lable_on[i] == 0] for i in range(len(_label_name)) ], axis= 0)
    normal_data = np.concatenate( [_TS_off, test_normal], axis=0)
    label_name = np.array( [0] * _TS_off.shape[0] + [1] * test_normal.shape[0] )
    _plot_fd_curve(path, _ts_name, 'Normal', ['Offline Normal', 'Online Normal'],
                   normal_data, label_name, **p)

    # 遍历测试集的子数据集
    for i in range(len(_label_name)):
        # P: pred, L: label
        TS, Label = _TS_on[i], _Lable_on[i]
        p['print_color'], p['seg_color'] = '1浅红', None
        # 画测试每类故障（正常+某类故障）
        _plot_fd_curve(path, _ts_name, _label_name[i], ['Normal', f'Fault {i+1:02d}'],
                       TS, Label, **p)

    if p['plot_whole_fd']:
        p['print_color'], p['seg_color']  = '2翠绿', None
        # 画整个测试集结果（正常+多类故障）
        _plot_fd_curve(path, _ts_name, 'Whole_fd', None,
                       np.concatenate(_TS_on), np.concatenate(_Lable_on), **p)

def _get_st_label(_ts):
    if _ts == 'T2': _ts = r'T^2'
    elif _ts == 'Q': _ts = r'Q'
    elif 'cust_' in _ts:
        if _ts[8:] in ['recon', 'kl']: _ts = 'L_{'+_ts[8:]+'}'
        else: _ts = _ts[8:].capitalize()
    return '$' + _ts + '$'

def _plot_fd_curve(path, _ts, _save_name, _label_name, TS, L, **p):
    segs = []
    start_p = 0
    list_L = np.unique(L).tolist()
    n_class = len(set(L))
    if 'seg_color' not in p.keys() or p['seg_color'] is None: p['seg_color'] = _generate_fd_colors(n_class)
    if _label_name is None: _label_name = ['Normal'] + [f'Fault {i:02d}' for i in range(1, n_class)]

    # loop samples in a fault type
    for j in range(1, L.shape[0]):
        if L[j] == L[start_p]: continue
        # 定义 seg 该采用的颜色
        color = p['seg_color'][list_L.index(L[start_p]) % len(p['seg_color'])]

        # 按（起点，终点，颜色）的格式添加 seg
        segs.append( (start_p, j, color) )
        start_p = j
    # 定义 seg 该采用的颜色
    color = p['seg_color'][list_L.index(L[start_p]) % len(p['seg_color'])]
    # 按（起点，终点，颜色）的格式添加 seg
    segs.append((start_p, L.shape[0], color))

    if p['language'] == 'zh':
        _basic0 = {'legend': ['正常样本', '故障样本'], 'xlabel': r'采样样本个数'}
    else:
        _basic0 = {'legend': _label_name, 'xlabel': r'Samples'}

    # 'font':（xlabel/ylabel，xticks/yticks，legend）
    legend_scale = max(np.sqrt(len(_label_name))* 0.6, 1.25)
    _basic = {'segs': segs, 'thrd': p['thrd'], 'ylabel': _get_st_label(_ts), 'yscale': 'log',
              'figsize': (27.7, 9), 'font': (int(54/legend_scale), int(49/legend_scale), int(54/legend_scale))}
    _basic = dict(_basic0, **_basic)
    p = dict(p, **_basic)
    _plot(TS, path=path, file=_save_name, **p)

# def _plot(data1, data2 = None, path = '', file = '', **p):
#     p = dict(_matplotlib_dt, **p)
#     fig = plt.figure(figsize= p['figsize'])
#
#     if p['language'] == 'zh':
#         # plt.rcParams['font.sans-serif'] = ['Times New Roman']
#         plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#         plt.rcParams['axes.unicode_minus'] = False
#
#     # ax
#     ax = fig.add_subplot(111)
#
#     # font
#     ax.set_xlabel(p['xlabel'], fontsize= p['font'][0])
#     ax.set_ylabel(p['ylabel'], fontsize= p['font'][0])
#     plt.xticks(fontsize= p['font'][1])
#     plt.yticks(fontsize= p['font'][1])
#     plt.yscale(p['yscale'])
#
#     # plot
#     markers = ['o', '^', 's', '*', 'D']
#     if p['if_markers']:
#         p['marker'] = markers[0]
#         p['markersize'] = p['figsize'][1]*0.8 if p['markersize'] is None else p['markersize']
#     if p['if_grid']: plt.grid(True, linestyle='--')
#
#     # fd plot
#     if p['thrd'] is not None:
#         _fd_plot(plt, ax, data1, **p)
#     # single y plot
#     elif len(data1.shape) ==1 or data1.shape[1] == 1:
#         ax.plot(np.arange(1, data1.shape[0] + 1), data1, **_update_dt_value(_plot_pr_dt, p))
#     # multi y plot
#     else:
#         colors = _generate_colors(data1.shape[1])
#         for j in range(data1.shape[1]):
#             if 'labels' in p.keys(): p['label'] = p['labels'][j]
#             p['color'], p['marker'] = colors[j], markers[int(np.mod(j,len(markers)))]
#             ax.plot(np.arange(1, data1.shape[0] + 1), data1[:,j], **_update_dt_value(_plot_pr_dt, p))
#
#     lgd = ax.legend(fontsize = p['font'][2], frameon = False, framealpha = 0.5)
#
#     # save
#     if not os.path.exists(path): os.makedirs(path)
#     file_name = path + '/' + file
#     plt.savefig(file_name + '.pdf', bbox_inches='tight')
#     # plt.savefig(file_name + '.png', bbox_inches='tight')
#     plt.close(fig)
#     print("Plot \033[4m{}\033[0m in '{}'".format(file + '.pdf', path))

def _plot(data1, data2=None, path='', file='', **p):
    p = dict(_matplotlib_dt, **p)
    fig = plt.figure(figsize=p['figsize'])

    if p['language'] == 'zh':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    # ax
    ax = fig.add_subplot(111)

    # font
    ax.set_xlabel(p['xlabel'], fontsize=p['font'][0])
    ax.set_ylabel(p['ylabel'], fontsize=p['font'][0])
    plt.xticks(fontsize=p['font'][1])
    plt.yticks(fontsize=p['font'][1])
    plt.yscale(p['yscale'])

    # plot
    markers = ['o', '^', 's', '*', 'D']
    if p['if_markers']:
        p['marker'] = markers[0]
        p['markersize'] = p['figsize'][1] * 0.8 if p['markersize'] is None else p['markersize']
    if p['if_grid']:
        plt.grid(True, linestyle='--')

    # fd plot
    if p['thrd'] is not None:
        _fd_plot(plt, ax, data1, **p)
    # single y plot
    elif len(data1.shape) == 1 or data1.shape[1] == 1:
        # 单数据绘图 - 确保有标签
        plot_params = _update_dict(_plot_pr_dt, p)
        if 'labels' in p.keys() and p['labels']:
            plot_params['label'] = p['labels'][0] if isinstance(p['labels'], (list, tuple)) else p['labels']
        ax.plot(np.arange(1, data1.shape[0] + 1), data1, **plot_params)
    # multi y plot
    else:
        colors = _generate_colors(data1.shape[1])
        for j in range(data1.shape[1]):
            plot_params = _update_dict(_plot_pr_dt, p)
            if 'labels' in p.keys() and p['labels'] is not None and j < len(p['labels']):
                plot_params['label'] = p['labels'][j]
            else:
                # 如果没有标签，创建默认标签
                plot_params['label'] = f'Series {j + 1}'
            plot_params['color'] = colors[j]
            plot_params['marker'] = markers[int(np.mod(j, len(markers)))]
            ax.plot(np.arange(1, data1.shape[0] + 1), data1[:, j], **plot_params)

    # 安全添加图例 - 只有在有标签时才显示图例
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        lgd = ax.legend(fontsize=p['font'][2], frameon=False, framealpha=0.5)
    else:
        print("警告: 没有找到带标签的图形元素，跳过图例显示")

    # save
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + '/' + file
    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.close(fig)

    clickable_text = _clickable_file_link(file_name + '.pdf')
    pr_name = f' {_cstr(f'[{p['pr_name']}]','淡蓝色','/')}' if 'pr_name' in p.keys() else ''
    print(f"Plot {_cstr( file + '.pdf',p['print_color'],'/')}{pr_name} in '{clickable_text}'")

def _fd_plot(plt, ax, TS, **p):
    mk_colors = []
    legend = p['legend'] if type(p['legend']) == list else [p['legend']]
    # 按标签分段来画
    segs = p['segs']
    legend_id = 0
    for i, seg in enumerate(segs):
        # seg[0] 和 seg[1] 代表分段 seg 的起始点和结束点
        y = TS[seg[0]: seg[1]]
        if i > 0:
            p['label'] = None
            # 切换类别处的两点加一个渐变
            _gradient_lines([(seg[0] - 1, TS[seg[0] - 1]), (seg[0], TS[seg[0]])], [p['color'] , seg[2]], ax,
                            **_update_dict(_plot_pr_dt, p))

        # seg[2] 中记录了 seg 该采用的 color
        p['color'], p['label'] = seg[2], None
        # set legend
        if p['color'] not in mk_colors:
            mk_colors.append(p['color'])
            p['label'] = legend[legend_id % len(legend)]
            legend_id += 1
        
        # get x， seg[0] 和 seg[1] 代表分段 seg 的起始点和结束点
        x = np.arange(seg[0], seg[1])
        ax.plot(x, y, **_update_dict(_plot_pr_dt, p))

    plt.xlim(0, TS.shape[0])
    
    # threshold
    x = np.arange(0, TS.shape[0])
    y = np.array([p['thrd']]*x.shape[0])
    p['color'] = 'gray'
    p['label'] = '阈值' if p['language'] == 'zh' else 'Threshold'
    ax.plot(x, y, '--', **_update_dict(_plot_pr_dt, p))

    # lgd = ax.legend(plots, legends, fontsize = p['font'][2])
    # lgd.get_frame().set_alpha(0.5)


if __name__ == '__main__':
    n1, f1, n2, f2 = np.random.rand(200), (np.random.rand(400) + 3) * 100, \
        np.random.rand(200), (np.random.rand(400) + 3) * 100
    data1 = [n1, np.concatenate([np.array(n1[-1]).reshape(-1,), f1, np.array(n2[0]).reshape(-1,)]),
             n2, np.concatenate([np.array(n2[-1]).reshape(-1,), f2]) ]
    colors = ['b', 'r']
    p = {'figsize': (27.7, 9),
         'font': (54, 49, 54),
         'xlabel': r'Samples',
         'ylabel': r'$T^2$',
         'legend': [r'Normal', r'Faulty'],
         'colors': colors,
         'thrd': 30,
         'yscale': 'log'}
    _plot(data1, path ='../Result/_plot', file ='example', **p)
