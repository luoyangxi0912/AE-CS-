
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from joff.customize import _plot_pr_dt
from joff._nn._attr import _update_dict
from joff._plot._basic_plot_func import _get_subplot_layout, _generate_colors, _gradient_lines, _multi_colored_line

# data：list 则多个图；否则一个图
def _view_var_plot(X, Y, path = '', if_concat_plot = False, **kwargs):
    p = _update_dict(_plot_pr_dt, kwargs)
    p['linewidth'] = 2
    if if_concat_plot and type(X) == list:
        X, Y = np.concatenate(X, 0), np.concatenate(Y, 0)

    # data 为 list，每个data[i]绘制单独的图，并为var分配子图
    if type(X) == list:
        n_class = np.max(np.concatenate(Y))
        for i, d in enumerate(X):
            print('Plot var view of DataSeg {}...'.format(i+1))
            if not os.path.exists(path + '/ViewDataSeg'): os.makedirs(path + '/ViewDataSeg')
            _subplot_for_var(d, Y[i], n_class, path + '/ViewDataSeg/DataSeg ' + str(i + 1), **p)
    else:
        n_class = np.max(Y)
        print('Plot var view of DataSet ...')

        # data 为 concat array，每个var绘制单独的图
        if if_concat_plot: _plot_each_var(X, Y, n_class, path + '/ViewDataConcat', **p)

        # data 为 array，绘制一张图，并为var分配子图
        else: _subplot_for_var(X, Y, n_class, path + '/ViewDataWhole', **p)

def _plot_each_var(X, Y, n_class, path = '', **p):
    p = _update_dict(_plot_pr_dt, p).copy()
    if 'color' in p.keys(): p.pop('color')
    p['linewidth'] = 1

    if not os.path.exists(path): os.makedirs(path)

    if len(Y.shape) > 1 and Y.shape[1] > 1:
        Y = np.array(np.argmax(Y, axis=1).reshape(-1, 1), dtype=np.float32)

    time = np.arange(X.shape[0])
    colors = _generate_colors(n_class)

    for i in range(X.shape[1]):
        print('Plot var {} for DataConcat ...'.format(i + 1))
        fig = plt.figure(figsize=[40, 18])
        ax = fig.add_subplot(111)
        _multi_colored_line(time, X[:,i], Y, colors, False, ax, **p)
        plt.savefig(path + '/v'+str(i+1)+'.pdf', bbox_inches='tight')
        # plt.savefig(file_name + '.png', bbox_inches='tight')
        plt.close(fig)

def _subplot_for_var(X, Y, n_class, file_name = '', **p):
    p = _update_dict(_plot_pr_dt, p).copy()
    if 'color' in p.keys(): p.pop('color')

    time = np.arange(X.shape[0])

    # 创建子图
    fig, axs = _get_subplot_layout(X.shape[1])
    colors = _generate_colors(n_class)

    # 绘制数据
    for i in range(X.shape[1]):  # 循环子图（变量）
        # 构造折线段
        row, col = int(i/axs.shape[1]), int(np.mod(i, axs.shape[1]))
        start_p = 0
        for j in range(1, len(time)):  # 循环样本点
            if Y[j] == Y[start_p]: continue
            axs[row, col].plot(time[start_p:j], X[start_p:j, i], color=colors[Y[start_p]], **p)
            _gradient_lines([(time[j - 1], X[j - 1, i]), (time[j], X[j, i])], [colors[Y[j - 1]], colors[Y[j]]], axs[row, col], **p) # 使用渐变色
            start_p = j
        axs[row, col].plot(time[start_p:len(time)], X[start_p:len(time), i], color=colors[Y[start_p]], **p)
        # 绘制散点
        for c in range(n_class):
            if c not in Y: continue
            label = 'Normal' if c == 0 else 'Fault {:0>2d}'.format(c)
            axs[row, col].plot(time[Y == c], X[Y == c, i], '.', label=label, markersize=8, color=colors[c])
        axs[row, col].set_ylabel(f'Variable {i + 1}')
        axs[row, col].legend(loc='lower right', frameon=False)

        # 设置横轴标签
        axs[row, col].set_xlabel('Time')

    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    # plt.savefig(file_name + '.png', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 生成随机数矩阵
    np.random.seed(1)
    data = np.random.rand(50, 5)
    classes = np.random.randint(0, 3, size=50)
    _view_var_plot(data, classes)
