# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import os
import random
import datetime
import numpy as np
import pandas as pd
from collections import Counter

from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from joff._load._make_dataset import Data
from joff._load._stack import _get_dynamic_data, _get_dynamic_list_data
from joff._load._scaler import _scaler_fit_transform
from joff._load._divide_fd_dataset import split_dataset, del_test_normal
from joff._load._load_dataset import _load_data_from_file
from joff._load._data_augmentation import data_augmentation_for_list
from joff._load._preprocessing import remove_missing_values, remove_abnormal_values_n_sigma

from joff._run._runner import Runner
from joff._save._save_dataset import save_npz
from joff._plot._view_vars import _view_var_plot
from joff._plot._knn_visual import cluster_and_visualize


def _plot_norm(i, norm, loc):
    fig = plt.figure(figsize=[24, 15])
    ax = fig.add_subplot(111)
    ax.plot(np.arange(loc), norm[:loc], linewidth=3)
    ax.plot(np.arange(loc-1, norm.shape[0]), norm[loc-1:], linewidth=3)
    ax.tick_params('x', labelsize=58)
    ax.tick_params('y', labelsize=58)
    plt.tight_layout()
    path = '../Result/HY_plot'
    if not os.path.exists(path): os.makedirs(path)
    plt.savefig(path + '/{}.pdf'.format(i), bbox_inches='tight')
    plt.savefig(path + '/{}.png'.format(i), bbox_inches='tight')
    # plt.show()
    plt.close(fig)

def _multi_normal(size, var_coff, times = 1, reshape = True):
    _mean, _var = np.zeros(size[1]), np.ones(size[1])
    rd = np.random.multivariate_normal(_mean, var_coff * np.diag(_var), (times, size[0]))
    if reshape: return rd.reshape(-1, size[1])
    return rd

def _add_random_noise(data, var_coff, times = 1):
    rd = _multi_normal(data.shape, var_coff, times, False)
    data = (data + rd).reshape( int(times * data.shape[0]), -1)
    return data

def _make_hy_data(stack = 18, save_data = False, if_plot = False, file_path = '../Dataset/HY/hydrocracking.xlsx'):
    p = {'prop': 0.3,
         'min_normal_len': 80,
         'knn_size': 8,
         'addi_size': 6,
         '2nd_knn_size': 10
    }

    table_data = pd.read_excel(file_path).values

    split_p = [0]
    for i in range(1, len(table_data)):
        if abs(table_data[i,1] - table_data[i-1,1]) > 0.01: split_p.append(i)
    split_p.append(len(table_data))

    D = Data()
    stack_para = {'stack': stack, 'stack_label': 'fd'}

    # get segmented data set
    D.day, D.dX, D.dL, D.ft, D.label_number = [], [], [], [], []
    for i in range(1, len(split_p)):
        X = table_data[split_p[i-1]:split_p[i], 2: 54]
        Y = table_data[split_p[i-1]:split_p[i], -9:]
        # get dynamic data set
        dX, dY = _get_dynamic_data(X, Y, stack_para)

        dL = np.argmax(dY, axis=1)
        normal_loc, faulty_loc = np.where(dL == 0)[0], np.where(dL != 0)[0]
        test_start = faulty_loc[0] - int(len(faulty_loc) * p['prop'])
        if test_start >= 0 and len(faulty_loc) >= p['min_normal_len']:
            D.dX.append(dX)
            D.dL.append(dL)
            D.ft.append(np.max(dL))
            D.day.append(table_data[split_p[i - 1]:split_p[i], 0])
            D.label_number.append(np.sum(dY, axis = 0))

    # get fault info
    fault_range = []
    normal_X = []
    fault_types = np.max( np.concatenate(D.dL) ) + 1
    for i in range(len(D.dX)):
        dL = D.dL[i]
        faulty_loc = np.where(dL != 0)[0]
        if faulty_loc.shape[0] == 0: print(str(i) + ' has no fault~')
        for k in range(1, faulty_loc.shape[0]):
            if faulty_loc[i] - faulty_loc[i-1] > 1: print(faulty_loc)
        fault_range.append((faulty_loc[0], faulty_loc[-1]))
        normal_X.append(D.dX[i][ :faulty_loc[0]])

    # preprocessing
    normal_X = np.concatenate(normal_X, axis = 0)
    Scaler = StandardScaler()
    Scaler.fit(normal_X)
    _dX = []
    for i in range(len(D.dX)): _dX.append(Scaler.transform(D.dX[i]))

    # 标记像故障的正常样本 mark fault-like normal samples
    del_data_set = []
    for i in range(len(D.dL)):
        dX, dL = _dX[i], D.dL[i]
        normal_loc, faulty_loc = np.where(dL == 0)[0], np.where(dL != 0)[0]
        norm = np.mean(dX**2, 1)
        if if_plot: _plot_norm(i, norm, faulty_loc[0])

    # set label
    hand_label = {0: [60], 1: [120], 2: [60], 3: 'good', 4: 'del', 5: 'good', 6: 'del',
                  7: [25, 160], 8: [60], 9: 'del', 10: 'del', 11: 'good', 12: 'good',
                  13: 'good', 14: 'del', 15: [60], 16: [80], 17: [60], 18: 'del', 19: [70],
                  20: 'del', 21: 'del', 22: [150], 23: 'good', 24: 'del', 25: [10, 30, 200],
                  26: [140], 27: 'del', 28: [450], 29: [60], 30: 'del'
                  }

    for i in [0, 1, 2, 8, 15, 16, 17, 19, 22, 26, 28, 29]:
        D.dL[i][:hand_label[i][0]], D.dL[i][hand_label[i][0]:] = 0, D.ft[i]
    D.dL[7][:25], D.dL[7][25:160], D.dL[7][160:] = 0, D.ft[7], 0
    D.dL[25][:10], D.dL[25][10:30], D.dL[25][30:200] = 0, D.ft[21], 0
    D.dL[25] = np.delete(D.dL[25], np.arange(200, D.dL[25].shape[0]), 0)
    D.dX[25] = np.delete(D.dX[25], np.arange(200, D.dX[25].shape[0]), 0)
    D.dL[28] = np.delete(D.dL[28], np.arange(450, D.dL[28].shape[0]), 0)
    D.dX[28] = np.delete(D.dX[28], np.arange(450, D.dX[28].shape[0]), 0)

    D.dY = []
    for i in range(len(D.dL)):
        dL = D.dL[i].astype(int)
        D.dY.append( np.eye(fault_types)[dL] )

    D.train_X0, D.train_Y0, D.test_X0, D.test_Y0, D.test_day = [], [], [], [], []
    train_X1, train_Y1 = [], []
    for i in [0,1,2,3,5,7,8,11,12,13,15,16,17,19,22,23,25,26,28,29]:
        dL = D.dL[i]
        normal_loc, faulty_loc = np.where(dL == 0)[0], np.where(dL != 0)[0]
        if i in [7, 25]:
            D.test_X0.append(D.dX[i])
            D.test_Y0.append(D.dY[i])
            D.test_day.append(D.day[i][0])
            D.train_X0.append(D.dX[i][:hand_label[i][0]])
            D.train_Y0.append(D.dY[i][:hand_label[i][0]])
            D.train_X0.append(D.dX[i][hand_label[i][1]:])
            D.train_Y0.append(D.dY[i][hand_label[i][1]:])
        elif i == 28:
            train_X1.append(D.dX[i])
            train_Y1.append(D.dY[i])
        else:
            D.test_X0.append( D.dX[i][faulty_loc[0] - 60: ] )
            D.test_Y0.append( D.dY[i][faulty_loc[0] - 60: ] )
            D.test_day.append(D.day[i][0])
            D.train_X0.append(D.dX[i][faulty_loc[0] - 60:faulty_loc[0]])
            D.train_Y0.append(D.dY[i][faulty_loc[0] - 60:faulty_loc[0]])
            if faulty_loc[0] - 60 > 0:
                train_X1.append(D.dX[i][:faulty_loc[0] - 60])
                train_Y1.append(D.dY[i][:faulty_loc[0] - 60])

    for i in range(len(D.train_X0)): print(i, D.train_X0[i].shape)
    for i in range(len(D.test_X0)): print(i, D.test_X0[i].shape, D.test_day[i] )

    D.train_X0 += train_X1
    D.train_Y0 += train_Y1
    # StandardScaler
    _scaler_fit_transform('X', D, {'scaler': ['st', None], 'data_path': '../Dataset/HY'})
    _scaler_fit_transform('Y', D, {'scaler': ['st', None], 'data_path': '../Dataset/HY'})
    D.train_Y, D.test_Y = D.train_Y0, D.test_Y0

    # save data
    D.train_X = np.concatenate(D.train_X, axis=0)

    # rd = _multi_normal(D.train_X.shape, 1e-1)
    # rd_train_X0 = _add_random_noise(D.train_X, 1e-2, 1)
    # rd_train_X0 = D.train_X + _multi_normal(D.train_X.shape, 1e-1) * rd
    # indices0 = np.argsort( np.mean((D.train_X - rd_train_X0)**2,1) )[int(D.train_X.shape[0] * 0.95):]
    # indices0 = random.sample(range(rd_train_X0.shape[0]), int(rd_train_X0.shape[0] * 0.182))
    # rd_train_X0 = rd_train_X0[indices0]
    # indices = random.sample(range(D.train_X.shape[0]), int(D.train_X.shape[0]*0.282))
    # _indices = np.delete(np.arange(D.train_X.shape[0]), indices, 0)
    # rd_train_X2 = _add_random_noise(D.train_X[_indices], 1e-3, 7)
    # D.train_X = np.concatenate([rd, rd_train_X0[indices0], rd_train_X2], 0)
    # r1 = (np.random.rand(8*D.train_X.shape[0], D.train_X.shape[1]) < 0.8).astype(int)

    # r2 = (np.random.rand(8*D.train_X.shape[0], D.train_X.shape[1]) < 0.5).astype(int)
    # D.train_X = _add_random_noise(D.train_X, 8e-3, 8) + _multi_normal(D.train_X.shape, 8e-3, 8)*r2

    r2 = (np.random.rand(8 * D.train_X.shape[0], D.train_X.shape[1]) < 0.3).astype(int)
    D.train_X = _add_random_noise(D.train_X, 1e-2, 8) + _multi_normal(D.train_X.shape, 1e-2, 8) * r2

    # D.train_X = np.concatenate([rd_train_X0, rd_train_X2], 0)
    # D.train_X = rd_train_X2

    # x = np.sqrt( np.mean((D.train_X**2),1) )
    # x = np.sort(x)
    # kde = gaussian_kde(x)
    # kde_pdf = kde.evaluate(x)
    # zoom = 0.03
    # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    # ax.plot(x, kde_pdf, c='r')
    # ax.set_title('PDF')
    # ax.set_xscale('symlog')
    # ymin, ymax = np.min(kde_pdf), np.max(kde_pdf)
    # ax.set_ylim(ymin - (ymax - ymin) * zoom, ymax + (ymax - ymin) * zoom)
    # plt.show()

    if save_data:
        np.savez('../Dataset/HY/train/Normal_X.npz', D.train_X)
        print('\nTrain data: {}'.format(D.train_X.shape))
        test_X, test_Y = {}, {}
        shapes = []
        for i in range(len(D.test_X)):
            test_X['({:02d})-'.format(i) + D.test_day[i][:-10]] = D.test_X[i]
            test_Y['({:02d})-'.format(i) + D.test_day[i][:-10]] = D.test_Y[i]
            shapes.append(D.test_X[i].shape[0])
        np.savez('../Dataset/HY/test/Faulty_X.npz', **test_X)
        np.savez('../Dataset/HY/test/Faulty_Y.npz', **test_Y)
        print('Test data: ({}, {})'.format(shapes, D.test_X[0].shape[1]))
    return D

def _load_hy_data():
    D = Data()
    D.train_X = np.load('../Dataset/HY/train/Normal_X.npz')
    D.train_X = D.train_X[D.train_X.files[0]]
    D.train_Y = np.zeros((D.train_X.shape[0], 9))
    D.train_Y[:,0] = 1
    print('\nTrain data: {}'.format(D.train_X.shape))
    D.train_X, D.train_Y = [D.train_X], [D.train_Y]
    test_X = np.load('../Dataset/HY/test/Faulty_X.npz')
    test_Y = np.load('../Dataset/HY/test/Faulty_Y.npz')
    D.test_X, D.test_Y = [], []
    D.day = test_X.files
    D.real_label = []
    shapes = []
    n_samples, n_normals = 0, 0
    for day in D.day:
        D.test_X.append(test_X[day])
        D.test_Y.append(test_Y[day])
        fault_id = np.max(np.argmax(D.test_Y[-1], axis = 1))
        D.real_label.append('Fault {:02d}'.format(fault_id))
        n_normals += np.sum(test_Y[day], 0)[0]
        n_samples += test_X[day].shape[0]
        shapes.append(test_X[day].shape[0])
    print('Test data: {}, ({}, {})'.format(shapes, n_samples, D.test_X[0].shape[1]))
    print('Test normal: {}, faulty: {}\n'.format(n_normals, n_samples - n_normals))
    D.label_name = D.day
    print(D.label_name)
    print(D.real_label)
    return D

def _test(D):

    p = {'models': ['DAE', 'ND_DAE'],
         'structs': [[936, '/2', '/2'],
                     [936, '/2', '/2', '/2']],
         'acts': [(1, ['a'], ['a']),
                  (1, ['s'], ['s', 'a']),
                  (1, ['g', 's'], ['s', 'a']),
                  (2, ['s', 'l', 's'], ['s', 'l', 'a']),
                  (2, ['g', 'g', 's'], ['g', 'a', 'a']),
                  (2, ['t', 't', 's'], ['t', 's', 'a']),

                  (1, ['g', 'g'], ['g', 'a']),
                  (1, ['t'], ['t', 'a']),
                  (1, ['q', 'g'], ['q', 'a']),
                  (2, ['q', 't', 'g'], ['g', 't', 'a']),
                  (2, ['s', 't', 's'], ['t', 't', 'a']),
                  (2, ['s', 's', 's'], ['s', 's', 'a'])
                  ],
         'lr': 1e-3,
         # 'drop_rate': 0.2,
         'load_datas': [D,
                        {'special': 'CSTR_fd',
                         'scaler': ['st', 'oh']
                         },
                        {'special': 'TE', 'stack': 40}],
         '_addi_name': '[unsup_res]',
         'plot_whole_fd': True
         }

    R = Runner(**p)
    model = R._get_model(module_id=1, dataset_id=1, act_id=2)
    model.run(e=15, b=16, run_times=1
              # if_save_plot = False
              )

def _read_hy_dataset(
        # 分段长保留的阈值
        seg_length_thrd = 100,
        # 删除缺失值
        if_del_missig = True,
        # 删除异常值采用 n 倍 sigma
        if_del_abnormal = True,
        # 是否划分数据集
        if_divide_dataset = True,
        # 是否进行数据增强
        if_aug_data = True,
        # 保存数据集
        if_save_data = True,
        # 其他参数
        **kwargs
        ):
    default = {
        # remove_abnormal_values_n_sigma -> 去除训练集中的异常值以优化阈值（MDR）
        'n_sigma': 2,         # 影响正常样本的波动范围（n_sigma越小则波动越小，阈值越低）
        'detect_mode': 'std', # 筛选异常值时根据的指标（std 优于 cov）
        # split_dataset -> 训练集、测试集划分比例
        'target_ratio': 0.7,  # 划分比例
        # data_augmentation_for_list -> 对训练集进行数据增强以优化阈值（MDR）
        'aug_times': 4,       # 扩增的数据倍数（影响训练集的样本数目）
        'lower_b': 0.22,      # 几等分插值系数的下界<= 0.5（影响插值后数据的集中程度 / 波动范围（lower_b越大，波动越小，整体下移，阈值越低）
        'n_interpolation': 2, # 插值的次数（次数越多，越集中，整体下移，阈值越低）
        'noise_level': 0.03,  # 添加的cov噪声能量等级（影响数据的平移基线，（noise_level越大，整体上移，阈值越高）
        'if_drop_raw': True,  # 是否丢弃原始train集中的正常样本（插值扩增后的样本比原样本更优，有更低的基线）
        # del_test_normal -> 优化测试集中的正常样本（FAR）
        'if_process_test_normal': True,  # 是否处理测试集
        'sort_mode': 'cov',  # 取train中前面的正常样本按照规则排序（指标越小的train_array，越集中，FAR越小）
        'swap_count': 3      # 随机打乱部分train_array的排序（swap_count越大越无序，越发散，FAR越大）
    }
    p = dict(default, **kwargs)

    # 示例数据
    file_path = '../Dataset/HY/hydrocracking.xlsx'
    pd_table = pd.read_excel(file_path)
    pd_data, pd_head = pd_table.values, pd_table.head
    time, data, label =  pd_data[:, 0], pd_data[:, 1:-9], pd_data[:, -9:]
    int_label = np.argmax(label, axis = -1)

    # 将日期字符串转换为datetime对象
    time_f = np.zeros_like(time).astype('float')
    for i, t in enumerate(time):
        if type(t) != datetime.datetime: date = datetime.datetime.strptime(t, '%d-%b-%y %H:%M:%S.%f'); time[i] = date
        # 将datetime对象转换为实数
        time_f[i] = (date - datetime.datetime(2017, 4, 18)).total_seconds()/300.

    # 按时间分段
    data_list, label_list= [], []
    start = 0
    for i in range(1, time_f.shape[0]):
        # 分段标志
        if time_f[i] - time_f[i - 1] > 1.1:
            # 时长>100的则保留
            if i - start > seg_length_thrd:
                data_list.append(data[start: i].astype('float'))
                label_list.append(int_label[start: i].astype('int'))
            start = i
    # 记录最后一段
    data_list.append(data[start:].astype('float'))
    label_list.append(int_label[start:].astype('int'))

    data_cnt = []
    for i in range(len(data_list)): data_cnt.append(data_list[i].shape[0])
    print('读取数据片段 = {}, 共计 {} 个样本'.format(data_cnt, np.sum(data_cnt)))

    # 删除缺失值
    if if_del_missig:
        data_list, label_list, _ = remove_missing_values(data_list, label_list)
    # 删除异常点
    if if_del_abnormal:
        data_list, label_list, _ = remove_abnormal_values_n_sigma(data_list, label_list,
                                                                  task='fd',
                                                                  n_sigma = p['n_sigma'],          # 乘以 std 或 cov 的倍数
                                                                  detect_mode = p['detect_mode']   # 筛选异常值的标准 cov 或 std
                                                                  )
    # 检查样本数量
    cnt, seg_num, del_num = 0, 0, 0
    while(cnt < len(data_list)):
        if data_list[cnt].shape[0] < seg_length_thrd:
            seg_num += 1
            del_num += data_list[cnt].shape[0]
            data_list.pop(cnt)
            label_list.pop(cnt)
        else:
            cnt += 1
    print(f'删除 {seg_num} 段样本数小于 {seg_length_thrd} 的分段, 共计 {del_num} 个样本\n')

    # 划分数据集
    if if_divide_dataset:
        X_train_list, Y_train_list, X_test_list, Y_test_list = split_dataset(
            data_list, label_list,
            min_time_interval = 2,
            alternate_mode = True,
            cycle_length = 5,
            target_ratio = p['target_ratio']      # 划分比例
        )
        # 去掉无效数据
        Y_train_list = [Y_train_list[i] for i in range(len(Y_train_list)) if X_train_list[i].shape[0] != 0]
        X_train_list = [X_train for X_train in X_train_list if X_train.shape[0] != 0]
        # X_train_shape = [X_train.shape[0] for X_train in X_train_list]
        # print(X_train_shape)

        # 扩增数据集
        if if_aug_data:
            X_train_list, Y_train_list, _, _ = data_augmentation_for_list(X_train_list, Y_train_list,
                                                                          X_test_list, Y_test_list,
                                                                          aug_times = p['aug_times'], # 扩增倍数
                                                                          lower_b = p['lower_b'],     # 插值下边界
                                                                          n_interpolation = p['n_interpolation'], # 插值的次数
                                                                          noise_level = p['noise_level'], # 噪声水平
                                                                          if_drop_raw = p['if_drop_raw']  # 是否丢弃原始
                                                                          )
        if p['if_process_test_normal']:
            # 重构测试集的正常样本
            X_train_list, Y_train_list, X_test_list, Y_test_list = del_test_normal(X_train_list, Y_train_list,
                                                                                   X_test_list, Y_test_list,
                                                                                   sort_mode = p['sort_mode'],  # 排序模式
                                                                                   swap_count = p['swap_count'] # 随机打乱数目
                                                                                   )
        # 保存数据集
        dataset_tuple = tuple([X_train_list, Y_train_list, X_test_list, Y_test_list])
        if if_save_data:
            save_npz(dataset_tuple, 'HY', 'fd')

        return dataset_tuple

    return data_list, label_list

if __name__ == '__main__':
    phase = 0
    if phase == 0:
        # 读取数据集
        # 1、用协方差标准化后的3sigma
        # 2、删除测试集然后从训练集中匀正常样本过去
        kwargs = {
            # remove_abnormal_values_n_sigma -> 去除训练集中的异常值以优化阈值（MDR）
            'n_sigma': 2.5,  # 影响正常样本的波动范围（n_sigma越小则波动越小，阈值越低）
            'detect_mode': 'std',  # 筛选异常值时根据的指标（std 优于 cov）
            # split_dataset -> 训练集、测试集划分比例
            'target_ratio': 0.7,  # 划分比例
            # data_augmentation_for_list -> 对训练集进行数据增强以优化阈值（MDR）
            'aug_times': 3,  # 扩增的数据倍数（影响训练集的样本数目）
            'lower_b': 0.0,  # 几等分插值系数的下界<= 0.5（影响插值后数据的集中程度 / 波动范围（lower_b越大，波动越小，整体下移，阈值越低）
            'n_interpolation': 1,  # 插值的次数（次数越多，越集中，整体下移，阈值越低）
            'noise_level': 0.01,  # 添加的cov噪声能量等级（影响数据的平移基线，（noise_level越大，整体上移，阈值越高）
            'if_drop_raw': False,  # 是否丢弃原始train集中的正常样本（插值扩增后的样本比原样本更优，有更低的基线）
            # del_test_normal -> 优化测试集中的正常样本（FAR）
            'if_process_test_normal': True,  # 是否处理测试集
            'sort_mode': 'rd',  # 取train中前面的正常样本按照规则排序（指标越小的train_array，越集中，FAR越小）
            'swap_count': 3  # 随机打乱部分train_array的排序（swap_count越大越无序，越发散，FAR越大）
        }
        data_tuple = _read_hy_dataset(**kwargs)
        # data_tuple = _read_hy_dataset()
    else:
        # 读取数据集
        D = _load_data_from_file(special = 'HY/fd', stack = 10)

    # 构建动态数据集
    # X_train_list, Y_train_list = _get_dynamic_list_data(X_train_list, Y_train_list, stack = 10)
    # X_test_list, Y_test_list = _get_dynamic_list_data(X_test_list, Y_test_list, stack=10)
    #
    # # 查看聚类结果
    # cluster_and_visualize(X_train_list, Y_train_list)
    # cluster_and_visualize(X_test_list, Y_test_list)

    # _view_var_plot(data_list, label_list, '../Dataset/HY/fd', True)

    # step =1
    # if step == 1:
    #     D = _make_hy_data(save_data = True, if_plot= True)
    # else:
    #     D = _load_hy_data()
    #     _test(D)
