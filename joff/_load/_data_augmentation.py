import numpy as np
from collections import defaultdict
from scipy.stats import truncnorm
from joff._load._stats_data_attri import _stats_data_categories
import warnings

def data_augmentation_for_list(train_X, train_Y, test_X, test_Y,
                               aug_times = 1, lower_b = 0.1, n_interpolation = 2, noise_level = 0.05,
                               c_list = 'all', task = 'fd', if_drop_raw = False, if_aug_test = False):
    """对指定标签的样本进行数据增强"""
    # 特殊的 n_list
    if task == 'fd':
        c_list = [0]
    elif c_list == 'all':
        max_c = np.max(np.concatenate(train_Y + test_Y, axis=0))
        c_list = list(np.arange(max_c))

    print(f'经过数据扩增（×{aug_times}）之后：')

    # 收集需要增强的样本索引（第i个数据集中的第j个样本）
    augmented_X, augmented_Y = {}, {}
    for phase in ['train', 'test']:
        augmented_X[phase], augmented_Y[phase] = [], []
        if phase == 'train': data_list, label_list = train_X, train_Y
        else: data_list, label_list = test_X, test_Y
        if not if_aug_test and phase == 'test': break
        for i in range(len(label_list)):
            for c in c_list:
                indices = np.where(label_list[i] == c)[0]
                data_i_c = data_list[i][indices]
                if len(data_i_c) <= 1: continue
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    # 局部协方差矩阵
                    cov = np.cov(data_i_c.T)
                    cov = (cov + cov.T) / 2
                    cov += np.diag([1e-6]*cov.shape[0])

                    for n in range(aug_times):
                        data_aug = generate_augmented_data(data_i_c, cov,
                                                           lower_b = lower_b,
                                                           n_interpolation = n_interpolation,
                                                           noise_level = noise_level)
                        label_aug = (np.ones(data_aug.shape[0]) * c).astype(int)
                        augmented_X[phase].append(data_aug)
                        augmented_Y[phase].append(label_aug)
        if phase == 'train':
            if not if_drop_raw:
                augmented_X[phase] = train_X.copy() + augmented_X[phase]
                augmented_Y[phase] = train_Y.copy() + augmented_Y[phase]
            _stats_data_categories(augmented_Y[phase], "训练集")
        else:
            if not if_drop_raw:
                augmented_X[phase] = test_X.copy() + augmented_X[phase]
                augmented_Y[phase] = test_Y.copy() + augmented_Y[phase]
            _stats_data_categories(augmented_Y[phase], "测试集")

    return augmented_X['train'], augmented_Y['train'], augmented_X['test'], augmented_Y['test']

def generate_augmented_data(data_curr, cov,
                            lower_b = 0.1, n_interpolation = 2,
                            if_add_noise = True, noise_level = 0.05):

    n, m = data_curr.shape[0], data_curr.shape[-1]
    ids = np.arange(n).tolist()
    data_prev = data_curr[ [0] + ids[:-1] ]
    data_next = data_curr[ ids[1:] + [-1] ]

    data_aug = data_curr.copy()
    lower, upper = lower_b, 1- lower_b
    if n_interpolation >= 1:
        alpha = np.random.uniform(lower, upper, n)[:, np.newaxis]
        # data_prev 指向 data_aug 的 alpha 等分点上
        data_aug = (1 - alpha) * data_prev + alpha * data_aug

    if n_interpolation >= 2:
        alpha = np.random.uniform(lower, upper, n)[:, np.newaxis]
        # data_next 指向 data_aug 的 alpha 等分点上
        data_aug = (1 - alpha) * data_next + alpha * data_aug

    if if_add_noise:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            noise = np.random.multivariate_normal(np.zeros(m), cov, n) * noise_level
        data_aug += noise
    return data_aug


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    train_X = [np.random.rand(10, 5), np.random.rand(8, 5)]
    train_Y = [np.array([0, 1, 2, 1, 0, 2, 1, 0, 2, 1]), np.array([1, 0, 2, 1, 0, 2, 1, 0])]
    test_X = [np.random.rand(6, 5)]
    test_Y = [np.array([0, 1, 2, 1, 0, 2])]

    # 数据增强
    new_train_X, new_train_Y = data_augmentation_for_list(train_X, train_Y, test_X, test_Y, c_list=[1, 2])

    print(f"增强前: {len(train_X)}个array")
    print(f"增强后: {len(new_train_X)}个array")
    if len(new_train_X) > len(train_X):
        print(f"新增数据形状: {new_train_X[-1].shape}")