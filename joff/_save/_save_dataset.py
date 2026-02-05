import numpy as np

def save_npz(dataset_tuple, dataset_name, task):
    train_X, train_Y, test_X, test_Y = dataset_tuple
    save_path0 = '../Dataset/' + dataset_name + '/' + task
    if_save_y = False if task == 'fd' else True
    print('')
    _save_npz(train_X, train_Y, save_path0 + '/train/train_', if_save_y)
    _save_npz(test_X, test_Y, save_path0 + '/test/test_', True)

def _save_npz(X, Y, save_path, if_save_y):
    if type(X) == list:
        dict_X, dict_Y = {}, {}
        for i in range(len(X)):
            dict_X['{:02d}_X'.format(i + 1)] = X[i]
            dict_Y['{:02d}_Y'.format(i + 1)] = Y[i]

        np.savez(save_path + 'X.npz', **dict_X)
        print('已成功保存数据集至' + save_path + 'X.npz，含{}个子array'.format(len(X)))
        if if_save_y:
            np.savez(save_path + 'Y.npz', **dict_Y)
            print('已成功保存数据集至' + save_path + 'Y.npz，含{}个子array'.format(len(X)))
    else:
        np.savez(save_path + 'X.npz', X)
        print('已成功保存数据集至' + save_path + 'X.npz')
        if if_save_y:
            np.savez(save_path + 'Y.npz', Y)
            print('已成功保存数据集至' + save_path + 'Y.npz')
