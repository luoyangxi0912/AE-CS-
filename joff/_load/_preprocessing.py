import numpy as np
from sklearn.preprocessing import StandardScaler
from joff._test._fd import _cal_cov_halfinv

def remove_missing_values(data_list, label_list):
    """
    严格版本：删除包含0值或NaN的样本
    """
    missing_cnt = []
    cleaned_data_list = []
    cleaned_label_list = []

    total_original = sum(len(data) for data in data_list)

    for i, _data in enumerate(data_list):
        _label = label_list[i]

        # 检查每行是否包含0值或NaN
        rows_with_zeros = np.any(_data == 0, axis=1)
        rows_with_nan = np.any(np.isnan(_data), axis=1)
        invalid_rows = rows_with_zeros | rows_with_nan

        # 保留有效的行
        valid_rows = ~invalid_rows
        valid_indices = np.where(valid_rows)[0]

        missing_count = np.sum(invalid_rows)
        missing_cnt.append(missing_count)

        cleaned_data_list.append(_data[valid_indices])
        cleaned_label_list.append(_label[valid_indices])

        # print(f'数据集 {i}: 原始样本数={_data.shape[0]}, 无效样本={missing_count}, 保留样本数={len(valid_indices)}')

    total_missing = np.sum(missing_cnt)
    print(
        f'\n总计: 原始样本数={total_original}, 缺失值总数={total_missing}, 缺失率={total_missing / total_original * 100:.2f}%')

    return cleaned_data_list, cleaned_label_list, missing_cnt

def remove_abnormal_values_n_sigma(data_list, label_list, task, n_sigma,
                                   detect_mode = 'std'):
    """
    基于3-sigma原则删除异常值

    参数:
    data_list: 数据列表，每个元素是一个数据子集
    label_list: 对应的标签列表

    返回:
    清理后的data_list和label_list，以及异常值统计
    """
    # 合并所有数据进行全局标准化
    data_array0 = np.concatenate(data_list).astype('float')
    label_array0 = np.concatenate(label_list).astype('int')
    if task == 'fd':
        data_array0 = data_array0[label_array0 == 0]
    if detect_mode == 'cov':
        mean = np.mean(data_array0, axis=0)
        cov = np.cov(data_array0, rowvar=False)
        _T2_cov_sqrt_inv = _cal_cov_halfinv(cov)
        data_array0_normalized = (data_array0 - mean) @ _T2_cov_sqrt_inv.T
    elif detect_mode == 'std':
        scaler0 = StandardScaler()
        scaler0.fit(data_array0)
        data_array0_normalized = scaler0.transform(data_array0)

    # 计算全局的误差统计量
    _error0 = np.sum(data_array0_normalized ** 2, axis=-1)  # 马氏距离的近似
    _mean0 = np.mean(_error0)
    _std0 = np.std(_error0)

    print(f"\n全局误差统计: mean={_mean0:.4f}, std={_std0:.4f}")
    print(f"异常值阈值: [{_mean0 - n_sigma * _std0:.4f}, {_mean0 + n_sigma * _std0:.4f}]")

    abnormal_cnt = []
    cleaned_data_list = []
    cleaned_label_list = []

    for i, _data in enumerate(data_list):
        _label = label_list[i]

        # 使用全局标准化器转换当前数据
        if detect_mode == 'cov':
            _data_normalized = (_data - mean) @ _T2_cov_sqrt_inv.T
        elif detect_mode == 'std':
            _data_normalized = scaler0.transform(_data)


        _error1 = np.sum(_data_normalized ** 2, axis=-1)

        # 修正：保留在3-sigma范围内的数据点
        # 条件应该是：mean - 3*std <= error <= mean + 3*std
        if task == 'fd':
            normal_indices = np.where(
                ((_error1 >= _mean0 - n_sigma * _std0) & (_error1 <= _mean0 + n_sigma * _std0) & (_label == 0)) | (_label != 0)
            )[0]  # 注意这里的[0]来获取索引数组
        else:
            normal_indices = np.where(
                (_error1 >= _mean0 - n_sigma * _std0) & (_error1 <= _mean0 + n_sigma * _std0)
            )[0]  # 注意这里的[0]来获取索引数组

        abnormal_count = _data.shape[0] - len(normal_indices)
        abnormal_cnt.append(abnormal_count)

        # 保留正常的数据点
        cleaned_data_list.append(_data[normal_indices])
        cleaned_label_list.append(_label[normal_indices])

        # print(f'数据集 {i}: 原始样本数={_data.shape[0]}, 异常值={abnormal_count}, 保留样本数={len(normal_indices)}')

    total_abnormal = np.sum(abnormal_cnt)
    total_original = sum(len(data) for data in data_list)
    print(
        f'总计: 原始样本数={total_original}, 异常值总数={total_abnormal}, 异常率={total_abnormal / total_original * 100:.2f}%\n')

    return cleaned_data_list, cleaned_label_list, abnormal_cnt

