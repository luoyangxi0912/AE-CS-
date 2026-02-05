import numpy as np
from collections import Counter

def _stats_data_categories(Y_list, name):
    """快速统计函数"""
    Y_list = Y_list.copy()
    for i in range(len(Y_list)):
        if Y_list[i].ndim > 1:
            Y_list[i] = np.argmax(Y_list[i], axis= -1)
    all_labels = np.concatenate([y.flatten() for y in Y_list])
    counter = Counter(all_labels)

    print(f"\n{name}:")
    print(f"  总样本: {len(all_labels)}, 分段数: {len(Y_list)}, 类别数: {len(counter)}")
    sorted_dist = dict(sorted(counter.items()))
    print(f"  分布: {sorted_dist}")

    return counter