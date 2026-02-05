# -*- coding: utf-8 -*-
'''
    我在做故障检测任务的数据集划分，现在我有一个数据集列表元组（X_list, Y_list）,包括数据和数据标签（0为正常，非0为各类故障）。请你帮我写python代码实现。
    你应该分别对list中每一段array做处理，从每一段array抽取数据组成新的训练集和测试集。最终结果应该是两个列表元组，列表长度与原来一致。
    你可以写多个函数来实现该功能。

    要求：
    1、训练集中全是正常样本，测试集中包含正常和故障样本；
    2、训练集和测试集中样本的顺序仍然遵循划分之前的顺序；
    3、划分到训练集中的正常样本应该与异常样本有足够远的采样时间间隔，即该正常样本（标签为0）到最近的异常样本（标签为非0）的时间间隔应该大于某个阈值，
       若不满足条件则只能被划分到测试集；
    4、对于满足条件的正常样本（即既可划分到训练集也可划分到测试集的样本）应采用交替分配的形式，可以设定一个变量来控制是否开启交替分配模式，
       用另一变量控制交替周期（一个周期内按预设比例分配满足条件3的正常样本到训练集和测试集）；
    5、确保最终划分训练集和测试集的总样本比例接近预设值。
'''

import random
import numpy as np
from typing import List, Tuple, Any
from joff._load._stats_data_attri import _stats_data_categories
from joff._load._scaler import _dataset_fit_transform

def calculate_time_intervals(normal_indices: np.ndarray, fault_indices: np.ndarray) -> np.ndarray:
    """
    计算每个正常样本到最近故障样本的时间间隔
    """
    if len(fault_indices) == 0:
        return np.full(len(normal_indices), np.inf)

    time_intervals = np.zeros(len(normal_indices))
    for i, normal_idx in enumerate(normal_indices):
        min_interval = np.min(np.abs(normal_idx - fault_indices))
        time_intervals[i] = min_interval

    return time_intervals


def calculate_global_allocation_plan(X_list: List[np.ndarray], Y_list: List[np.ndarray],
                                     min_time_interval: int, target_ratio: float) -> List[Tuple[int, int]]:
    """
    计算全局分配计划，确定每个段应该分配到训练集的正常样本数量
    """
    # 统计所有段的正常样本信息
    segment_info = []
    total_normal = 0
    total_candidates = 0

    for i, (X_segment, Y_segment) in enumerate(zip(X_list, Y_list)):
        normal_indices = np.where(Y_segment == 0)[0]
        fault_indices = np.where(Y_segment != 0)[0]

        # 计算时间间隔
        time_intervals = calculate_time_intervals(normal_indices, fault_indices)

        # 筛选候选样本
        candidate_indices = normal_indices[time_intervals > min_time_interval]

        segment_info.append({
            'segment_idx': i,
            'total_normal': len(normal_indices),
            'candidate_count': len(candidate_indices),
            'candidate_indices': candidate_indices
        })

        total_normal += len(normal_indices)
        total_candidates += len(candidate_indices)

    # 计算全局目标训练集大小
    target_train_size = int(total_normal * target_ratio)

    # 如果候选样本总数小于目标训练集大小，则使用所有候选样本
    if total_candidates < target_train_size:
        # 分配所有候选样本到训练集
        for info in segment_info:
            info['target_train_count'] = info['candidate_count']
        return [(info['segment_idx'], info['target_train_count']) for info in segment_info]

    # 否则，按比例分配目标训练集大小到各个段
    remaining_train_size = target_train_size
    allocation_plan = []

    # 第一轮分配：每个段至少分配其候选样本数的比例
    for info in segment_info:
        proportional_share = int(target_train_size * (info['candidate_count'] / total_candidates))
        actual_share = min(proportional_share, info['candidate_count'])
        info['target_train_count'] = actual_share
        remaining_train_size -= actual_share

    # 第二轮分配：如果有剩余，按候选样本数比例分配
    if remaining_train_size > 0:
        # 按候选样本数排序，优先分配给候选样本多的段
        sorted_segments = sorted(segment_info, key=lambda x: x['candidate_count'], reverse=True)

        for info in sorted_segments:
            if remaining_train_size <= 0:
                break

            # 计算该段还能分配多少
            available = info['candidate_count'] - info['target_train_count']
            if available > 0:
                add_count = min(available, remaining_train_size)
                info['target_train_count'] += add_count
                remaining_train_size -= add_count

    return [(info['segment_idx'], info['target_train_count']) for info in segment_info]


def alternate_allocation(candidate_indices: np.ndarray, target_train_count: int,
                         cycle_length: int = 10, train_ratio_in_cycle: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用交替周期方法分配候选样本

    Args:
        candidate_indices: 候选样本索引
        target_train_count: 目标训练集样本数
        cycle_length: 交替周期长度
        train_ratio_in_cycle: 周期内分配到训练集的比例

    Returns:
        train_indices: 分配到训练集的索引
        test_indices: 分配到测试集的索引
    """
    if len(candidate_indices) == 0:
        return np.array([]), np.array([])

    train_indices = []
    test_indices = []

    # 计算每个周期内分配到训练集的样本数
    train_per_cycle = int(cycle_length * train_ratio_in_cycle)

    # 按周期分配
    for cycle_start in range(0, len(candidate_indices), cycle_length):
        cycle_end = min(cycle_start + cycle_length, len(candidate_indices))
        cycle_indices = candidate_indices[cycle_start:cycle_end]

        # 当前周期的训练样本数
        current_train_count = min(train_per_cycle, len(cycle_indices))

        # 分配当前周期
        for i, idx in enumerate(cycle_indices):
            if i < current_train_count:
                train_indices.append(idx)
            else:
                test_indices.append(idx)

    # 如果训练样本数超过目标，调整
    if len(train_indices) > target_train_count:
        # 随机移除多余的训练样本
        excess = len(train_indices) - target_train_count
        indices_to_remove = np.random.choice(len(train_indices), excess, replace=False)
        for idx in sorted(indices_to_remove, reverse=True):
            removed_idx = train_indices.pop(idx)
            test_indices.append(removed_idx)
    elif len(train_indices) < target_train_count:
        # 从测试集中补充训练样本
        shortage = target_train_count - len(train_indices)
        if shortage <= len(test_indices):
            for _ in range(shortage):
                train_indices.append(test_indices.pop(0))

    return np.array(train_indices), np.array(test_indices)


def split_segment_with_plan(X_segment: np.ndarray, Y_segment: np.ndarray,
                            min_time_interval: int, target_train_count: int,
                            alternate_mode: bool = True, cycle_length: int = 10,
                            train_ratio_in_cycle: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    根据分配计划对单个数据段进行划分
    """
    # 获取正常和故障样本的索引
    normal_indices = np.where(Y_segment == 0)[0]
    fault_indices = np.where(Y_segment != 0)[0]

    # 计算时间间隔
    time_intervals = calculate_time_intervals(normal_indices, fault_indices)

    # 根据时间间隔筛选可分配到训练集的正常样本
    train_candidate_indices = normal_indices[time_intervals > min_time_interval]
    test_only_normal_indices = normal_indices[time_intervals <= min_time_interval]

    # 故障样本全部进入测试集
    X_test_fault = X_segment[fault_indices]
    Y_test_fault = Y_segment[fault_indices]
    X_test_normal_forced = X_segment[test_only_normal_indices]
    Y_test_normal_forced = Y_segment[test_only_normal_indices]

    # 处理可分配到训练集的候选样本
    if len(train_candidate_indices) == 0:
        # 如果没有候选训练样本，所有正常样本都进入测试集
        X_train_seg = np.array([])
        Y_train_seg = np.array([])
        X_test_normal_candidate = X_segment[normal_indices]
        Y_test_normal_candidate = Y_segment[normal_indices]
    else:
        # 确定实际分配到训练集的样本数
        actual_train_count = min(target_train_count, len(train_candidate_indices))

        if alternate_mode and actual_train_count > 0 and actual_train_count < len(train_candidate_indices):
            # 交替分配模式：使用周期交替方法
            train_indices, test_indices = alternate_allocation(
                train_candidate_indices, actual_train_count, cycle_length, train_ratio_in_cycle
            )
        else:
            # 非交替模式：直接按数量分配
            train_indices = train_candidate_indices[:actual_train_count]
            test_indices = train_candidate_indices[actual_train_count:]

        X_train_seg = X_segment[train_indices]
        Y_train_seg = Y_segment[train_indices]
        X_test_normal_candidate = X_segment[test_indices]
        Y_test_normal_candidate = Y_segment[test_indices]

    # 合并测试集
    X_test_parts = []
    Y_test_parts = []

    if len(X_test_fault) > 0:
        X_test_parts.append(X_test_fault)
        Y_test_parts.append(Y_test_fault)

    if len(X_test_normal_forced) > 0:
        X_test_parts.append(X_test_normal_forced)
        Y_test_parts.append(Y_test_normal_forced)

    if len(X_test_normal_candidate) > 0:
        X_test_parts.append(X_test_normal_candidate)
        Y_test_parts.append(Y_test_normal_candidate)

    # 合并所有测试集部分
    if len(X_test_parts) > 0:
        X_test_seg = np.concatenate(X_test_parts, axis=0)
        Y_test_seg = np.concatenate(Y_test_parts, axis=0)
    else:
        X_test_seg = np.array([])
        Y_test_seg = np.array([])

    return X_train_seg, Y_train_seg, X_test_seg, Y_test_seg


def split_dataset(X_list: List[np.ndarray], Y_list: List[np.ndarray],
                  min_time_interval: int = 10, target_ratio: float = 0.7,
                  alternate_mode: bool = True, cycle_length: int = 10) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    主函数：对整个数据集列表进行划分

    Args:
        X_list: 数据列表，每个元素是一个数据段
        Y_list: 标签列表，每个元素是对应的标签段
        min_time_interval: 最小时间间隔阈值
        target_ratio: 训练集目标比例
        alternate_mode: 是否开启交替分配模式
        cycle_length: 交替周期长度

    Returns:
        X_train_list, Y_train_list, X_test_list, Y_test_list:
        划分后的训练集和测试集列表，长度与原始列表一致
    """
    # 计算全局分配计划
    allocation_plan = calculate_global_allocation_plan(
        X_list, Y_list, min_time_interval, target_ratio
    )

    # 将分配计划转换为字典以便查找
    plan_dict = dict(allocation_plan)

    # 初始化结果列表，保持与原始列表相同的长度
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []

    total_train_size = 0
    total_test_size = 0
    total_normal = 0

    for i, (X_segment, Y_segment) in enumerate(zip(X_list, Y_list)):
        print(f"处理第 {i + 1} 个数据段，长度: {len(X_segment)}")

        # 获取该段的目标训练集大小
        target_train_count = plan_dict.get(i, 0)

        X_train_seg, Y_train_seg, X_test_seg, Y_test_seg = split_segment_with_plan(
            X_segment, Y_segment, min_time_interval, target_train_count,
            alternate_mode, cycle_length, target_ratio
        )

        # 将结果添加到列表中
        X_train_list.append(X_train_seg)
        Y_train_list.append(Y_train_seg)
        X_test_list.append(X_test_seg)
        Y_test_list.append(Y_test_seg)

        # 统计信息
        total_train_size += len(X_train_seg)
        total_test_size += len(X_test_seg)
        total_normal += len(np.where(Y_segment == 0)[0])

    # 计算实际比例
    actual_ratio = total_train_size / total_normal if total_normal > 0 else 0

    # 打印统计信息
    print(f"\n划分完成:")
    print(f"训练集总大小: {total_train_size} (全部为正常样本)")
    print(f"测试集总大小: {total_test_size}")
    print(f"目标训练集比例: {target_ratio:.2f}, 实际训练集比例: {actual_ratio:.2f}")

    # 计算测试集中的正常和故障样本
    total_normal_in_test = 0
    total_fault_in_test = 0
    for Y_test_seg in Y_test_list:
        if len(Y_test_seg) > 0:
            total_normal_in_test += np.sum(Y_test_seg == 0)
            total_fault_in_test += np.sum(Y_test_seg != 0)

    print(f"测试集中 - 正常样本: {total_normal_in_test}, 故障样本: {total_fault_in_test}")

    # 快速统计类别数
    train_counter = _stats_data_categories(Y_train_list, "训练集")
    test_counter = _stats_data_categories(Y_test_list, "测试集")

    # 检查类别一致性
    train_classes = set(train_counter.keys())
    test_classes = set(test_counter.keys())

    if train_classes == test_classes:
        print(f"\n✓ 训练集和测试集类别一致")
    else:
        print(f"\n⚠ 训练集和测试集类别不一致:")
        if train_classes - test_classes != set():
            print(f"  训练集特有: {train_classes - test_classes}")
        if test_classes - train_classes != set():
            print(f"  测试集特有: {test_classes - train_classes}\n")

    return X_train_list, Y_train_list, X_test_list, Y_test_list


def del_test_normal(X_train_list, Y_train_list, X_test_list, Y_test_list,
                    sort_mode = 'rd', swap_count = 0):
    """
    处理数据集：
    1. 删除Y_test_list中y==0的样本
    2. 对X_train_list按样本数排序
    3. 对X_test_list按删除数排序
    4. 按照排序顺序一一对应，从X_train_list选取连续样本

    参数:
    X_test_list: list, 包含多个测试特征数组
    Y_test_list: list, 包含多个测试标签数组
    X_train_list: list, 包含多个训练特征数组（全为y==0）
    """

    # 检查输入
    assert len(X_test_list) == len(Y_test_list), "X_test_list和Y_test_list长度必须相同"

    # 步骤1: 删除Y_test中y==0的样本并计算删除数
    deletion_counts = []

    for i, (X_test, Y_test) in enumerate(zip(X_test_list, Y_test_list)):
        # 记录删除的样本数
        deleted_count = np.sum((Y_test == 0).flatten())
        deletion_counts.append(deleted_count)
    max_deletion_count = max(deletion_counts)

    # 步骤2: 对X_test_list按删除数排序（从大到小）
    print(f"\n根据删除数量对X_test_list({len(X_test_list)})进行排序（从大到小）")
    test_sorted_indices = sorted(range(len(deletion_counts)),
                                 key=lambda k: deletion_counts[k],
                                 reverse=True)

    # 步骤3: 对X_train_list排序（从小到大）
    print(f"采用{sort_mode}准则对X_train_list({len(X_train_list)})进行排序（从小到大）")
    if sort_mode == 'rd':
        train_sorted_indices = np.random.permutation(np.arange(len(X_train_list)))
    elif 'std' in sort_mode:
        normalized_X_train, _, _ = _dataset_fit_transform(X_train_list)
        train_std_sum = [np.mean(np.std(X_train[:max_deletion_count], axis=0)) for X_train in normalized_X_train]
        train_sorted_indices = sorted(range(len(train_std_sum)),
                                      key=lambda k: train_std_sum[k])
    elif 'cov' in sort_mode:
        train_cov_det = [np.linalg.det(np.cov(X_train[:max_deletion_count].T)) for X_train in X_train_list]
        train_sorted_indices = sorted(range(len(train_cov_det)),
                                      key=lambda k: train_cov_det[k])

    if swap_count > 0:
        # 随机选择要交换的索引
        swap_indices1 = random.sample(range(len(test_sorted_indices)), swap_count)
        swap_indices2 = random.sample(range(len(test_sorted_indices), len(train_sorted_indices)), swap_count)
        print(f"随机打乱train_sorted_indices({len(train_sorted_indices)})中的{swap_count}对: {swap_indices1} <-> {swap_indices2}")
        # 执行交换
        for i in range(len(swap_indices1)):
            idx1, idx2 = swap_indices1[i], swap_indices2[i]
            train_sorted_indices[idx1], train_sorted_indices[idx2] = (train_sorted_indices[idx2],
                                                                      train_sorted_indices[idx1])

    # 步骤4: 按排序顺序一一对应
    # 取两个列表中较小的长度作为对应数量
    num_pairs = len(X_test_list)

    # 创建映射：排序后的位置 -> 原始索引
    # test_mapping = {pos: test_sorted_indices[pos] for pos in range(num_pairs)}
    test_mapping, train_mapping = {}, {}
    cnt_pos2 = 0
    for pos in range(num_pairs):
        test_mapping[pos] = test_sorted_indices[pos]
        for pos2 in range(cnt_pos2, len(X_train_list)):
            train_id, test_id = train_sorted_indices[pos2], test_sorted_indices[pos]
            if X_train_list[train_id].shape[0] > deletion_counts[test_id]:
                train_mapping[pos] = train_id
                cnt_pos2 = pos2 + 1
                break

    # print(f"\n将创建 {num_pairs} 个对应关系:")
    for i in range(num_pairs):
        train_idx = train_mapping[i]  # 训练数组的原始索引
        test_idx = test_mapping[i]  # 测试数组的原始索引

        # 需要选取的样本数
        n_samples_needed = deletion_counts[test_idx]

        # 获取对应的训练数据
        X_train, Y_train, X_test, Y_test = (X_train_list[train_idx].copy(), Y_train_list[train_idx].copy(),
                                            X_test_list[test_idx].copy(), Y_test_list[test_idx].copy())

        # print(f"  对应 {i}:")
        # print(f"    使用训练数组 {train_idx} (样本数: {len(X_train)})")
        # print(f"    对应测试数组 {test_idx} (删除数: {n_samples_needed})")
        # print(f"    从训练数组位置 0 开始选取 {n_samples_needed} 个连续样本")

        # 检查是否有足够的样本
        # if n_samples_needed < len(X_train) and n_samples_needed >0:
        selected_idx = np.arange(n_samples_needed)
        remain_idx = np.arange(n_samples_needed,len(X_train))

        # 从训练数据前端选取连续样本，保存剩下的样本
        X_train_list[train_idx] = X_train[remain_idx]
        Y_train_list[train_idx] = Y_train[remain_idx]

        # 替换测试集中的部分样本
        X_test_list[test_idx] = np.concatenate([X_train[selected_idx], X_test[Y_test!=0]], axis=0)
        Y_test_list[test_idx] = np.concatenate([Y_train[selected_idx], Y_test[Y_test!=0]], axis=0)

    return X_train_list, Y_train_list, X_test_list, Y_test_list

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)

    # 示例数据段1
    X1 = np.random.randn(100, 5)  # 100个样本，5个特征
    Y1 = np.zeros(100)
    Y1[40:60] = 1  # 中间20个样本为故障
    Y1[80:90] = 2  # 另外10个样本为另一种故障

    # 示例数据段2
    X2 = np.random.randn(80, 5)
    Y2 = np.zeros(80)
    Y2[20:30] = 1
    Y2[50:65] = 3

    X_list = [X1, X2]
    Y_list = [Y1, Y2]

    # 划分数据集
    X_train_list, Y_train_list, X_test_list, Y_test_list = split_dataset(
        X_list, Y_list,
        min_time_interval=5,
        target_ratio=0.7,
        alternate_mode=True,
        cycle_length=10
    )

    # 验证结果结构
    print(f"\n结果结构验证:")
    print(f"原始数据段数: {len(X_list)}")
    print(f"训练集段数: {len(X_train_list)}")
    print(f"测试集段数: {len(X_test_list)}")

    for i in range(len(X_list)):
        print(
            f"段 {i + 1}: 原始大小={len(X_list[i])}, 训练集大小={len(X_train_list[i])}, 测试集大小={len(X_test_list[i])}")