import numpy as np
from collections import defaultdict


def reorganize_fault_dataset(test_X, test_Y, normal_first=True, verbose=False):
    """重组数据集为多个(正常样本+故障类别i)数据集，返回两个列表test_X, test_Y

    参数:
        test_X: 测试集特征数据
        test_Y: 测试集标签数据
        normal_first: 是否将正常样本放在数据集前面，默认为False（保持原始时间顺序）
        verbose: 是否打印详细日志
    """

    def log(msg):
        if verbose:
            print(msg)

    # 首先将 test_X 和 test_Y 转换为单个数组
    if isinstance(test_X, list):
        X_combined = np.concatenate(test_X, axis=0)
        n_total = len(X_combined)
    else:
        X_combined = test_X
        n_total = len(test_X)

    log(f"数据总长度: {n_total}")
    log(f"正常样本放置顺序: {'正常样本在前' if normal_first else '保持时间顺序'}")

    # 处理标签数据 - 统一转换为整数标签
    processed_test_Y = []
    for y in test_Y:
        if isinstance(y, np.ndarray) and y.ndim > 1:
            # 如果是 one-hot 编码，转换为整数
            processed_test_Y.append(np.argmax(y, axis=-1))
        else:
            # 如果是整数标签，直接使用
            processed_test_Y.append(y)

    # 确保所有标签都是整数
    Y_combined = np.concatenate(processed_test_Y, dtype=int)
    unique_labels = np.unique(Y_combined)
    log(f"标签值范围: {unique_labels}")

    # 找到所有类别切换的位置
    change_points = np.where(Y_combined[:-1] != Y_combined[1:])[0] + 1
    change_points = np.concatenate(([0], change_points, [len(Y_combined)]))
    log(f"变化点数量: {len(change_points)}")

    # 提取所有连续段
    segments = []
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i + 1]
        label = Y_combined[start]
        # 确保索引不超出范围
        end = min(end, n_total)
        # 确保开始索引小于结束索引
        if start < end:
            segments.append({
                'start': start,
                'end': end,
                'label': label,
                'indices': list(range(start, end)),
                'length': end - start
            })

    log(f"总段数: {len(segments)}")

    # 分离正常段和故障段
    normal_segments = [s for s in segments if s['label'] == 0]
    fault_segments_by_class = defaultdict(list)
    for s in segments:
        if s['label'] != 0:
            fault_segments_by_class[s['label']].append(s)

    # 如果没有正常段，直接返回空结果
    if not normal_segments:
        log("警告: 没有找到正常段!")
        return [], []

    # 计算总正常样本数
    total_normal = sum(s['length'] for s in normal_segments)
    fault_classes = sorted(fault_segments_by_class.keys())

    if not fault_classes:
        log("警告: 没有找到故障段!")
        return [], []

    # 每个故障类别分配的正常样本数（平均分配）
    n_normal_per_class = total_normal // len(fault_classes)
    remainder = total_normal % len(fault_classes)  # 余数

    # 计算每个故障类别应该分配的正常样本数
    normal_allocations = {}
    for i, fault_class in enumerate(fault_classes):
        # 前remainder个故障类别多分配1个样本
        if i < remainder:
            normal_allocations[fault_class] = n_normal_per_class + 1
        else:
            normal_allocations[fault_class] = n_normal_per_class

    log(f"总正常样本数: {total_normal}")
    log(f"故障类别数: {len(fault_classes)}")
    log(f"每个故障类别分配的正常样本数: {normal_allocations}")

    # 为每个正常段创建可用样本跟踪
    normal_segment_availability = {i: seg['length'] for i, seg in enumerate(normal_segments)}

    # 为每个故障类别分配正常样本
    allocation_plan = defaultdict(list)

    # 改进的分配策略：优先从较长的正常段中抽取更多样本
    for fault_class in fault_classes:
        fault_segments = fault_segments_by_class[fault_class]
        log(f"\n为故障类别 {fault_class} 分配正常样本:")
        log(f"  故障段数: {len(fault_segments)}")

        normal_needed = normal_allocations[fault_class]
        current_allocated = 0

        # 为每个故障段找到最近的正常段，并计算权重
        fault_normal_pairs = []
        for fault_segment in fault_segments:
            # 找到故障段前的正常段（按距离排序）
            preceding_normals = []
            for i, normal_seg in enumerate(normal_segments):
                if normal_seg['end'] <= fault_segment['start'] and normal_segment_availability[i] > 0:
                    distance = fault_segment['start'] - normal_seg['end']  # 距离越小越近
                    preceding_normals.append((i, normal_seg, distance))

            # 按距离排序，最近的优先
            preceding_normals.sort(key=lambda x: x[2])

            if preceding_normals:
                # 只考虑最近的3个正常段
                for normal_idx, normal_seg, distance in preceding_normals[:3]:
                    # 计算权重：距离越近、正常段越长，权重越高
                    weight = normal_seg['length'] / (distance + 1)  # 避免除以0
                    fault_normal_pairs.append((fault_segment, normal_seg, normal_idx, weight))

        # 按权重排序，权重高的优先分配
        fault_normal_pairs.sort(key=lambda x: x[3], reverse=True)

        # 分配样本
        for fault_segment, normal_seg, normal_idx, weight in fault_normal_pairs:
            if current_allocated >= normal_needed:
                break

            available = normal_segment_availability[normal_idx]
            if available > 0:
                # 计算这个分配应该抽取的样本数
                # 优先从较长的正常段中抽取更多样本
                min_samples = min(10, available)  # 最少抽取10个样本，避免太分散
                max_samples = min(normal_needed - current_allocated, available)

                # 根据权重调整抽取样本数
                n_to_take = min(max(min_samples, int(weight * 5)), max_samples)

                if n_to_take > 0:
                    allocation_plan[fault_class].append({
                        'fault_segment': fault_segment,
                        'normal_segment': normal_seg,
                        'normal_segment_idx': normal_idx,
                        'n_to_take': n_to_take
                    })

                    normal_segment_availability[normal_idx] -= n_to_take
                    current_allocated += n_to_take

                    log(f"    为故障段 [{fault_segment['start']}, {fault_segment['end']}] 分配正常段 [{normal_seg['start']}, {normal_seg['end']}] 的 {n_to_take} 个样本")

    # 第二轮分配：检查每个故障类别的分配情况，补充不足的部分
    for fault_class in fault_classes:
        fault_segments = fault_segments_by_class[fault_class]
        current_allocated = sum(plan['n_to_take'] for plan in allocation_plan[fault_class])
        remaining_needed = normal_allocations[fault_class] - current_allocated

        if remaining_needed > 0:
            log(f"  故障类别 {fault_class} 还需要补充 {remaining_needed} 个正常样本")

            # 找到所有还有剩余样本的正常段，按剩余样本数排序
            available_normals = [(i, seg, normal_segment_availability[i])
                                 for i, seg in enumerate(normal_segments)
                                 if normal_segment_availability[i] > 0]

            # 按剩余样本数排序，样本多的优先
            available_normals.sort(key=lambda x: x[2], reverse=True)

            # 优先从剩余样本多的正常段中抽取
            for normal_idx, normal_seg, available in available_normals:
                if remaining_needed <= 0:
                    break

                if available > 0:
                    # 从剩余样本多的正常段中抽取更多样本
                    n_to_take = min(remaining_needed, available)

                    # 选择一个故障段来关联这个正常段（选择第一个故障段）
                    fault_segment = fault_segments[0]

                    # 检查是否已经为这个故障段分配了这个正常段
                    existing_plan = next((p for p in allocation_plan[fault_class]
                                          if p['fault_segment']['start'] == fault_segment['start'] and
                                          p['normal_segment']['start'] == normal_seg['start']), None)

                    if existing_plan:
                        existing_plan['n_to_take'] += n_to_take
                    else:
                        allocation_plan[fault_class].append({
                            'fault_segment': fault_segment,
                            'normal_segment': normal_seg,
                            'normal_segment_idx': normal_idx,
                            'n_to_take': n_to_take
                        })

                    normal_segment_availability[normal_idx] -= n_to_take
                    remaining_needed -= n_to_take

                    log(f"    补充: 为故障段 [{fault_segment['start']}, {fault_segment['end']}] 分配正常段 [{normal_seg['start']}, {normal_seg['end']}] 的 {n_to_take} 个样本")

    # 检查分配结果和剩余正常样本
    log(f"\n分配结果检查:")
    total_allocated = 0
    for fault_class in fault_classes:
        class_allocated = sum(plan['n_to_take'] for plan in allocation_plan[fault_class])
        total_allocated += class_allocated
        log(f"  故障类别 {fault_class}: 分配了 {class_allocated} 个正常样本 (目标: {normal_allocations[fault_class]})")

    remaining_normal = sum(normal_segment_availability.values())
    log(f"  总分配正常样本: {total_allocated}, 剩余正常样本: {remaining_normal}")

    # 如果还有剩余正常样本，分配给前几个故障类别
    if remaining_normal > 0:
        log(f"  将剩余 {remaining_normal} 个正常样本分配给前几个故障类别")
        for i, fault_class in enumerate(fault_classes):
            if remaining_normal <= 0:
                break

            # 找到这个故障类别的故障段
            fault_segments = fault_segments_by_class[fault_class]
            if not fault_segments:
                continue

            # 找到还有剩余样本的正常段，按剩余样本数排序
            available_normals = [(i, seg, normal_segment_availability[i])
                                 for i, seg in enumerate(normal_segments)
                                 if normal_segment_availability[i] > 0]

            if not available_normals:
                break

            # 按剩余样本数排序，样本多的优先
            available_normals.sort(key=lambda x: x[2], reverse=True)

            # 取第一个正常段
            normal_idx, normal_seg, available = available_normals[0]
            n_to_take = min(remaining_normal, available)

            # 选择第一个故障段
            fault_segment = fault_segments[0]

            # 添加到分配计划
            allocation_plan[fault_class].append({
                'fault_segment': fault_segment,
                'normal_segment': normal_seg,
                'normal_segment_idx': normal_idx,
                'n_to_take': n_to_take
            })

            normal_segment_availability[normal_idx] -= n_to_take
            remaining_normal -= n_to_take

            log(f"    为故障类别 {fault_class} 额外分配正常段 [{normal_seg['start']}, {normal_seg['end']}] 的 {n_to_take} 个样本")

    # 创建重组的数据集
    test_X_list = []
    test_Y_list = []

    for fault_class in fault_classes:
        plans = allocation_plan[fault_class]

        if not plans:
            log(f"警告: 故障类别 {fault_class} 没有分配到正常样本，跳过")
            continue

        selected_normal_indices = []

        log(f"\n创建故障类别 {fault_class} 的数据集:")

        # 收集所有分配的正常样本
        for plan in plans:
            normal_seg = plan['normal_segment']
            n_to_take = plan['n_to_take']

            # 从正常段的末尾取样本（时间上最近的）
            start_idx = max(0, len(normal_seg['indices']) - n_to_take)
            actual_indices = normal_seg['indices'][start_idx:start_idx + n_to_take]

            selected_normal_indices.extend(actual_indices)

            log(f"  从正常段 [{normal_seg['start']}, {normal_seg['end']}] 取 {len(actual_indices)} 个样本")

        # 收集该故障类别的所有故障样本
        fault_indices = []
        for seg in fault_segments_by_class[fault_class]:
            fault_indices.extend(seg['indices'])

        # 根据参数决定样本顺序
        if normal_first:
            # 将正常样本放在前面，故障样本放在后面
            combined_indices = selected_normal_indices + fault_indices
            log(f"  样本顺序: 正常样本在前 ({len(selected_normal_indices)}个) + 故障样本在后 ({len(fault_indices)}个)")
        else:
            # 保持原始时间顺序
            combined_indices = sorted(selected_normal_indices + fault_indices)
            log(f"  样本顺序: 保持时间顺序")

        # 创建数据集 - 直接从合并的数组中提取
        X_samples = X_combined[combined_indices]

        # 保持原始的多分类标签
        Y_samples = Y_combined[combined_indices]

        # 转换为 one-hot 编码，保持9分类
        num_classes = len(unique_labels)  # 包括0-8共9个类别
        Y_onehot = np.eye(num_classes)[Y_samples]

        test_X_list.append(X_samples)
        test_Y_list.append(Y_onehot)

        log(f"  最终数据集: {len(X_samples)} 个样本 (正常: {len(selected_normal_indices)}, 故障: {len(fault_indices)})")

        # 计算标签分布
        unique, counts = np.unique(Y_samples, return_counts=True)
        label_dist = dict(zip(unique, counts))
        log(f"  标签分布: {label_dist}")

    return test_X_list, test_Y_list

def create_sample_dataset():
    """创建一个示例数据集用于测试"""
    # 创建模拟数据
    n_samples = 1000
    test_X = [np.random.randn(10) for _ in range(n_samples)]  # 10维特征
    test_Y = [0] * n_samples  # 初始化为全正常

    # 添加故障段
    # 故障类别1: 样本200-299
    for i in range(200, 300):
        test_Y[i] = 1

    # 故障类别2: 样本400-449
    for i in range(400, 450):
        test_Y[i] = 2

    # 故障类别3: 样本600-699
    for i in range(600, 700):
        test_Y[i] = 3

    # 故障类别1: 样本800-849
    for i in range(800, 850):
        test_Y[i] = 1

    return test_X, test_Y


def analyze_dataset(X, Y, title="数据集"):
    """分析数据集并打印统计信息"""
    Y_array = np.array(Y)
    unique, counts = np.unique(Y_array, return_counts=True)

    print(f"\n{title}统计:")
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count} 个样本")

    print(f"  总样本数: {len(Y)}")

    # 找到类别切换点
    change_points = np.where(Y_array[:-1] != Y_array[1:])[0] + 1
    change_points = np.concatenate(([0], change_points, [len(Y_array)]))

    segments = []
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i + 1]
        label = Y_array[start]
        segments.append({
            'start': start,
            'end': end,
            'label': label,
            'length': end - start
        })

    print(f"  连续段数: {len(segments)}")
    for seg in segments[:5]:  # 只显示前5个段，避免输出太长
        print(f"    段 [{seg['start']}, {seg['end']}): 类别 {seg['label']}, 长度 {seg['length']}")
    if len(segments) > 5:
        print(f"    ... 还有 {len(segments) - 5} 个段")


def analyze_datasets(test_X_list, test_Y_list, title="数据集"):
    """分析多个数据集并打印统计信息"""
    print(f"\n{title}统计:")

    for i, (X, Y) in enumerate(zip(test_X_list, test_Y_list)):
        Y_array = np.array(Y)
        unique, counts = np.unique(Y_array, return_counts=True)

        print(f"  故障类别 {i + 1}:")
        for label, count in zip(unique, counts):
            print(f"    类别 {label}: {count} 个样本")

        print(f"    总样本数: {len(Y)}")

        # 找到类别切换点
        change_points = np.where(Y_array[:-1] != Y_array[1:])[0] + 1
        change_points = np.concatenate(([0], change_points, [len(Y_array)]))

        segments = []
        for j in range(len(change_points) - 1):
            start, end = change_points[j], change_points[j + 1]
            label = Y_array[start]
            segments.append({
                'start': start,
                'end': end,
                'label': label,
                'length': end - start
            })

        print(f"    连续段数: {len(segments)}")
        for seg in segments[:3]:  # 只显示前3个段，避免输出太长
            print(f"      段 [{seg['start']}, {seg['end']}): 类别 {seg['label']}, 长度 {seg['length']}")
        if len(segments) > 3:
            print(f"      ... 还有 {len(segments) - 3} 个段")


def check_duplicate_samples(X):
    """检查数据集中是否有重复样本"""
    # 将样本转换为元组以便比较
    sample_tuples = [tuple(x.flatten()) for x in X]
    unique_samples = set(sample_tuples)
    return len(sample_tuples) != len(unique_samples)


# 完整的验证代码
if __name__ == "__main__":
    # 创建示例数据集
    print("创建示例数据集...")
    test_X, test_Y = create_sample_dataset()

    # 分析原始数据集
    analyze_dataset(test_X, test_Y, "原始数据集")

    # 应用重组函数
    print("\n应用重组函数...")
    test_X_list, test_Y_list = reorganize_fault_dataset(test_X, test_Y, verbose=True)

    # 分析重组后的数据集
    analyze_datasets(test_X_list, test_Y_list, "重组后数据集")

    # 验证样本数是否一致
    original_fault_samples = sum(1 for y in test_Y if y != 0)
    new_fault_samples = sum(sum(1 for y in Y_fault if y != 0) for Y_fault in test_Y_list)

    print(f"\n验证结果:")
    print(f"原始数据集故障样本数: {original_fault_samples}")
    print(f"重组后数据集故障样本数: {new_fault_samples}")
    print(f"故障样本数是否一致: {original_fault_samples == new_fault_samples}")

    # 检查每个数据集是否有重复样本
    for i, X_fault in enumerate(test_X_list):
        has_duplicates = check_duplicate_samples(X_fault)
        print(f"故障类别 {i + 1} 数据集是否有重复样本: {has_duplicates}")

    # 验证每个故障类别的样本数
    print("\n各故障类别样本数验证:")
    fault_classes = set(y for y in test_Y if y != 0)
    for fault_class in fault_classes:
        original_count = sum(1 for y in test_Y if y == fault_class)
        # 在重组后的数据集中找到对应的故障类别
        for i, Y_fault in enumerate(test_Y_list):
            if any(y == fault_class for y in Y_fault):
                new_count = sum(1 for y in Y_fault if y == fault_class)
                print(
                    f"故障类别 {fault_class}: 原始={original_count}, 重组后={new_count}, 一致={original_count == new_count}")
                break