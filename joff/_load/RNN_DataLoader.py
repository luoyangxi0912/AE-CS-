import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置中文字体和负号显示
# 根据操作系统设置中文字体
try:
    # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    print("已设置中文字体支持")
except:
    print("警告: 中文字体设置失败，将使用默认字体")

class RNNDataloader:
    """
    适用于RNN训练的时序连续DataLoader

    参数:
        X: 动态数据集列表，每个元素是一个连续段的动态样本数组 (n_dy_samples, stack * x_dim) * len_list
        L: 标签数据列表，每个元素是对应连续段的标签数组 (n_dy_samples, l_dim) * len_list
        stack: 滑动窗口长度
        batch_size: 批次大小
        shuffle_segments: 是否打乱不同段的顺序
        shuffle_samples: 是否打乱同一段内批次样本的顺序（保持批次结构）
        drop_last: 是否丢弃最后一个不完整的批次
        if_debug: 是否打印调试信息
        注意：
        1）dataloader的输入X中的一个动态样本为 (x_dim * stack)，对于(X, L) 中每一个分段的动态样本子集都能转化为连续的批次，即对于前后
        两个批次，它们中的每一个id相同的动态样本都是连续的，且一个批次的尺寸应该为 (batch_size, x_dim * stack)。
        2）RNN DataLoader中批次之间的连续性不是指的 batch1 的结尾连接 batch2的开头，而是指的 batch1中每一个动态样本（因为本身就是动态
        样本，长度为stack的序列）的时序末尾连接 batch2 中 每一个动态样本的开头，所以批次间可能有重叠（长度为batch_size-stack）是正常的。
    """

    def __init__(self, X: List[Union[np.ndarray, torch.Tensor]],
                 L: List[Union[np.ndarray, torch.Tensor]],
                 stack: int = None,
                 batch_size: int = None,    # 最好 <= stack（测试集必须=stack），不然会有重复的循环序列
                 shuffle_segments: bool = True,
                 shuffle_samples: bool = True,
                 drop_last: bool = True,
                 if_debug: bool = False):

        # 转换为torch张量
        self.X_list = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in X]
        self.L_list = [l if isinstance(l, torch.Tensor) else torch.tensor(l, dtype=torch.float32) for l in L]

        self.stack = stack
        self.batch_size = batch_size if batch_size is not None else stack
        self.shuffle_segments = shuffle_segments
        self.shuffle_samples = shuffle_samples
        self.drop_last = drop_last
        self.if_debug = if_debug
        self.name = 'RNNDataloader'

        # 验证输入
        if len(self.X_list) != len(self.L_list):
            raise ValueError(f"X和L的长度不一致: X有{len(self.X_list)}段, L有{len(self.L_list)}段")

        # 当前批次的状态信息
        self.current_segment_idx = -1
        self.reset_state = True

        # 记录原始分段信息
        self._record_original_segments()

        # 生成批次
        self._generate_batches()

        # 应用打乱
        if self.shuffle_segments:
            self._shuffle_segments()

        if self.shuffle_samples:
            self._shuffle_samples_within_segments()

        # 计算新的分割点位置
        self._compute_seg_len()

        # 验证生成结果
        self._validate_batches()

        # 初始化迭代计数器
        self.current_batch_idx = 0

        # 保存数据集
        X, L = [], []
        for (X_batch, L_batch) in self.all_batches:
            X.append(X_batch.numpy())
            L.append(L_batch.numpy())
        self.X, self.Y = np.concatenate(X), np.concatenate(L)
        # 求分段长度
        self.seg_index.insert(0,-1)
        self.seg_len = [(self.seg_index[i+1]-self.seg_index[i]) * self.batch_size for i in range(len(self.seg_index)-1) ]
        # self.seg_len = self.seg_index
        # print(self.seg_len)
        print(f"\nRNNDataLoader创建完成，总批次数: {len(self.all_batches)}，总样本数: {len(self.all_batches)*self.batch_size}")
        print(f"    其中子数据集样本数 (seg_len): {self.seg_len}")

    def _record_original_segments(self):
        """记录原始分段信息"""
        self.original_segments = []
        self.total_samples = 0

        for seg_idx, (X_seg, L_seg) in enumerate(zip(self.X_list, self.L_list)):
            if X_seg.shape[0] != L_seg.shape[0]:
                raise ValueError(f"段{seg_idx}: X和L的样本数不一致: X有{X_seg.shape[0]}, L有{L_seg.shape[0]}")

            seg_length = X_seg.shape[0]
            seg_start = self.total_samples
            seg_end = seg_start + seg_length - 1

            self.original_segments.append({
                'seg_idx': seg_idx,
                'start_idx': seg_start,
                'end_idx': seg_end,
                'length': seg_length,
                'X': X_seg,
                'L': L_seg
            })

            self.total_samples += seg_length

        if self.if_debug:
            print(f"原始分段信息:")
            for seg_info in self.original_segments:
                print(
                    f"  段{seg_info['seg_idx']}: 动态样本 [{seg_info['start_idx']}, {seg_info['end_idx']}], 长度={seg_info['length']}")

    def _generate_batches(self):
        """生成时序连续的批次"""
        self.all_batches = []  # 存储批次数据
        self.batch_segment_map = []  # 存储批次所属的原始段索引
        self.batch_start_indices = []  # 存储批次起始动态索引（在段内的局部索引）
        self.segment_batch_ranges = {}  # 存储每个原始段的批次范围
        self.segment_sample_shuffles = {}  # 存储每个原始段的样本打乱顺序

        # 统计每个原始段生成的批次数量
        self.original_segment_batch_counts = []

        for seg_info in self.original_segments:
            seg_idx = seg_info['seg_idx']
            X_seg = seg_info['X']
            L_seg = seg_info['L']
            seg_length = seg_info['length']

            # 计算段内可以生成的批次数量
            # 每个批次有batch_size个样本，每个样本是连续的
            # 批次间有重叠，重叠长度为stack
            if self.drop_last:
                # 只生成完整批次
                num_batches = (seg_length - self.batch_size) // self.stack + 1
                if num_batches <= 0:
                    if self.if_debug:
                        print(f"警告: 原始段{seg_idx}无法生成任何完整批次，跳过")
                    self.original_segment_batch_counts.append(0)
                    continue
            else:
                # 允许生成不完整批次
                if seg_length >= self.batch_size:
                    num_batches = (seg_length - self.batch_size) // self.stack + 1
                    # 如果有剩余样本，添加一个不完整批次
                    if (seg_length - self.batch_size) % self.stack != 0:
                        num_batches += 1
                else:
                    num_batches = 1  # 只有一个不完整批次

            if num_batches <= 0:
                if self.if_debug:
                    print(f"警告: 原始段{seg_idx}无法生成任何批次，跳过")
                self.original_segment_batch_counts.append(0)
                continue

            # 记录原始段内批次的起始索引
            start_batch_idx = len(self.all_batches)

            for i in range(num_batches):
                # 计算批次起始动态索引（在段内的局部索引）
                batch_start_local = i * self.stack

                # 计算批次实际大小
                if batch_start_local + self.batch_size <= seg_length:
                    # 完整批次
                    batch_end_local = batch_start_local + self.batch_size - 1
                    actual_batch_size = self.batch_size
                else:
                    # 不完整批次
                    batch_end_local = seg_length - 1
                    actual_batch_size = seg_length - batch_start_local
                    if self.drop_last:
                        break  # 跳过不完整批次

                # 收集批次数据
                X_batch = X_seg[batch_start_local:batch_end_local + 1]
                L_batch = L_seg[batch_start_local:batch_end_local + 1]

                # 验证批次大小
                if X_batch.shape[0] != actual_batch_size:
                    if self.if_debug:
                        print(f"错误: 段{seg_idx}批次{i}大小不正确，跳过")
                    continue

                # 存储批次
                self.all_batches.append((X_batch, L_batch))
                self.batch_segment_map.append(seg_idx)
                self.batch_start_indices.append(batch_start_local)

                if self.if_debug and i < 3:
                    print(
                        f"  原始段{seg_idx}批次{i}: 段内索引[{batch_start_local}, {batch_end_local}], 大小={actual_batch_size}")

            # 记录原始段的批次范围
            if num_batches > 0:
                end_batch_idx = len(self.all_batches) - 1
                self.segment_batch_ranges[seg_idx] = (start_batch_idx, end_batch_idx)
                # 初始化原始段的样本打乱顺序
                self.segment_sample_shuffles[seg_idx] = list(range(self.batch_size))
                # 记录原始段生成的批次数量
                self.original_segment_batch_counts.append(num_batches)

                if self.if_debug:
                    print(f"  原始段{seg_idx}: 生成 {num_batches} 个批次，批次范围 [{start_batch_idx}, {end_batch_idx}]")
            else:
                self.original_segment_batch_counts.append(0)

    def _compute_seg_len(self):
        """计算新的分割点位置列表（seg_index）"""
        # seg_index: 每个原始段结束的批次索引（最后一个批次的索引）
        self.seg_index = []
        cumulative_batches = 0

        for seg_idx, batch_count in enumerate(self.original_segment_batch_counts):
            if batch_count > 0:
                cumulative_batches += batch_count
                self.seg_index.append(cumulative_batches - 1)  # 该段最后一个批次的索引

        if self.if_debug and self.seg_index:
            print(f"\n新的分割点位置 (seg_index):")
            for i, seg_end_idx in enumerate(self.seg_index):
                if i == 0:
                    seg_start_idx = 0
                else:
                    seg_start_idx = self.seg_index[i - 1] + 1
                print(f"  段{i}: 批次范围 [{seg_start_idx}, {seg_end_idx}]")

    def _shuffle_segments(self):
        """打乱不同原始段的顺序（段间打乱）"""
        if self.if_debug:
            print("打乱原始段顺序...")

        # 按原始段分组批次
        segment_groups = {}
        for batch_idx, seg_idx in enumerate(self.batch_segment_map):
            if seg_idx not in segment_groups:
                segment_groups[seg_idx] = []
            segment_groups[seg_idx].append(batch_idx)

        # 打乱原始段顺序
        segment_indices = list(segment_groups.keys())
        random.shuffle(segment_indices)

        # 重新组织批次
        new_all_batches = []
        new_batch_segment_map = []
        new_batch_start_indices = []

        # 重要：我们需要保持原始段索引不变，因为这是用来定位原始数据的
        # 我们只是改变段的顺序，不改变段内的批次顺序
        for new_seg_order, old_seg_idx in enumerate(segment_indices):
            batch_indices = segment_groups[old_seg_idx]

            # 保持每个段内批次的相对顺序
            for batch_idx in batch_indices:
                X_batch, L_batch = self.all_batches[batch_idx]
                batch_start_in_segment = self.batch_start_indices[batch_idx]

                new_all_batches.append((X_batch, L_batch))
                # 重要：保持原始段索引，但使用新的顺序
                new_batch_segment_map.append(old_seg_idx)
                new_batch_start_indices.append(batch_start_in_segment)

        # 更新数据
        self.all_batches = new_all_batches
        self.batch_segment_map = new_batch_segment_map
        self.batch_start_indices = new_batch_start_indices

        # 重新计算段范围
        self._update_segment_ranges(new_batch_segment_map)

        # 重新计算seg_index
        self._recompute_seg_len_after_shuffle(new_batch_segment_map)

        if self.if_debug:
            print(f"打乱段顺序完成，新段顺序: {segment_indices}")

    def _recompute_seg_len_after_shuffle(self, batch_segment_map: List[int]):
        """打乱后重新计算seg_index（分割点位置）"""
        if not batch_segment_map:
            self.seg_index = []
            return

        # 统计每个新段（打乱后的段）的批次数量
        segment_batch_counts = {}
        for seg_idx in batch_segment_map:
            segment_batch_counts[seg_idx] = segment_batch_counts.get(seg_idx, 0) + 1

        # 按新段索引排序（0, 1, 2, ...）
        sorted_seg_indices = sorted(segment_batch_counts.keys())

        # 重新计算seg_index（每个段最后一个批次的索引）
        self.seg_index = []
        cumulative_batches = 0

        for seg_idx in sorted_seg_indices:
            batch_count = segment_batch_counts[seg_idx]
            cumulative_batches += batch_count
            self.seg_index.append(cumulative_batches - 1)  # 该段最后一个批次的索引

        if self.if_debug:
            print(f"打乱后的seg_index: {self.seg_index}")
            print(f"各段批次数: {segment_batch_counts}")

    def _shuffle_samples_within_segments(self):
        """打乱同一原始段内批次样本的顺序（所有批次应用相同的打乱顺序）"""
        if self.if_debug:
            print("打乱原始段内批次样本顺序...")

        # 对每个原始段独立处理
        for seg_idx in self.segment_sample_shuffles.keys():
            if seg_idx not in self.segment_batch_ranges:
                continue

            # 为该原始段生成一个随机排列
            permutation = list(range(self.batch_size))
            random.shuffle(permutation)

            # 保存这个排列
            self.segment_sample_shuffles[seg_idx] = permutation

            # 应用这个排列到该原始段的所有批次
            start_batch, end_batch = self.segment_batch_ranges[seg_idx]

            if self.if_debug:
                print(f"  原始段{seg_idx}: 批次范围[{start_batch}, {end_batch}], 打乱顺序: {permutation[:5]}...")

            for batch_idx in range(start_batch, end_batch + 1):
                X_batch, L_batch = self.all_batches[batch_idx]

                # 只对完整批次进行打乱
                if X_batch.shape[0] == self.batch_size:
                    # 应用排列
                    X_shuffled = X_batch[permutation]
                    L_shuffled = L_batch[permutation]

                    # 更新批次
                    self.all_batches[batch_idx] = (X_shuffled, L_shuffled)
                elif self.if_debug:
                    print(f"    警告: 批次{batch_idx}大小不正确，跳过打乱")

    def _update_segment_ranges(self, batch_segment_map: List[int]):
        """更新原始段批次范围映射"""
        self.segment_batch_ranges = {}

        if not batch_segment_map:
            return

        current_seg = batch_segment_map[0]
        start_idx = 0

        for batch_idx, seg_idx in enumerate(batch_segment_map):
            if seg_idx != current_seg:
                self.segment_batch_ranges[current_seg] = (start_idx, batch_idx - 1)
                current_seg = seg_idx
                start_idx = batch_idx

        # 处理最后一段
        self.segment_batch_ranges[current_seg] = (start_idx, len(batch_segment_map) - 1)

    def _validate_batches(self):
        """验证批次生成的正确性"""
        if not self.if_debug:
            return

        print(f"\n验证批次:")
        print(f"总批次数: {len(self.all_batches)}")

        if len(self.all_batches) == 0:
            print("警告: 没有生成任何批次!")
            return

        # 检查每个批次的大小
        incorrect_batches = []
        incomplete_batches = []

        for i, (X_batch, _) in enumerate(self.all_batches):
            if X_batch.shape[0] != self.batch_size:
                if self.drop_last:
                    incorrect_batches.append((i, X_batch.shape[0]))
                else:
                    incomplete_batches.append((i, X_batch.shape[0]))

        if incorrect_batches:
            print(f"错误: 发现 {len(incorrect_batches)} 个大小不正确的批次 (drop_last=True时不应出现)!")
            for batch_idx, size in incorrect_batches:
                print(f"  批次{batch_idx}: 大小={size}, 期望={self.batch_size}")

        if incomplete_batches and not self.drop_last:
            print(f"提示: 发现 {len(incomplete_batches)} 个不完整批次 (drop_last=False):")
            for batch_idx, size in incomplete_batches:
                print(f"  批次{batch_idx}: 大小={size}, 期望={self.batch_size}")

        if not incorrect_batches and not incomplete_batches:
            print("✓ 所有批次大小正确")
        elif not incorrect_batches and incomplete_batches:
            print("✓ 所有批次符合预期 (包含不完整批次)")

    def __len__(self):
        return len(self.all_batches)

    def __iter__(self):
        """实现迭代器接口"""
        self.current_batch_idx = 0
        return self

    def __next__(self):
        """迭代器下一个元素"""
        if self.current_batch_idx >= len(self.all_batches):
            raise StopIteration

        # 获取批次数据
        X_batch, L_batch = self.all_batches[self.current_batch_idx]
        segment_idx = self.batch_segment_map[self.current_batch_idx]

        # 验证批次大小
        if self.if_debug and X_batch.shape[0] != self.batch_size:
            if self.drop_last:
                print(
                    f"警告: 在__next__中批次{self.current_batch_idx}大小不正确: {X_batch.shape[0]} != {self.batch_size}")
            else:
                print(f"提示: 批次{self.current_batch_idx}是不完整批次: {X_batch.shape[0]} < {self.batch_size}")

        # 确定是否需要重置hidden state
        if self.current_batch_idx == 0:
            reset_state = True
        else:
            prev_segment = self.batch_segment_map[self.current_batch_idx - 1]
            reset_state = (segment_idx != prev_segment)

        # 更新状态属性
        self.current_segment_idx = segment_idx
        self.reset_state = reset_state

        # 更新迭代计数器
        self.current_batch_idx += 1

        return X_batch, L_batch

    def __getitem__(self, idx: int):
        """获取一个批次的数据"""
        if idx >= len(self.all_batches):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.all_batches) - 1}]")

        X_batch, L_batch = self.all_batches[idx]
        segment_idx = self.batch_segment_map[idx]

        # 验证批次大小
        if self.if_debug and X_batch.shape[0] != self.batch_size:
            if self.drop_last:
                print(f"警告: 在__getitem__中批次{idx}大小不正确: {X_batch.shape[0]} != {self.batch_size}")
            else:
                print(f"提示: 批次{idx}是不完整批次: {X_batch.shape[0]} < {self.batch_size}")

        # 确定是否需要重置hidden state
        if idx == 0:
            reset_state = True
        else:
            prev_segment = self.batch_segment_map[idx - 1]
            reset_state = (segment_idx != prev_segment)

        # 更新状态属性
        self.current_segment_idx = segment_idx
        self.reset_state = reset_state

        return X_batch, L_batch

    def get_batch_info(self, idx: int) -> Dict:
        """获取批次的详细信息"""
        if idx >= len(self.all_batches):
            raise IndexError(f"索引 {idx} 超出范围")

        X_batch, L_batch = self.all_batches[idx]
        segment_idx = self.batch_segment_map[idx]  # 这是原始段索引
        start_dyn_idx_in_segment = self.batch_start_indices[idx]
        actual_batch_size = X_batch.shape[0]

        # 获取该原始段的打乱顺序
        shuffle_perm = self.segment_sample_shuffles.get(segment_idx, list(range(self.batch_size)))

        # 查找原始段信息 - 使用正确的原始段索引
        original_seg_idx = segment_idx  # 这里应该是原始段索引
        seg_info = self.original_segments[original_seg_idx]
        seg_start_global = seg_info['start_idx']

        # 计算全局起始索引
        start_dyn_idx_global = seg_start_global + start_dyn_idx_in_segment

        # 计算原始样本范围
        original_ranges = []
        for i in range(actual_batch_size):
            # 应用打乱顺序（如果有）
            if self.shuffle_samples:
                # shuffle_perm[i] 是打乱后的位置i对应的原始位置
                orig_i = shuffle_perm[i] if i < len(shuffle_perm) else i
            else:
                orig_i = i

            dyn_idx_global = start_dyn_idx_global + orig_i
            original_start = dyn_idx_global
            original_end = dyn_idx_global
            original_ranges.append((original_start, original_end))

        return {
            'X': X_batch,
            'L': L_batch,
            'segment_idx': segment_idx,  # 原始段索引
            'original_segment_idx': original_seg_idx,  # 也是原始段索引
            'start_dyn_idx_in_segment': start_dyn_idx_in_segment,
            'start_dyn_idx_global': start_dyn_idx_global,
            'original_ranges': original_ranges,
            'shuffle_permutation': shuffle_perm,
            'actual_batch_size': actual_batch_size,
            'expected_batch_size': self.batch_size,
            'stack': self.stack
        }

    def get_segment_info(self, segment_idx: int) -> Optional[Dict]:
        """获取原始段的信息"""
        if segment_idx >= len(self.original_segments):
            return None

        seg_info = self.original_segments[segment_idx]

        # 获取该段的批次范围
        if segment_idx in self.segment_batch_ranges:
            start_batch, end_batch = self.segment_batch_ranges[segment_idx]
        else:
            start_batch, end_batch = -1, -1

        # 获取该段的打乱顺序
        shuffle_perm = self.segment_sample_shuffles.get(segment_idx, list(range(self.batch_size)))

        # 计算该段的批次信息
        batches_info = []
        if start_batch != -1 and end_batch != -1:
            for batch_idx in range(start_batch, end_batch + 1):
                batch_info = self.get_batch_info(batch_idx)
                batches_info.append({
                    'batch_idx': batch_idx,
                    'actual_size': batch_info['actual_batch_size'],
                    'start_dyn_idx': batch_info['start_dyn_idx']
                })

        return {
            'original_segment': seg_info,
            'batch_range': (start_batch, end_batch),
            'num_batches': end_batch - start_batch + 1 if start_batch != -1 else 0,
            'shuffle_permutation': shuffle_perm,
            'batches_info': batches_info
        }

    def get_batch_len(self) -> List[int]:
        """获取新的分割点位置列表"""
        return self.seg_index


def generate_mimo_data(
        n_segments: int = 3,
        min_seg_len: int = 100,
        max_seg_len: int = 200,
        u_dim: int = 2,
        y_dim: int = 3,
        stack: int = 5,
        noise_std: float = 0.1
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    生成MIMO系统仿真数据: x' = f(x,u), y = Cx

    参数:
        n_segments: 数据段数量
        min_seg_len: 每段最小长度
        max_seg_len: 每段最大长度
        u_dim: 输入维度
        y_dim: 输出维度
        stack: 堆叠长度
        noise_std: 测量噪声标准差

    返回:
        X_list: 动态数据列表 [u y]*stack
        L_list: 标签数据列表 [u_next y_next]
        full_data_list: 完整的原始数据列表 [u, y] 用于验证
    """
    np.random.seed(42)

    X_list = []
    L_list = []
    full_data_list = []

    for seg_idx in range(n_segments):
        # 随机生成本段长度
        seg_len = np.random.randint(min_seg_len, max_seg_len)

        # 生成输入序列 (时间序列)
        u = np.zeros((seg_len + 1, u_dim))  # +1 用于生成标签

        # 使用正弦和余弦信号作为输入，模拟控制信号
        for i in range(u_dim):
            freq = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.5, 2.0)
            u[:, i] = amplitude * np.sin(freq * np.arange(seg_len + 1) + phase)

        # 添加一些阶跃变化
        step_times = np.random.choice(seg_len // 2, size=2, replace=False)
        for t in step_times:
            u[t:, :] += np.random.uniform(-1, 1, size=u_dim)

        # 添加噪声
        u += np.random.normal(0, 0.05, size=u.shape)

        # 定义简单的MIMO系统动态
        # x' = A*x + B*u, y = C*x
        n_states = 4  # 内部状态维度
        A = np.array([[0.9, 0.1, -0.05, 0.02],
                      [0.05, 0.85, 0.1, -0.03],
                      [-0.02, 0.1, 0.8, 0.15],
                      [0.01, -0.05, 0.1, 0.75]])
        B = np.random.randn(n_states, u_dim) * 0.5
        C = np.random.randn(y_dim, n_states) * 0.8

        # 仿真系统
        x = np.zeros((seg_len + 1, n_states))
        x[0, :] = np.random.randn(n_states)

        for t in range(seg_len):
            x[t + 1, :] = A @ x[t, :] + B @ u[t, :] + np.random.normal(0, 0.01, size=n_states)

        # 生成输出 (测量值)
        y = np.zeros((seg_len + 1, y_dim))
        for t in range(seg_len + 1):
            y[t, :] = C @ x[t, :] + np.random.normal(0, noise_std, size=y_dim)

        # 构建动态样本 [u y]*stack
        dynamic_samples = []
        labels = []

        # 我们需要足够的样本来构建stack窗口
        for t in range(stack - 1, seg_len):
            # 构建stack窗口内的数据
            window_data = []
            for s in range(stack):
                # 获取t-stack+1+s时刻的数据
                idx = t - stack + 1 + s
                window_data.extend([u[idx, :], y[idx, :]])

            # 动态样本: 将窗口数据展平
            dynamic_sample = np.concatenate(window_data)  # 形状: [(u_dim+y_dim)*stack]
            dynamic_samples.append(dynamic_sample)

            # 标签: 下一时刻的u和y
            label = np.concatenate([u[t + 1, :], y[t + 1, :]])  # 形状: [u_dim+y_dim]
            labels.append(label)

        # 转换为numpy数组
        X_seg = np.array(dynamic_samples)  # [n_samples, (u_dim+y_dim)*stack]
        L_seg = np.array(labels)  # [n_samples, u_dim+y_dim]

        # 保存完整的原始数据用于验证
        full_data = {
            'u': u[:-1, :],  # 去掉最后一个，因为标签用的是下一时刻
            'y': y[:-1, :],
            'X_original': X_seg.copy(),
            'L_original': L_seg.copy()
        }

        X_list.append(X_seg)
        L_list.append(L_seg)
        full_data_list.append(full_data)

        print(f"段{seg_idx}: 长度={seg_len}, 样本数={X_seg.shape[0]}, X形状={X_seg.shape}, L形状={L_seg.shape}")

    return X_list, L_list, full_data_list


def test_rnn_dataloader_accuracy(
        dataloader: RNNDataloader,
        X_list: List[np.ndarray],
        L_list: List[np.ndarray],
        full_data_list: List[Dict],
        test_name: str = "Test",
        plot_results: bool = True
) -> Dict:
    """
    测试RNNDataloader的准确性

    参数:
        dataloader: 要测试的RNNDataloader实例
        X_list: 原始输入的X数据列表
        L_list: 原始输入的L数据列表
        full_data_list: 完整的原始数据列表（用于验证）
        test_name: 测试名称
        plot_results: 是否绘制结果

    返回:
        包含测试结果的字典
    """

    print(f"\n{'=' * 60}")
    print(f"开始测试: {test_name}")
    print(f"shuffle_segments={dataloader.shuffle_segments}, shuffle_samples={dataloader.shuffle_samples}")
    print(f"{'=' * 60}")

    # 1. 构建全局原始数据数组以便索引
    print("\n1. 构建全局原始数据数组...")

    # 将所有段的X和L连接成全局数组
    X_global = np.concatenate([x for x in X_list], axis=0)
    L_global = np.concatenate([l for l in L_list], axis=0)

    print(f"   全局X形状: {X_global.shape}")
    print(f"   全局L形状: {L_global.shape}")

    # 计算每个原始段的起始和结束索引
    segment_boundaries = []
    start_idx = 0
    for x_seg in X_list:
        end_idx = start_idx + x_seg.shape[0] - 1
        segment_boundaries.append((start_idx, end_idx))
        start_idx = end_idx + 1

    print(f"   原始段边界: {segment_boundaries}")

    # 2. 遍历dataloader中的所有批次，验证准确性
    print("\n2. 遍历并验证所有批次...")

    # 存储误差信息
    X_errors = []
    L_errors = []
    batch_infos = []
    segment_errors = {}

    # 用于验证连续性的变量
    prev_batch_info = None
    continuity_errors = 0

    for batch_idx in range(len(dataloader)):
        # 获取批次数据和信息
        X_batch, L_batch = dataloader[batch_idx]
        batch_info = dataloader.get_batch_info(batch_idx)

        # 重要：对于段间打乱，我们需要从batch_info中获取正确的原始段索引
        # batch_info['original_segment_idx'] 应该包含正确的原始段索引
        original_segment_idx = batch_info['original_segment_idx']
        segment_idx = batch_info['segment_idx']

        batch_info_dict = {
            'batch_idx': batch_idx,
            'segment_idx': segment_idx,
            'original_segment_idx': original_segment_idx,
            'start_dyn_idx_global': batch_info['start_dyn_idx_global'],
            'actual_batch_size': batch_info['actual_batch_size'],
            'shuffle_perm': batch_info['shuffle_permutation']
        }
        batch_infos.append(batch_info_dict)

        # 对于批次中的每个样本，验证其准确性
        batch_X_error = 0
        batch_L_error = 0
        valid_samples = 0

        for sample_idx in range(batch_info['actual_batch_size']):
            # 获取当前样本在批次中的索引（考虑打乱）
            if dataloader.shuffle_samples and batch_info['shuffle_permutation']:
                # 如果有打乱，需要找到原始顺序中的位置
                # 注意：shuffle_permutation 是打乱后的顺序，要找到原始位置
                # 如果 permutation = [2, 0, 1]，那么：
                # - 打乱后的位置0对应原始位置2
                # - 打乱后的位置1对应原始位置0
                # - 打乱后的位置2对应原始位置1
                orig_sample_idx_in_batch = batch_info['shuffle_permutation'][sample_idx]
            else:
                orig_sample_idx_in_batch = sample_idx

            # 计算该样本在全局原始数据中的索引
            global_idx = batch_info['start_dyn_idx_global'] + orig_sample_idx_in_batch

            # 验证索引是否在有效范围内
            if global_idx >= X_global.shape[0]:
                print(f"   警告: 批次{batch_idx}样本{sample_idx}的全局索引{global_idx}超出范围!")
                continue

            # 从全局原始数据中获取对应样本
            X_original = X_global[global_idx]
            L_original = L_global[global_idx]

            # 获取dataloader中的样本
            X_dataloader = X_batch[sample_idx].numpy()
            L_dataloader = L_batch[sample_idx].numpy()

            # 计算误差
            X_error = np.mean(np.abs(X_dataloader - X_original))
            L_error = np.mean(np.abs(L_dataloader - L_original))

            batch_X_error += X_error
            batch_L_error += L_error
            valid_samples += 1

            # 如果误差太大，打印详细信息
            if X_error > 1e-6 or L_error > 1e-6:
                print(f"   批次{batch_idx}样本{sample_idx}: X误差={X_error:.6e}, L误差={L_error:.6e}")
                print(f"     全局索引: {global_idx}, 段内偏移: {orig_sample_idx_in_batch}")

        if valid_samples > 0:
            avg_X_error = batch_X_error / valid_samples
            avg_L_error = batch_L_error / valid_samples
            X_errors.append(avg_X_error)
            L_errors.append(avg_L_error)

            # 按原始段统计误差
            if original_segment_idx not in segment_errors:
                segment_errors[original_segment_idx] = {'X': [], 'L': []}
            segment_errors[original_segment_idx]['X'].append(avg_X_error)
            segment_errors[original_segment_idx]['L'].append(avg_L_error)

        # 验证连续性（仅当不打乱时）
        if not dataloader.shuffle_samples and not dataloader.shuffle_segments:
            if prev_batch_info is not None:
                # 检查是否在同一段中
                if (batch_info['original_segment_idx'] == prev_batch_info['original_segment_idx'] and
                        prev_batch_info['start_dyn_idx_global'] + prev_batch_info['actual_batch_size'] !=
                        batch_info['start_dyn_idx_global']):

                    expected_start = prev_batch_info['start_dyn_idx_global'] + dataloader.stack
                    actual_start = batch_info['start_dyn_idx_global']

                    if expected_start != actual_start:
                        continuity_errors += 1
                        print(f"   连续性错误: 批次{prev_batch_info['batch_idx']} -> 批次{batch_idx}")
                        print(f"     预期起始索引: {expected_start}, 实际起始索引: {actual_start}")

        prev_batch_info = batch_info_dict

    # 3. 计算统计信息
    print("\n3. 计算统计信息...")

    X_errors_array = np.array(X_errors) if X_errors else np.array([])
    L_errors_array = np.array(L_errors) if L_errors else np.array([])

    stats = {
        'total_batches': len(dataloader),
        'X_errors': X_errors,
        'L_errors': L_errors,
        'avg_X_error': np.mean(X_errors_array) if len(X_errors_array) > 0 else 0,
        'std_X_error': np.std(X_errors_array) if len(X_errors_array) > 0 else 0,
        'max_X_error': np.max(X_errors_array) if len(X_errors_array) > 0 else 0,
        'min_X_error': np.min(X_errors_array) if len(X_errors_array) > 0 else 0,
        'avg_L_error': np.mean(L_errors_array) if len(L_errors_array) > 0 else 0,
        'std_L_error': np.std(L_errors_array) if len(L_errors_array) > 0 else 0,
        'max_L_error': np.max(L_errors_array) if len(L_errors_array) > 0 else 0,
        'min_L_error': np.min(L_errors_array) if len(L_errors_array) > 0 else 0,
        'continuity_errors': continuity_errors,
        'segment_errors': segment_errors,
        'batch_infos': batch_infos
    }

    # 4. 打印统计信息
    print(f"\n测试 '{test_name}' 结果:")
    print(f"   总批次数: {stats['total_batches']}")
    if len(X_errors_array) > 0:
        print(f"   X数据误差 - 均值: {stats['avg_X_error']:.6e}, 标准差: {stats['std_X_error']:.6e}")
        print(f"   X数据误差 - 最大值: {stats['max_X_error']:.6e}, 最小值: {stats['min_X_error']:.6e}")
    if len(L_errors_array) > 0:
        print(f"   L数据误差 - 均值: {stats['avg_L_error']:.6e}, 标准差: {stats['std_L_error']:.6e}")
        print(f"   L数据误差 - 最大值: {stats['max_L_error']:.6e}, 最小值: {stats['min_L_error']:.6e}")

    if not dataloader.shuffle_samples and not dataloader.shuffle_segments:
        print(f"   连续性错误数: {stats['continuity_errors']}")

    # 5. 绘制结果（如果要求）
    if plot_results and len(X_errors) > 0 and len(L_errors) > 0:
        _plot_test_results(stats, dataloader, test_name)

    # 6. 检查是否所有原始数据都被覆盖
    print("\n4. 检查数据覆盖情况...")
    coverage_info = _check_data_coverage(dataloader, X_global, batch_infos)
    stats.update(coverage_info)

    return stats


def _plot_test_results(stats: Dict, dataloader: RNNDataloader, test_name: str):
    """绘制测试结果"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'RNN DataLoader 测试结果: {test_name}', fontsize=16)

    # 1. X和L误差曲线
    if 'X_errors' in stats and 'L_errors' in stats and stats['X_errors'] and stats['L_errors']:
        batch_indices = list(range(len(stats['X_errors'])))
        axes[0, 0].plot(batch_indices, stats['X_errors'], marker='o', linestyle='-', alpha=0.7, label='X误差')
        axes[0, 0].plot(batch_indices, stats['L_errors'], marker='s', linestyle='--', alpha=0.7, label='L误差')
        axes[0, 0].set_xlabel('批次索引')
        axes[0, 0].set_ylabel('批次误差')
        axes[0, 0].set_title('批次误差曲线')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, '无误差数据', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('批次误差曲线')

    # 2. X误差直方图
    if 'X_errors' in stats and stats['X_errors']:
        axes[0, 1].hist(stats['X_errors'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].axvline(stats['avg_X_error'], color='red', linestyle='--', label=f'均值: {stats["avg_X_error"]:.2e}')
        axes[0, 1].set_xlabel('X数据误差')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('X数据误差分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, '无X误差数据', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('X数据误差分布')

    # 3. L误差直方图
    if 'L_errors' in stats and stats['L_errors']:
        axes[0, 2].hist(stats['L_errors'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].axvline(stats['avg_L_error'], color='red', linestyle='--', label=f'均值: {stats["avg_L_error"]:.2e}')
        axes[0, 2].set_xlabel('L数据误差')
        axes[0, 2].set_ylabel('频次')
        axes[0, 2].set_title('L数据误差分布')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, '无L误差数据', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('L数据误差分布')

    # 4. 按段统计的误差
    if 'segment_errors' in stats and stats['segment_errors']:
        seg_indices = list(stats['segment_errors'].keys())
        seg_X_errors = [np.mean(stats['segment_errors'][i]['X']) for i in seg_indices]
        seg_L_errors = [np.mean(stats['segment_errors'][i]['L']) for i in seg_indices]

        x_pos = np.arange(len(seg_indices))
        width = 0.35

        axes[1, 0].bar(x_pos - width / 2, seg_X_errors, width, label='X误差', alpha=0.7, color='blue')
        axes[1, 0].bar(x_pos + width / 2, seg_L_errors, width, label='L误差', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('段索引')
        axes[1, 0].set_ylabel('平均误差')
        axes[1, 0].set_title('各段平均误差')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f'段{i}' for i in seg_indices])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, '无段误差数据', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('各段平均误差')

    # 5. 批次大小分布
    if 'batch_infos' in stats and stats['batch_infos']:
        batch_indices = list(range(len(stats['batch_infos'])))
        batch_sizes = [info['actual_batch_size'] for info in stats['batch_infos']]
        axes[1, 1].plot(batch_indices, batch_sizes, marker='o', linestyle='-', alpha=0.7)
        axes[1, 1].axhline(dataloader.batch_size, color='red', linestyle='--',
                           label=f'目标大小: {dataloader.batch_size}')
        axes[1, 1].set_xlabel('批次索引')
        axes[1, 1].set_ylabel('批次大小')
        axes[1, 1].set_title('批次大小分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '无批次信息', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('批次大小分布')

    # 6. 段索引分布
    if 'batch_infos' in stats and stats['batch_infos']:
        segment_indices = [info['segment_idx'] for info in stats['batch_infos']]
        unique_segments = list(set(segment_indices))
        segment_counts = [segment_indices.count(seg) for seg in unique_segments]

        axes[1, 2].bar(unique_segments, segment_counts, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 2].set_xlabel('段索引')
        axes[1, 2].set_ylabel('批次数量')
        axes[1, 2].set_title('各段批次数量分布')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, '无段索引数据', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('各段批次数量分布')

    plt.tight_layout()
    plt.show()


def _check_data_coverage(dataloader: RNNDataloader, X_global: np.ndarray, batch_infos: List[Dict]):
    """检查数据覆盖情况"""

    total_samples = X_global.shape[0]
    coverage = np.zeros(total_samples, dtype=int)

    for batch_info in batch_infos:
        start_idx = batch_info['start_dyn_idx_global']
        batch_size = batch_info['actual_batch_size']
        end_idx = min(start_idx + batch_size, total_samples)

        for idx in range(start_idx, end_idx):
            if idx < len(coverage):
                coverage[idx] += 1

    uncovered_indices = np.where(coverage == 0)[0]
    multiple_coverage = np.where(coverage > 1)[0]

    print(f"   总样本数: {total_samples}")
    print(f"   覆盖样本数: {np.sum(coverage > 0)}")
    print(f"   未覆盖样本数: {len(uncovered_indices)}")
    print(f"   多次覆盖样本数: {len(multiple_coverage)}")

    if len(uncovered_indices) > 0:
        print(f"   未覆盖样本索引: {uncovered_indices[:10]}{'...' if len(uncovered_indices) > 10 else ''}")

    if len(multiple_coverage) > 0:
        print(f"   多次覆盖样本索引: {multiple_coverage[:10]}{'...' if len(multiple_coverage) > 10 else ''}")

    return {
        'total_samples': total_samples,
        'covered_samples': np.sum(coverage > 0),
        'uncovered_samples': len(uncovered_indices),
        'multiple_coverage_samples': len(multiple_coverage),
        'avg_coverage': np.mean(coverage[coverage > 0]) if np.sum(coverage > 0) > 0 else 0
    }


def test_shuffle_consistency(
        dataloader_with_shuffle: RNNDataloader,
        dataloader_without_shuffle: RNNDataloader,
        X_list: List[np.ndarray],
        L_list: List[np.ndarray]
) -> Dict:
    """
    测试打乱操作的consistency

    验证打乱操作是否只是改变了顺序，而没有改变数据内容
    """

    print(f"\n{'=' * 60}")
    print("测试打乱操作的consistency")
    print(f"{'=' * 60}")

    # 获取两个dataloader的所有批次信息
    shuffle_batch_infos = []
    no_shuffle_batch_infos = []

    # 收集不打乱dataloader的所有批次信息，建立全局索引到批次的映射
    global_idx_to_no_shuffle_batch = {}
    for i in range(len(dataloader_without_shuffle)):
        batch_info = dataloader_without_shuffle.get_batch_info(i)
        start_idx = batch_info['start_dyn_idx_global']
        batch_size = batch_info['actual_batch_size']

        # 将这个批次中的所有全局索引映射到这个批次
        for offset in range(batch_size):
            global_idx = start_idx + offset
            if global_idx not in global_idx_to_no_shuffle_batch:
                global_idx_to_no_shuffle_batch[global_idx] = []
            global_idx_to_no_shuffle_batch[global_idx].append({
                'batch_idx': i,
                'offset_in_batch': offset,
                'batch_info': batch_info
            })

        no_shuffle_batch_infos.append(batch_info)

    print(f"不打乱dataloader批次数: {len(no_shuffle_batch_infos)}")
    print(f"建立的全局索引映射数量: {len(global_idx_to_no_shuffle_batch)}")

    # 现在检查打乱dataloader的每个样本
    shuffle_mapping = {}
    errors = []

    for shuffle_batch_idx in range(len(dataloader_with_shuffle)):
        shuffle_batch_info = dataloader_with_shuffle.get_batch_info(shuffle_batch_idx)
        X_shuffle, L_shuffle = dataloader_with_shuffle[shuffle_batch_idx]

        batch_size = shuffle_batch_info['actual_batch_size']
        start_global = shuffle_batch_info['start_dyn_idx_global']
        shuffle_perm = shuffle_batch_info['shuffle_permutation']

        # 统计这个批次中每个样本的映射情况
        sample_mappings = []

        for sample_idx_in_shuffle_batch in range(batch_size):
            # 考虑段内打乱
            if dataloader_with_shuffle.shuffle_samples and shuffle_perm:
                # 找到原始顺序中的位置
                original_pos_in_batch = shuffle_perm[sample_idx_in_shuffle_batch]
            else:
                original_pos_in_batch = sample_idx_in_shuffle_batch

            # 计算这个样本的全局索引
            global_idx = start_global + original_pos_in_batch

            # 在不打乱的dataloader中查找这个全局索引
            if global_idx in global_idx_to_no_shuffle_batch:
                possible_mappings = global_idx_to_no_shuffle_batch[global_idx]

                # 找到最合适的映射（同一个段）
                best_mapping = None
                for mapping in possible_mappings:
                    mapping_batch_info = mapping['batch_info']
                    if mapping_batch_info['original_segment_idx'] == shuffle_batch_info['original_segment_idx']:
                        best_mapping = mapping
                        break

                # 如果没有找到同段的映射，使用第一个
                if not best_mapping and possible_mappings:
                    best_mapping = possible_mappings[0]

                if best_mapping:
                    sample_mappings.append({
                        'sample_idx_in_shuffle': sample_idx_in_shuffle_batch,
                        'no_shuffle_batch_idx': best_mapping['batch_idx'],
                        'offset_in_no_shuffle_batch': best_mapping['offset_in_batch'],
                        'global_idx': global_idx
                    })

        # 分析这个批次的映射模式
        if sample_mappings:
            # 统计不同的不打乱批次
            no_shuffle_batch_indices = list(set([m['no_shuffle_batch_idx'] for m in sample_mappings]))

            if len(no_shuffle_batch_indices) == 1:
                # 整个批次映射到同一个不打乱批次
                no_shuffle_batch_idx = no_shuffle_batch_indices[0]

                # 检查是否连续
                offsets = [m['offset_in_no_shuffle_batch'] for m in sample_mappings]
                if len(set(offsets)) == len(offsets):  # 所有偏移都不同
                    # 尝试找到偏移模式
                    sorted_offsets = sorted(offsets)
                    is_continuous = all(sorted_offsets[i + 1] - sorted_offsets[i] == 1
                                        for i in range(len(sorted_offsets) - 1))

                    if is_continuous:
                        # 连续映射，计算平均偏移
                        avg_offset = int(np.mean(offsets))
                        shuffle_mapping[shuffle_batch_idx] = {
                            'matching_batch_no_shuffle': no_shuffle_batch_idx,
                            'avg_offset': avg_offset,
                            'mapping_type': 'continuous',
                            'num_samples_mapped': len(sample_mappings)
                        }
                    else:
                        # 不连续映射
                        shuffle_mapping[shuffle_batch_idx] = {
                            'matching_batch_no_shuffle': no_shuffle_batch_idx,
                            'offsets': offsets,
                            'mapping_type': 'discontinuous',
                            'num_samples_mapped': len(sample_mappings)
                        }
            else:
                # 映射到多个不打乱批次
                shuffle_mapping[shuffle_batch_idx] = {
                    'matching_batches': no_shuffle_batch_indices,
                    'mapping_type': 'multiple',
                    'num_samples_mapped': len(sample_mappings)
                }

        # 对于完全映射的批次，验证数据一致性
        if shuffle_mapping.get(shuffle_batch_idx) and shuffle_mapping[shuffle_batch_idx][
            'mapping_type'] == 'continuous':
            mapping_info = shuffle_mapping[shuffle_batch_idx]
            no_shuffle_batch_idx = mapping_info['matching_batch_no_shuffle']
            avg_offset = mapping_info['avg_offset']

            # 获取不打乱批次的数据
            X_no_shuffle, L_no_shuffle = dataloader_without_shuffle[no_shuffle_batch_idx]
            no_shuffle_batch_info = no_shuffle_batch_infos[no_shuffle_batch_idx]

            # 比较重叠部分的数据
            overlap_size = min(batch_size, X_no_shuffle.shape[0] - avg_offset)
            if overlap_size > 0:
                X_shuffle_part = X_shuffle[:overlap_size].numpy()
                L_shuffle_part = L_shuffle[:overlap_size].numpy()
                X_no_shuffle_part = X_no_shuffle[avg_offset:avg_offset + overlap_size].numpy()
                L_no_shuffle_part = L_no_shuffle[avg_offset:avg_offset + overlap_size].numpy()

                X_error = np.mean(np.abs(X_shuffle_part - X_no_shuffle_part))
                L_error = np.mean(np.abs(L_shuffle_part - L_no_shuffle_part))

                if X_error > 1e-6 or L_error > 1e-6:
                    errors.append({
                        'shuffle_batch_idx': shuffle_batch_idx,
                        'no_shuffle_batch_idx': no_shuffle_batch_idx,
                        'offset': avg_offset,
                        'X_error': X_error,
                        'L_error': L_error
                    })

    # 打印统计信息
    print(f"\n打乱批次映射统计:")
    print(f"   总打乱批次数: {len(dataloader_with_shuffle)}")
    print(f"   成功映射批次数: {len(shuffle_mapping)}")

    # 统计映射类型
    mapping_types = {}
    for mapping in shuffle_mapping.values():
        mapping_type = mapping['mapping_type']
        mapping_types[mapping_type] = mapping_types.get(mapping_type, 0) + 1

    print(f"   映射类型统计:")
    for mapping_type, count in mapping_types.items():
        print(f"     {mapping_type}: {count}")

    if errors:
        print(f"\n数据一致性错误:")
        for error in errors[:10]:  # 只显示前10个错误
            print(
                f"   打乱批次{error['shuffle_batch_idx']} -> 不打乱批次{error['no_shuffle_batch_idx']}[偏移{error['offset']}]:")
            print(f"     X误差: {error['X_error']:.6e}, L误差: {error['L_error']:.6e}")

        if len(errors) > 10:
            print(f"   还有 {len(errors) - 10} 个错误未显示...")

    # 绘制映射关系图
    _plot_shuffle_mapping(shuffle_mapping, dataloader_with_shuffle, dataloader_without_shuffle)

    return {
        'shuffle_mapping': shuffle_mapping,
        'errors': errors,
        'mapping_types': mapping_types
    }


def _plot_shuffle_mapping(shuffle_mapping: Dict, dataloader_with_shuffle, dataloader_without_shuffle):
    """绘制打乱映射关系图"""

    # 准备数据
    shuffle_indices = []
    no_shuffle_indices = []
    mapping_types = []

    for shuffle_idx, mapping_info in shuffle_mapping.items():
        if mapping_info['mapping_type'] == 'continuous':
            shuffle_indices.append(shuffle_idx)
            no_shuffle_indices.append(mapping_info['matching_batch_no_shuffle'])
            mapping_types.append(0)  # 连续映射
        elif mapping_info['mapping_type'] == 'discontinuous':
            shuffle_indices.append(shuffle_idx)
            # 对于不连续映射，取第一个匹配的批次
            if isinstance(mapping_info['matching_batch_no_shuffle'], list):
                no_shuffle_indices.append(mapping_info['matching_batch_no_shuffle'][0])
            else:
                no_shuffle_indices.append(mapping_info['matching_batch_no_shuffle'])
            mapping_types.append(1)  # 不连续映射
        elif mapping_info['mapping_type'] == 'multiple':
            shuffle_indices.append(shuffle_idx)
            # 对于多批次映射，取第一个批次
            no_shuffle_indices.append(mapping_info['matching_batches'][0])
            mapping_types.append(2)  # 多批次映射

    if not shuffle_indices:
        print("没有有效的映射数据可绘制")
        return

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 映射关系散点图
    scatter = axes[0].scatter(shuffle_indices, no_shuffle_indices,
                              c=mapping_types, cmap='viridis', alpha=0.7, s=50)
    axes[0].set_xlabel('打乱批次索引')
    axes[0].set_ylabel('不打乱批次索引')
    axes[0].set_title('批次映射关系')
    axes[0].grid(True, alpha=0.3)

    # 添加颜色条图例
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['连续映射', '不连续映射', '多批次映射'])

    # 2. 映射分布直方图
    axes[1].hist(no_shuffle_indices, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1].set_xlabel('不打乱批次索引')
    axes[1].set_ylabel('频次')
    axes[1].set_title('不打乱批次索引分布')
    axes[1].grid(True, alpha=0.3)

    # 计算并显示重复值
    unique, counts = np.unique(no_shuffle_indices, return_counts=True)
    duplicate_indices = unique[counts > 1]
    if len(duplicate_indices) > 0:
        axes[1].axvspan(min(duplicate_indices) - 0.5, max(duplicate_indices) + 0.5,
                        alpha=0.2, color='red', label='重复映射区域')
        axes[1].legend()

    # 3. 段信息展示
    if shuffle_indices:
        # 获取打乱批次的段信息
        shuffle_segments = []
        for idx in shuffle_indices:
            batch_info = dataloader_with_shuffle.get_batch_info(idx)
            shuffle_segments.append(batch_info['original_segment_idx'])

        # 获取不打乱批次的段信息
        no_shuffle_segments = []
        for idx in no_shuffle_indices:
            batch_info = dataloader_without_shuffle.get_batch_info(idx)
            no_shuffle_segments.append(batch_info['original_segment_idx'])

        # 绘制段信息
        width = 0.35
        x = np.arange(len(shuffle_indices))

        axes[2].bar(x - width / 2, shuffle_segments, width, label='打乱批次段', alpha=0.7)
        axes[2].bar(x + width / 2, no_shuffle_segments, width, label='不打乱批次段', alpha=0.7)
        axes[2].set_xlabel('映射对索引')
        axes[2].set_ylabel('原始段索引')
        axes[2].set_title('批次对应的原始段')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print("\n映射关系统计:")
    print(f"  总映射对数: {len(shuffle_indices)}")
    print(f"  不打乱批次索引唯一值数量: {len(set(no_shuffle_indices))}")
    print(f"  打乱批次索引唯一值数量: {len(set(shuffle_indices))}")

    # 检查重复映射
    if len(set(no_shuffle_indices)) < len(no_shuffle_indices):
        print(f"  警告: 有不打乱批次被多次映射")
        print(f"  重复的不打乱批次索引: {duplicate_indices}")

def run_comprehensive_test():
    """运行全面的测试"""

    print("=" * 80)
    print("RNN DataLoader 全面测试")
    print("=" * 80)

    # 1. 生成测试数据
    print("\n1. 生成测试数据...")
    X_list, L_list, full_data_list = generate_mimo_data(
        n_segments=4,
        min_seg_len=50,
        max_seg_len=80,
        u_dim=2,
        y_dim=3,
        stack=10,
        noise_std=0.05
    )

    print(f"\n生成的数据段信息:")
    for i, (X_seg, L_seg) in enumerate(zip(X_list, L_list)):
        print(f"  段{i}: X形状={X_seg.shape}, L形状={L_seg.shape}")

    # 2. 测试1: 不打乱的情况
    print("\n\n2. 测试1: 不打乱的情况 (shuffle_segments=False, shuffle_samples=False)")
    dl_no_shuffle = RNNDataloader(
        X=X_list,
        L=L_list,
        stack=10,
        batch_size=16,
        shuffle_segments=False,
        shuffle_samples=False,
        drop_last=True,
        if_debug=False
    )

    stats_no_shuffle = test_rnn_dataloader_accuracy(
        dataloader=dl_no_shuffle,
        X_list=X_list,
        L_list=L_list,
        full_data_list=full_data_list,
        test_name="无打乱测试",
        plot_results=True
    )

    # 3. 测试2: 仅段间打乱
    print("\n\n3. 测试2: 仅段间打乱 (shuffle_segments=True, shuffle_samples=False)")
    dl_segment_shuffle = RNNDataloader(
        X=X_list,
        L=L_list,
        stack=10,
        batch_size=16,
        shuffle_segments=True,
        shuffle_samples=False,
        drop_last=True,
        if_debug=False
    )

    stats_segment_shuffle = test_rnn_dataloader_accuracy(
        dataloader=dl_segment_shuffle,
        X_list=X_list,
        L_list=L_list,
        full_data_list=full_data_list,
        test_name="段间打乱测试",
        plot_results=True
    )

    # 4. 测试3: 段内打乱
    print("\n\n4. 测试3: 段内打乱 (shuffle_segments=False, shuffle_samples=True)")
    dl_sample_shuffle = RNNDataloader(
        X=X_list,
        L=L_list,
        stack=10,
        batch_size=16,
        shuffle_segments=False,
        shuffle_samples=True,
        drop_last=True,
        if_debug=False
    )

    stats_sample_shuffle = test_rnn_dataloader_accuracy(
        dataloader=dl_sample_shuffle,
        X_list=X_list,
        L_list=L_list,
        full_data_list=full_data_list,
        test_name="段内打乱测试",
        plot_results=True
    )

    # 5. 测试4: 完全打乱
    print("\n\n5. 测试4: 完全打乱 (shuffle_segments=True, shuffle_samples=True)")
    dl_full_shuffle = RNNDataloader(
        X=X_list,
        L=L_list,
        stack=10,
        batch_size=16,
        shuffle_segments=True,
        shuffle_samples=True,
        drop_last=True,
        if_debug=False
    )

    stats_full_shuffle = test_rnn_dataloader_accuracy(
        dataloader=dl_full_shuffle,
        X_list=X_list,
        L_list=L_list,
        full_data_list=full_data_list,
        test_name="完全打乱测试",
        plot_results=True
    )

    # 6. 测试5: 测试drop_last=False的情况
    print("\n\n6. 测试5: 允许不完整批次 (drop_last=False)")
    dl_drop_last_false = RNNDataloader(
        X=X_list,
        L=L_list,
        stack=10,
        batch_size=16,
        shuffle_segments=False,
        shuffle_samples=False,
        drop_last=False,
        if_debug=True  # 打开调试信息查看批次生成
    )

    stats_drop_last_false = test_rnn_dataloader_accuracy(
        dataloader=dl_drop_last_false,
        X_list=X_list,
        L_list=L_list,
        full_data_list=full_data_list,
        test_name="允许不完整批次测试",
        plot_results=True
    )

    # 7. 验证打乱操作的consistency
    print("\n\n7. 验证打乱操作的consistency")
    shuffle_mapping = test_shuffle_consistency(
        dataloader_with_shuffle=dl_full_shuffle,
        dataloader_without_shuffle=dl_no_shuffle,
        X_list=X_list,
        L_list=L_list
    )

    # 8. 打印最终总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    all_stats = [
        ("无打乱", stats_no_shuffle),
        ("段间打乱", stats_segment_shuffle),
        ("段内打乱", stats_sample_shuffle),
        ("完全打乱", stats_full_shuffle),
        ("不完整批次", stats_drop_last_false)
    ]

    print(f"\n{'测试类型':<15} | {'批次数':<8} | {'X误差均值':<12} | {'L误差均值':<12} | {'连续性错误':<10}")
    print("-" * 70)

    for name, stats in all_stats:
        continuity_errors = stats.get('continuity_errors', 0)
        print(
            f"{name:<15} | {stats['total_batches']:<8} | {stats['avg_X_error']:.2e} | {stats['avg_L_error']:.2e} | {continuity_errors:<10}")

    # 9. 验证RNN连续性的特殊测试
    print("\n\n8. RNN连续性特殊测试")
    print("验证批次间的连续性对于RNN训练的重要性")

    # 创建一个小型测试数据来验证连续性
    test_X = [np.array([[i * 10 + j] for j in range(20)]) for i in range(2)]  # 2段，每段20个样本
    test_L = [np.array([[i * 100 + j] for j in range(20)]) for i in range(2)]

    dl_continuity = RNNDataloader(
        X=test_X,
        L=test_L,
        stack=5,
        batch_size=8,
        shuffle_segments=False,
        shuffle_samples=False,
        drop_last=True,
        if_debug=True
    )

    print("\n连续性测试的批次信息:")
    for i in range(min(5, len(dl_continuity))):  # 只显示前5个批次
        X_batch, L_batch = dl_continuity[i]
        batch_info = dl_continuity.get_batch_info(i)
        print(f"批次{i}: 起始索引={batch_info['start_dyn_idx_global']}, 段={batch_info['segment_idx']}")
        print(f"      X样本: {X_batch[:, 0].tolist()}")

    return {
        'stats_no_shuffle': stats_no_shuffle,
        'stats_segment_shuffle': stats_segment_shuffle,
        'stats_sample_shuffle': stats_sample_shuffle,
        'stats_full_shuffle': stats_full_shuffle,
        'stats_drop_last_false': stats_drop_last_false,
        'shuffle_mapping': shuffle_mapping
    }


def test_edge_cases():
    """测试边界情况"""

    print("\n" + "=" * 80)
    print("边界情况测试")
    print("=" * 80)

    # 1. 测试非常短的段
    print("\n1. 测试非常短的段")
    short_X = [np.random.randn(15, 10) for _ in range(3)]  # 每段只有15个样本
    short_L = [np.random.randn(15, 2) for _ in range(3)]

    try:
        dl_short = RNNDataloader(
            X=short_X,
            L=short_L,
            stack=10,
            batch_size=8,
            shuffle_segments=False,
            shuffle_samples=False,
            drop_last=True,
            if_debug=True
        )
        print(f"成功创建dataloader，批次数: {len(dl_short)}")
    except Exception as e:
        print(f"创建失败: {e}")

    # 2. 测试drop_last=False的情况
    print("\n2. 测试drop_last=False的边界情况")
    dl_no_drop = RNNDataloader(
        X=short_X,
        L=short_L,
        stack=5,
        batch_size=8,
        shuffle_segments=False,
        shuffle_samples=False,
        drop_last=False,
        if_debug=True
    )

    print(f"批次数: {len(dl_no_drop)}")
    for i in range(len(dl_no_drop)):
        X_batch, L_batch = dl_no_drop[i]
        print(f"批次{i}: 大小={X_batch.shape[0]}")

    # 3. 测试batch_size > stack的情况
    print("\n3. 测试batch_size > stack的情况")
    try:
        dl_large_batch = RNNDataloader(
            X=short_X,
            L=short_L,
            stack=5,
            batch_size=12,
            shuffle_segments=False,
            shuffle_samples=False,
            drop_last=True,
            if_debug=True
        )
        print(f"成功创建，批次数: {len(dl_large_batch)}")
    except Exception as e:
        print(f"创建失败: {e}")

    # 4. 测试单一段的情况
    print("\n4. 测试单一段的情况")
    single_X = [np.random.randn(100, 10)]
    single_L = [np.random.randn(100, 2)]

    dl_single = RNNDataloader(
        X=single_X,
        L=single_L,
        stack=10,
        batch_size=16,
        shuffle_segments=False,
        shuffle_samples=True,  # 段内打乱
        drop_last=True,
        if_debug=False
    )

    print(f"单一段dataloader批次数: {len(dl_single)}")

    return {
        'dl_short': dl_short if 'dl_short' in locals() else None,
        'dl_no_drop': dl_no_drop,
        'dl_large_batch': dl_large_batch if 'dl_large_batch' in locals() else None,
        'dl_single': dl_single
    }


if __name__ == "__main__":
    # 运行主测试
    results = run_comprehensive_test()

    # 运行边界情况测试
    edge_case_results = test_edge_cases()

    print("\n" + "=" * 80)
    print("所有测试完成!")
    print("=" * 80)