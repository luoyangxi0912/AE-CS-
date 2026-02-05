#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块 - AE-CS项目
根据IEEE论文和EDA结果实现

功能:
1. 数据加载和清洗
2. 人工缺失值生成 (MCAR/MAR/MNAR)
3. 数据归一化/标准化
4. 缺失值掩码生成
5. 时间窗口切分
6. 数据集划分
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')


class HangmeiPreprocessor:
    """Hangmei数据集预处理器"""

    def __init__(self,
                 data_path: str = r"D:\数据补全\hangmei_90_拼接好的.csv",
                 scaler_type: str = 'standard',
                 window_size: int = 48,
                 stride: int = 1):
        """
        初始化预处理器

        Args:
            data_path: 数据文件路径
            scaler_type: 归一化方法 ('standard', 'minmax', 'robust')
            window_size: 时间窗口大小
            stride: 滑动步长
        """
        self.data_path = Path(data_path)
        self.scaler_type = scaler_type
        self.window_size = window_size
        self.stride = stride

        # 初始化归一化器
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.feature_names = None
        self.n_features = None
        self.is_fitted = False

    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据

        Returns:
            DataFrame: 原始数据
        """
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path, encoding='gbk')

        # 移除非数值列
        # 保留: 航煤90%回收温度 和所有传感器数据
        # 移除: 序号, 日期
        cols_to_drop = ['序号', '日期']
        cols_to_keep = [col for col in df.columns if col not in cols_to_drop]
        df = df[cols_to_keep]

        # 将数据转换为float类型
        df = df.astype(np.float32)

        self.feature_names = df.columns.tolist()
        self.n_features = len(self.feature_names)

        print(f"Loaded data shape: {df.shape}")
        print(f"Features: {self.n_features}")

        return df

    def create_missing_mask(self,
                           data: np.ndarray,
                           missing_rate: float = 0.2,
                           missing_type: str = 'MCAR',
                           seed: Optional[int] = None) -> np.ndarray:
        """
        创建人工缺失值掩码

        根据论文Section II.C的问题定义:
        M ∈ {0, 1}^(T×N), where m_{t,n} = 1 if x_{t,n} is observed

        Args:
            data: 原始数据 [T, N]
            missing_rate: 缺失率
            missing_type: 缺失类型 ('MCAR', 'MAR', 'MNAR')
            seed: 随机种子

        Returns:
            mask: 缺失值掩码，1表示观测，0表示缺失
        """
        if seed is not None:
            np.random.seed(seed)

        T, N = data.shape
        mask = np.ones((T, N), dtype=np.float32)

        if missing_type == 'MCAR':
            # Missing Completely At Random
            # 随机选择缺失位置
            missing_idx = np.random.random((T, N)) < missing_rate
            mask[missing_idx] = 0

        elif missing_type == 'MAR':
            # Missing At Random
            # 缺失与观测到的变量相关
            total_elements = T * N
            target_missing = int(total_elements * missing_rate)

            # 为每个变量分配目标缺失数量
            missing_per_var = target_missing // N
            extra_missing = target_missing % N

            for n in range(N):
                # 为该变量分配缺失配额
                var_target = missing_per_var + (1 if n < extra_missing else 0)

                # 选择一个相关变量作为条件
                related_vars = [i for i in range(N) if i != n]
                related_var = np.random.choice(related_vars)

                # 基于相关变量的值确定缺失概率
                # 相关变量值越高，缺失概率越大
                related_values = data[:, related_var]
                # 归一化到[0,1]范围
                if related_values.max() > related_values.min():
                    normalized_values = (related_values - related_values.min()) / (related_values.max() - related_values.min())
                else:
                    normalized_values = np.ones(T) * 0.5

                # 使用加权随机采样：值越大，被选中概率越高
                probabilities = normalized_values / normalized_values.sum()

                # 采样缺失位置
                if var_target > 0 and var_target <= T:
                    selected_idx = np.random.choice(T, size=var_target, replace=False, p=probabilities)
                    mask[selected_idx, n] = 0

        elif missing_type == 'MNAR':
            # Missing Not At Random
            # 缺失与未观测到的值本身相关
            total_elements = T * N
            target_missing = int(total_elements * missing_rate)

            # 为每个变量分配目标缺失数量
            missing_per_var = target_missing // N
            extra_missing = target_missing % N

            for n in range(N):
                # 为该变量分配缺失配额
                var_target = missing_per_var + (1 if n < extra_missing else 0)

                values = data[:, n]

                # 计算每个位置的"极端性"得分
                # 离均值越远，得分越高
                mean_val = values.mean()
                std_val = values.std()

                if std_val > 0:
                    # 标准化偏差（绝对值）
                    extremeness = np.abs(values - mean_val) / std_val
                    # 转换为概率（极端值更可能被选中）
                    probabilities = extremeness / extremeness.sum()
                else:
                    # 如果方差为0，均匀采样
                    probabilities = np.ones(T) / T

                # 采样缺失位置
                if var_target > 0 and var_target <= T:
                    selected_idx = np.random.choice(T, size=var_target, replace=False, p=probabilities)
                    mask[selected_idx, n] = 0

        else:
            raise ValueError(f"Unknown missing type: {missing_type}")

        actual_rate = 1 - mask.mean()
        print(f"Created {missing_type} mask: target={missing_rate:.2%}, actual={actual_rate:.2%}")

        return mask

    def create_structured_missing(self,
                                  data: np.ndarray,
                                  missing_rate: float = 0.2,
                                  block_size: int = 5,
                                  seed: Optional[int] = None) -> np.ndarray:
        """
        创建结构化缺失模式（连续缺失块）

        Args:
            data: 原始数据
            missing_rate: 目标缺失率
            block_size: 缺失块大小
            seed: 随机种子

        Returns:
            mask: 缺失值掩码
        """
        if seed is not None:
            np.random.seed(seed)

        T, N = data.shape
        mask = np.ones((T, N), dtype=np.float32)

        # 计算需要的缺失块数量
        total_cells = T * N
        target_missing = int(total_cells * missing_rate)
        n_blocks = target_missing // block_size

        for _ in range(n_blocks):
            # 随机选择起始位置和变量
            t_start = np.random.randint(0, T - block_size + 1)
            n_var = np.random.randint(0, N)

            # 创建连续缺失块
            mask[t_start:t_start+block_size, n_var] = 0

        actual_rate = 1 - mask.mean()
        print(f"Created structured missing: target={missing_rate:.2%}, actual={actual_rate:.2%}")

        return mask

    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        数据归一化

        Args:
            data: 输入数据 [T, N]
            fit: 是否拟合scaler

        Returns:
            normalized: 归一化后的数据
        """
        if fit:
            normalized = self.scaler.fit_transform(data)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            normalized = self.scaler.transform(data)

        return normalized.astype(np.float32)

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        反归一化

        Args:
            data: 归一化的数据

        Returns:
            original: 原始尺度的数据
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted.")

        return self.scaler.inverse_transform(data).astype(np.float32)

    def create_windows(self,
                       data: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        创建时间窗口序列

        Args:
            data: 输入数据 [T, N]
            mask: 缺失值掩码 [T, N]

        Returns:
            windows: 窗口化数据 [n_windows, window_size, N]
            window_masks: 窗口化掩码 [n_windows, window_size, N]
        """
        T, N = data.shape
        n_windows = (T - self.window_size) // self.stride + 1

        windows = np.zeros((n_windows, self.window_size, N), dtype=np.float32)

        for i in range(n_windows):
            start = i * self.stride
            end = start + self.window_size
            windows[i] = data[start:end]

        if mask is not None:
            window_masks = np.zeros((n_windows, self.window_size, N), dtype=np.float32)
            for i in range(n_windows):
                start = i * self.stride
                end = start + self.window_size
                window_masks[i] = mask[start:end]
            return windows, window_masks

        return windows, None

    def split_data(self,
                   data: np.ndarray,
                   mask: np.ndarray,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   shuffle: bool = False,
                   seed: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        划分数据集

        注意: 对于时间序列，通常不shuffle以保持时间顺序

        Args:
            data: 数据 [n_samples, ...]
            mask: 掩码 [n_samples, ...]
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            shuffle: 是否打乱
            seed: 随机种子

        Returns:
            splits: {'train': (X, M), 'val': (X, M), 'test': (X, M)}
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        n_samples = len(data)
        indices = np.arange(n_samples)

        if shuffle and seed is not None:
            np.random.seed(seed)
            np.random.shuffle(indices)

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        splits = {
            'train': (data[train_idx], mask[train_idx]),
            'val': (data[val_idx], mask[val_idx]),
            'test': (data[test_idx], mask[test_idx])
        }

        print(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return splits

    def prepare_data(self,
                     missing_rate: float = 0.2,
                     missing_type: str = 'MCAR',
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     seed: Optional[int] = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        完整的数据准备流程

        重要：先划分数据集，再归一化，避免数据泄露！

        Args:
            missing_rate: 缺失率
            missing_type: 缺失类型
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子

        Returns:
            splits: 划分后的数据集
        """
        # 1. 加载原始数据
        df = self.load_data()
        data = df.values

        T, N = data.shape

        # 2. 先划分原始数据（时间序列按顺序划分）
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        print(f"Data split before normalization: train={train_data.shape[0]}, val={val_data.shape[0]}, test={test_data.shape[0]}")

        # 3. 归一化：只在训练集上fit，避免数据泄露
        train_normalized = self.normalize(train_data, fit=True)
        val_normalized = self.normalize(val_data, fit=False)
        test_normalized = self.normalize(test_data, fit=False)

        # 验证归一化是否正确
        print(f"Normalization check:")
        print(f"  Train: mean={train_normalized.mean():.6f}, std={train_normalized.std():.6f}")
        print(f"  Val:   mean={val_normalized.mean():.6f}, std={val_normalized.std():.6f}")
        print(f"  Test:  mean={test_normalized.mean():.6f}, std={test_normalized.std():.6f}")

        # 4. 为每个子集创建缺失值掩码
        if seed is not None:
            np.random.seed(seed)

        train_mask = self.create_missing_mask(
            train_normalized,
            missing_rate=missing_rate,
            missing_type=missing_type,
            seed=seed
        )

        val_mask = self.create_missing_mask(
            val_normalized,
            missing_rate=missing_rate,
            missing_type=missing_type,
            seed=seed + 1 if seed is not None else None
        )

        test_mask = self.create_missing_mask(
            test_normalized,
            missing_rate=missing_rate,
            missing_type=missing_type,
            seed=seed + 2 if seed is not None else None
        )

        # 5. 为每个子集创建时间窗口
        train_windows, train_window_masks = self.create_windows(train_normalized, train_mask)
        val_windows, val_window_masks = self.create_windows(val_normalized, val_mask)
        test_windows, test_window_masks = self.create_windows(test_normalized, test_mask)

        print(f"Created windows: train={len(train_windows)}, val={len(val_windows)}, test={len(test_windows)}")

        # 6. 返回划分后的数据集
        splits = {
            'train': (train_windows, train_window_masks),
            'val': (val_windows, val_window_masks),
            'test': (test_windows, test_window_masks)
        }

        return splits

    def save_preprocessor(self, path: str):
        """保存预处理器状态"""
        state = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'window_size': self.window_size,
            'stride': self.stride,
            'scaler_type': self.scaler_type
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved to: {path}")

    def load_preprocessor(self, path: str):
        """加载预处理器状态"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.n_features = state['n_features']
        self.window_size = state['window_size']
        self.stride = state['stride']
        self.scaler_type = state['scaler_type']
        self.is_fitted = True
        print(f"Preprocessor loaded from: {path}")


class BernoulliCorruptor:
    """
    Bernoulli损坏器 - 用于相干去噪

    根据论文公式(6):
    M^(k) = M ⊙ B^(k), B^(k)_{t,n} ~ Bernoulli(1 - p_drop)
    """

    def __init__(self, p_drop: float = 0.1):
        """
        Args:
            p_drop: 额外丢弃概率
        """
        self.p_drop = p_drop

    def corrupt(self,
                mask: np.ndarray,
                n_versions: int = 3,
                seed: Optional[int] = None) -> List[np.ndarray]:
        """
        生成K个损坏版本的掩码

        Args:
            mask: 原始掩码 [batch, T, N]
            n_versions: 损坏版本数量K
            seed: 随机种子

        Returns:
            corrupted_masks: K个损坏掩码的列表
        """
        if seed is not None:
            np.random.seed(seed)

        corrupted_masks = []

        for k in range(n_versions):
            # 生成Bernoulli掩码
            bernoulli_mask = np.random.random(mask.shape) > self.p_drop
            bernoulli_mask = bernoulli_mask.astype(np.float32)

            # 与原始掩码相乘
            corrupted = mask * bernoulli_mask
            corrupted_masks.append(corrupted)

        return corrupted_masks

    def compute_weights(self,
                        original_mask: np.ndarray,
                        corrupted_masks: List[np.ndarray],
                        sigma_c: Optional[float] = None) -> List[float]:
        """
        计算损坏版本的权重

        根据论文公式(7):
        w^(k) = exp(-||M^(k) - M||_1 / σ_c^2)

        Args:
            original_mask: 原始掩码
            corrupted_masks: 损坏掩码列表
            sigma_c: 带宽参数 (如果为None，自动根据掩码大小设置)

        Returns:
            weights: 权重列表
        """
        # 自适应设置sigma_c：使用掩码元素总数的平方根
        if sigma_c is None:
            n_elements = original_mask.size
            sigma_c = np.sqrt(n_elements) * 0.1  # 经验值：总元素数的10%的平方根

        weights = []

        for corrupted in corrupted_masks:
            l1_diff = np.abs(corrupted - original_mask).sum()
            weight = np.exp(-l1_diff / (sigma_c ** 2))
            weights.append(weight)

        # 归一化权重
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            # 如果所有权重都是0，使用均匀分布
            weights = [1.0 / len(corrupted_masks)] * len(corrupted_masks)

        return weights


def create_evaluation_masks(data: np.ndarray,
                           observed_mask: np.ndarray,
                           eval_ratio: float = 0.1,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    创建评估掩码

    从观测值中随机选择一部分作为评估集，
    在训练时隐藏这些值，在评估时使用它们计算指标

    Args:
        data: 数据
        observed_mask: 观测掩码 (1=observed)
        eval_ratio: 评估集比例
        seed: 随机种子

    Returns:
        eval_mask: 评估掩码 (1=用于评估的观测值)
    """
    if seed is not None:
        np.random.seed(seed)

    # 只从观测值中选择
    observed_idx = np.where(observed_mask == 1)
    n_observed = len(observed_idx[0])
    n_eval = int(n_observed * eval_ratio)

    # 随机选择
    eval_idx = np.random.choice(n_observed, n_eval, replace=False)

    eval_mask = np.zeros_like(observed_mask)
    eval_mask[observed_idx[0][eval_idx], observed_idx[1][eval_idx]] = 1

    return eval_mask.astype(np.float32)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("数据预处理模块测试")
    print("=" * 80)

    # 创建预处理器
    preprocessor = HangmeiPreprocessor(
        scaler_type='standard',
        window_size=48,
        stride=1
    )

    # 准备数据
    splits = preprocessor.prepare_data(
        missing_rate=0.2,
        missing_type='MCAR',
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )

    # 打印数据形状
    print("\n数据形状:")
    for name, (X, M) in splits.items():
        print(f"  {name}: X={X.shape}, M={M.shape}")
        missing_rate = 1 - M.mean()
        print(f"         缺失率: {missing_rate:.2%}")

    # 测试Bernoulli损坏器
    print("\n测试Bernoulli损坏器:")
    corruptor = BernoulliCorruptor(p_drop=0.1)

    train_X, train_M = splits['train']
    sample_mask = train_M[:5]  # 取5个样本测试

    corrupted_masks = corruptor.corrupt(sample_mask, n_versions=3, seed=42)
    weights = corruptor.compute_weights(sample_mask, corrupted_masks)

    print(f"  原始掩码观测率: {sample_mask.mean():.2%}")
    for i, (cm, w) in enumerate(zip(corrupted_masks, weights)):
        print(f"  损坏版本{i+1}观测率: {cm.mean():.2%}, 权重: {w:.4f}")

    # 保存预处理器
    preprocessor.save_preprocessor("D:/数据补全/checkpoints/preprocessor.pkl")

    print("\n" + "=" * 80)
    print("预处理模块测试完成!")
    print("=" * 80)
