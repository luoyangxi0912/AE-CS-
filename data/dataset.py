#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorFlow数据加载器 - AE-CS项目
提供高效的数据批处理和预取功能
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path

# 导入预处理器
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocessor import HangmeiPreprocessor, BernoulliCorruptor


class AECSDataset:
    """AE-CS模型的TensorFlow数据集封装"""

    def __init__(self,
                 data: np.ndarray,
                 mask: np.ndarray,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 buffer_size: int = 1000,
                 seed: Optional[int] = None):
        """
        初始化数据集

        Args:
            data: 数据 [n_samples, window_size, n_features]
            mask: 掩码 [n_samples, window_size, n_features]
            batch_size: 批大小
            shuffle: 是否打乱
            buffer_size: 打乱缓冲区大小
            seed: 随机种子
        """
        self.data = data.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed

        self.n_samples = len(data)
        self.window_size = data.shape[1]
        self.n_features = data.shape[2]

        self._dataset = None

    def create_dataset(self) -> tf.data.Dataset:
        """
        创建TensorFlow Dataset

        Returns:
            tf.data.Dataset: (X, M) 对
        """
        # 创建Dataset
        dataset = tf.data.Dataset.from_tensor_slices((self.data, self.mask))

        # 打乱
        if self.shuffle:
            if self.seed is not None:
                dataset = dataset.shuffle(
                    buffer_size=self.buffer_size,
                    seed=self.seed,
                    reshuffle_each_iteration=True
                )
            else:
                dataset = dataset.shuffle(
                    buffer_size=self.buffer_size,
                    reshuffle_each_iteration=True
                )

        # 批处理
        dataset = dataset.batch(self.batch_size, drop_remainder=False)

        # 预取
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self._dataset = dataset
        return dataset

    def get_dataset(self) -> tf.data.Dataset:
        """获取或创建数据集"""
        if self._dataset is None:
            self._dataset = self.create_dataset()
        return self._dataset

    @property
    def steps_per_epoch(self) -> int:
        """每个epoch的步数"""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __len__(self):
        return self.n_samples


class AECSDataLoader:
    """AE-CS数据加载器管理器"""

    def __init__(self,
                 preprocessor: Optional[HangmeiPreprocessor] = None,
                 batch_size: int = 32,
                 shuffle_train: bool = True,
                 seed: Optional[int] = 42):
        """
        初始化数据加载器

        Args:
            preprocessor: 预处理器实例
            batch_size: 批大小
            shuffle_train: 是否打乱训练集
            seed: 随机种子
        """
        self.preprocessor = preprocessor or HangmeiPreprocessor()
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.splits = None

    def prepare(self,
                missing_rate: float = 0.2,
                missing_type: str = 'MCAR',
                train_ratio: float = 0.7,
                val_ratio: float = 0.15) -> Dict[str, AECSDataset]:
        """
        准备所有数据集

        Args:
            missing_rate: 缺失率
            missing_type: 缺失类型
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            数据集字典
        """
        # 使用预处理器准备数据
        self.splits = self.preprocessor.prepare_data(
            missing_rate=missing_rate,
            missing_type=missing_type,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=self.seed
        )

        # 创建训练集
        train_X, train_M = self.splits['train']
        self.train_dataset = AECSDataset(
            train_X, train_M,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            seed=self.seed
        )

        # 创建验证集
        val_X, val_M = self.splits['val']
        self.val_dataset = AECSDataset(
            val_X, val_M,
            batch_size=self.batch_size,
            shuffle=False
        )

        # 创建测试集
        test_X, test_M = self.splits['test']
        self.test_dataset = AECSDataset(
            test_X, test_M,
            batch_size=self.batch_size,
            shuffle=False
        )

        return {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }

    def get_train_dataset(self) -> tf.data.Dataset:
        """获取训练数据集"""
        if self.train_dataset is None:
            raise ValueError("Data not prepared. Call prepare() first.")
        return self.train_dataset.get_dataset()

    def get_val_dataset(self) -> tf.data.Dataset:
        """获取验证数据集"""
        if self.val_dataset is None:
            raise ValueError("Data not prepared. Call prepare() first.")
        return self.val_dataset.get_dataset()

    def get_test_dataset(self) -> tf.data.Dataset:
        """获取测试数据集"""
        if self.test_dataset is None:
            raise ValueError("Data not prepared. Call prepare() first.")
        return self.test_dataset.get_dataset()

    @property
    def n_features(self) -> int:
        """特征数量"""
        return self.preprocessor.n_features

    @property
    def window_size(self) -> int:
        """窗口大小"""
        return self.preprocessor.window_size

    @property
    def feature_names(self):
        """特征名称"""
        return self.preprocessor.feature_names


def create_coherent_denoising_batch(X: tf.Tensor,
                                    M: tf.Tensor,
                                    p_drop: float = 0.1,
                                    n_versions: int = 3) -> Tuple[tf.Tensor, tf.Tensor, Tuple[tf.Tensor, ...]]:
    """
    创建相干去噪的批次数据

    Args:
        X: 输入数据 [batch, T, N]
        M: 掩码 [batch, T, N]
        p_drop: 丢弃概率
        n_versions: 损坏版本数量

    Returns:
        X: 原始数据
        M: 原始掩码
        corrupted_Ms: 损坏掩码元组 (tuple of tensors)
    """
    corrupted_Ms = []

    for _ in range(n_versions):
        # 生成Bernoulli掩码
        bernoulli = tf.cast(
            tf.random.uniform(tf.shape(M)) > p_drop,
            tf.float32
        )
        # 与原始掩码相乘
        corrupted = M * bernoulli
        corrupted_Ms.append(corrupted)

    return X, M, tuple(corrupted_Ms)


# ============================================================================
# 自定义训练数据生成器
# ============================================================================

class CoherentDenoisingGenerator:
    """
    相干去噪数据生成器

    在每个batch中自动生成K个损坏版本
    """

    def __init__(self,
                 dataset: AECSDataset,
                 p_drop: float = 0.1,
                 n_versions: int = 3):
        """
        Args:
            dataset: 数据集
            p_drop: 丢弃概率
            n_versions: 损坏版本数量K
        """
        self.dataset = dataset
        self.p_drop = p_drop
        self.n_versions = n_versions

    def __call__(self):
        """生成器函数"""
        for X, M in self.dataset.get_dataset():
            # 生成损坏版本
            corrupted_Ms = []
            for _ in range(self.n_versions):
                bernoulli = tf.cast(
                    tf.random.uniform(tf.shape(M)) > self.p_drop,
                    tf.float32
                )
                corrupted = M * bernoulli
                corrupted_Ms.append(corrupted)

            yield X, M, tuple(corrupted_Ms)

    def create_tf_dataset(self) -> tf.data.Dataset:
        """创建TensorFlow Dataset"""
        # 定义输出签名 - 使用tuple而不是list
        output_signature = (
            tf.TensorSpec(shape=(None, self.dataset.window_size, self.dataset.n_features),
                         dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.dataset.window_size, self.dataset.n_features),
                         dtype=tf.float32),
            tuple(tf.TensorSpec(shape=(None, self.dataset.window_size, self.dataset.n_features),
                          dtype=tf.float32) for _ in range(self.n_versions))
        )

        return tf.data.Dataset.from_generator(
            self,
            output_signature=output_signature
        )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TensorFlow数据加载器测试")
    print("=" * 80)

    # 检查GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPU设备: {gpus}")

    # 创建数据加载器
    loader = AECSDataLoader(batch_size=64, seed=42)

    # 准备数据
    datasets = loader.prepare(
        missing_rate=0.2,
        missing_type='MCAR',
        train_ratio=0.7,
        val_ratio=0.15
    )

    print(f"\n数据加载器信息:")
    print(f"  特征数: {loader.n_features}")
    print(f"  窗口大小: {loader.window_size}")
    print(f"  训练样本: {len(datasets['train'])}")
    print(f"  验证样本: {len(datasets['val'])}")
    print(f"  测试样本: {len(datasets['test'])}")

    # 测试训练数据集
    print("\n测试训练数据集:")
    train_ds = loader.get_train_dataset()

    for i, (X_batch, M_batch) in enumerate(train_ds.take(3)):
        print(f"  Batch {i+1}: X={X_batch.shape}, M={M_batch.shape}")
        print(f"           缺失率: {1 - tf.reduce_mean(M_batch).numpy():.2%}")

    # 测试相干去噪生成器
    print("\n测试相干去噪生成器:")
    gen = CoherentDenoisingGenerator(
        datasets['train'],
        p_drop=0.1,
        n_versions=3
    )

    denoising_ds = gen.create_tf_dataset()

    for i, (X, M, corrupted_Ms) in enumerate(denoising_ds.take(2)):
        print(f"  Batch {i+1}:")
        print(f"    原始掩码观测率: {tf.reduce_mean(M).numpy():.2%}")
        for j, cm in enumerate(corrupted_Ms):
            print(f"    损坏版本{j+1}观测率: {tf.reduce_mean(cm).numpy():.2%}")

    print("\n" + "=" * 80)
    print("数据加载器测试完成!")
    print("=" * 80)
