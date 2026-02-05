"""
循环插补模块 - 迭代精化缺失数据补全

核心思想：
1. 首次推理得到初步预测
2. 用预测值更新缺失位置
3. 基于更新后的数据重新计算邻域关系
4. 重复直到收敛

优势：
- 邻域计算基于更准确的数据
- 信息从观测区域逐步传播到缺失区域
- 适合高缺失率场景
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Tuple


def compute_spatial_knn_init_iterative(X_filled, X_observed, mask, k=5):
    """
    循环插补版本的空间KNN初始化

    关键区别：使用已填充的数据 X_filled 计算邻域距离

    Args:
        X_filled: [batch, time, features] - 当前迭代的填充数据（用于计算距离）
        X_observed: [batch, time, features] - 原始观测数据（观测位置保持不变）
        mask: [batch, time, features] - 掩码矩阵
        k: int - 邻域数量

    Returns:
        X_init: [batch, time, features] - 空间KNN初始化后的数据
    """
    batch_size, time_steps, n_features = X_filled.shape
    X_filled_np = X_filled.numpy()
    X_obs_np = X_observed.numpy()
    M = mask.numpy()

    # 初始化：观测位置保持原始值
    X_init = X_obs_np * M

    for b in range(batch_size):
        X_b = X_filled_np[b]  # 使用填充后的数据计算距离
        M_b = M[b]

        for t in range(time_steps):
            missing_vars = np.where(M_b[t, :] == 0)[0]
            if len(missing_vars) == 0:
                continue

            # 计算距离（基于填充后的完整数据）
            distances = np.full(time_steps, np.inf)
            for s in range(time_steps):
                if s == t:
                    continue
                # 使用所有变量计算距离（因为数据已填充）
                diff = X_b[t, :] - X_b[s, :]
                distances[s] = np.sqrt(np.mean(diff ** 2))

            k_actual = min(k, int(np.sum(distances < np.inf)))
            if k_actual == 0:
                continue

            k_indices = np.argsort(distances)[:k_actual]
            k_distances = distances[k_indices]

            sigma = np.median(k_distances) + 1e-8
            weights = np.exp(-k_distances ** 2 / (sigma ** 2))
            weights = weights / (np.sum(weights) + 1e-8)

            for v in missing_vars:
                # 使用填充后的邻居值
                X_init[b, t, v] = np.sum(weights * X_b[k_indices, v])

    return tf.constant(X_init, dtype=tf.float32)


def compute_temporal_knn_init_iterative(X_filled, X_observed, mask, k=5):
    """
    循环插补版本的时间KNN初始化

    关键区别：使用已填充的数据 X_filled 计算变量间距离

    Args:
        X_filled: [batch, time, features] - 当前迭代的填充数据
        X_observed: [batch, time, features] - 原始观测数据
        mask: [batch, time, features] - 掩码矩阵
        k: int - 邻域数量

    Returns:
        X_init: [batch, time, features] - 时间KNN初始化后的数据
    """
    batch_size, time_steps, n_features = X_filled.shape
    X_filled_np = X_filled.numpy()
    X_obs_np = X_observed.numpy()
    M = mask.numpy()

    X_init = X_obs_np * M

    for b in range(batch_size):
        X_b = X_filled_np[b]
        M_b = M[b]

        # 使用填充后的完整数据计算变量间距离
        var_distances = np.zeros((n_features, n_features))
        for j in range(n_features):
            for m in range(n_features):
                if j == m:
                    var_distances[j, m] = np.inf
                else:
                    diff = X_b[:, j] - X_b[:, m]
                    var_distances[j, m] = np.sqrt(np.mean(diff ** 2))

        for v in range(n_features):
            k_actual = min(k, n_features - 1)
            neighbor_vars = np.argsort(var_distances[v, :])[:k_actual]
            neighbor_dists = var_distances[v, neighbor_vars]

            sigma = np.median(neighbor_dists) + 1e-8
            var_weights = np.exp(-neighbor_dists ** 2 / (sigma ** 2))
            var_weights = var_weights / (np.sum(var_weights) + 1e-8)

            missing_times = np.where(M_b[:, v] == 0)[0]
            for t in missing_times:
                X_init[b, t, v] = np.sum(var_weights * X_b[t, neighbor_vars])

    return tf.constant(X_init, dtype=tf.float32)


class IterativeImputer:
    """
    循环插补器

    支持两种模式：
    1. 简单迭代：仅重复前向传播
    2. 完整迭代：每次迭代重新计算KNN初始化
    """

    def __init__(self, model, max_iters: int = 5, tol: float = 1e-4,
                 momentum: float = 0.0, recompute_knn: bool = True,
                 verbose: bool = False):
        """
        Args:
            model: AECS模型
            max_iters: 最大迭代次数
            tol: 收敛阈值（预测变化量）
            momentum: 动量系数（0-1），用于平滑更新，防止震荡
            recompute_knn: 是否每次迭代重新计算KNN初始化
            verbose: 是否打印迭代信息
        """
        self.model = model
        self.max_iters = max_iters
        self.tol = tol
        self.momentum = momentum
        self.recompute_knn = recompute_knn
        self.verbose = verbose

    def impute(self, X: tf.Tensor, mask: tf.Tensor,
               X_true: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, Dict]:
        """
        执行循环插补

        Args:
            X: [batch, time, features] - 输入数据（观测值）
            mask: [batch, time, features] - 掩码
            X_true: 可选，真实值用于评估RMSE

        Returns:
            X_imputed: 最终插补结果
            history: 迭代历史（包含每轮的change和可选的rmse）
        """
        # 初始化：缺失位置为0
        X_current = X * mask
        X_prev = X_current

        history = {
            'n_iters': 0,
            'change': [],
            'rmse': [] if X_true is not None else None,
            'converged': False
        }

        for i in range(self.max_iters):
            if self.recompute_knn:
                # 完整迭代：重新计算KNN初始化
                X_pred = self._forward_with_updated_knn(X_current, X, mask)
            else:
                # 简单迭代：直接前向传播
                X_pred = self.model(X_current, mask, training=False)

            # 更新缺失位置（带动量）
            if self.momentum > 0:
                X_imputed = (1 - self.momentum) * X_pred + self.momentum * X_prev
            else:
                X_imputed = X_pred

            # 保持观测位置不变
            X_new = mask * X + (1.0 - mask) * X_imputed

            # 计算变化量
            change = tf.reduce_mean(tf.abs(X_new - X_current)).numpy()
            history['change'].append(change)
            history['n_iters'] = i + 1

            # 计算RMSE（如果提供真实值）
            if X_true is not None:
                rmse = self._compute_rmse(X_new, X_true, mask)
                history['rmse'].append(rmse)
                if self.verbose:
                    print(f"Iter {i+1}: RMSE={rmse:.6f}, Change={change:.6f}")
            elif self.verbose:
                print(f"Iter {i+1}: Change={change:.6f}")

            # 检查收敛
            if change < self.tol:
                history['converged'] = True
                if self.verbose:
                    print(f"Converged at iteration {i+1}")
                break

            X_prev = X_current
            X_current = X_new

        return X_new, history

    def _forward_with_updated_knn(self, X_current, X_observed, mask):
        """
        使用更新后的KNN初始化进行前向传播

        这是循环插补的核心：
        - 空间/时间KNN基于当前填充结果计算
        - 邻域关系随迭代改进
        """
        # 计算缺失率
        missing_rate = 1.0 - tf.reduce_mean(mask, axis=[1, 2])

        # 第一编码器：零值填充（标准方式）
        x_zero = X_current * mask
        z_orig = self.model.encoder_orig(x_zero, mask, training=False)

        # 第二编码器：使用更新后的数据计算空间KNN
        x_space_init = compute_spatial_knn_init_iterative(
            X_current, X_observed, mask, k=self.model.k_spatial
        )
        z_space = self.model.encoder_space(x_space_init, mask, training=False)

        # 第三编码器：使用更新后的数据计算时间KNN
        x_time_init = compute_temporal_knn_init_iterative(
            X_current, X_observed, mask, k=self.model.k_temporal
        )
        z_time = self.model.encoder_time(x_time_init, mask, training=False)

        # 自适应融合
        alpha, z_fused = self.model.gating_network(
            z_orig, z_space, z_time, missing_rate, training=False
        )

        # 解码
        x_pred = self.model.decoder(z_fused, training=False)

        return x_pred

    def _compute_rmse(self, X_pred, X_true, mask):
        """计算缺失位置的RMSE"""
        missing_mask = 1.0 - mask
        n_missing = tf.reduce_sum(missing_mask) + 1e-8
        mse = tf.reduce_sum(tf.square(X_pred - X_true) * missing_mask) / n_missing
        return tf.sqrt(mse).numpy()


def adaptive_iterative_impute(model, X, mask, X_true=None,
                              base_iters=3, high_missing_iters=10,
                              missing_threshold=0.4):
    """
    自适应循环插补

    根据缺失率自动调整迭代次数：
    - 低缺失率：少量迭代即可
    - 高缺失率：需要更多迭代让信息传播

    Args:
        model: AECS模型
        X: 输入数据
        mask: 掩码
        X_true: 可选，真实值
        base_iters: 基础迭代次数
        high_missing_iters: 高缺失率时的迭代次数
        missing_threshold: 高缺失率阈值

    Returns:
        X_imputed: 插补结果
        history: 迭代历史
    """
    # 计算缺失率
    missing_rate = 1.0 - tf.reduce_mean(mask).numpy()

    # 根据缺失率选择迭代次数
    if missing_rate > missing_threshold:
        max_iters = high_missing_iters
    else:
        max_iters = base_iters

    imputer = IterativeImputer(
        model,
        max_iters=max_iters,
        recompute_knn=True,
        momentum=0.1 if missing_rate > missing_threshold else 0.0,
        verbose=True
    )

    print(f"Missing rate: {missing_rate:.2%}, using {max_iters} iterations")
    return imputer.impute(X, mask, X_true)
