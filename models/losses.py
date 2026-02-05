"""
Loss Functions for AE-CS

实现专利中的四个损失函数:
1. L_recon: 重建损失 - 仅在观测位置计算
2. L_consist: 一致性损失 - 加权一致性约束
3. L_space: 空间邻域保持损失 - 保持空间流形结构
4. L_time: 时间邻域保持损失 - 保持时间流形结构

注意：L_missing（伪标签损失）已被移除。
实验证明 KNN 伪标签在"无共同观测"时退化为 0（归一化后的均值），
用 MSE 强制拟合这些噪声伪标签反而导致模型回归均值，拉低 R²。
"""

import tensorflow as tf
import numpy as np


def compute_consistency_weights(mask_orig, mask_corrupted_list, sigma_c=1.0):
    """
    计算一致性损失的自适应权重（专利步骤S2）

    专利公式：
        w^(q) = exp(-||M^(q) - M^(0)||_1 / σ_c²)

    破坏程度越大（差异越大），权重越小
    这使得模型更关注轻度破坏的样本，避免过度依赖严重损坏的表示

    Args:
        mask_orig: [batch, time, features] - 原始掩码 M^(0)
        mask_corrupted_list: list of [batch, time, features] - Q个增强掩码 M^(q)
        sigma_c: float - 高斯核带宽参数

    Returns:
        weights: [Q] - 每个增强样本的权重
    """
    weights = []

    for mask_corrupted in mask_corrupted_list:
        # 计算L1距离（破坏程度）
        diff = tf.abs(mask_corrupted - mask_orig)
        l1_distance = tf.reduce_sum(diff)

        # 高斯核权重
        weight = tf.exp(-l1_distance / (sigma_c ** 2))
        weights.append(float(weight.numpy()))

    return weights


def bernoulli_corruption_with_weight(mask, p_drop=0.1, sigma_c=1.0):
    """
    带权重计算的Bernoulli损坏（专利步骤S2）

    生成增强掩码并计算对应的一致性权重

    Args:
        mask: [batch, time, features] - 原始掩码
        p_drop: float - 丢弃概率
        sigma_c: float - 权重计算的带宽参数

    Returns:
        corrupted_mask: [batch, time, features] - 损坏后的掩码
        weight: float - 该损坏版本的权重
    """
    # 生成 Bernoulli 随机变量
    bernoulli_mask = tf.cast(
        tf.random.uniform(tf.shape(mask)) > p_drop,
        tf.float32
    )

    # 只在原本有观测值的位置进行损坏
    corrupted_mask = mask * bernoulli_mask

    # 计算权重
    diff = tf.abs(corrupted_mask - mask)
    l1_distance = tf.reduce_sum(diff)
    weight = float(tf.exp(-l1_distance / (sigma_c ** 2)).numpy())

    return corrupted_mask, weight


def generate_augmented_masks(mask, Q=5, p_drop=0.1, sigma_c=1.0):
    """
    生成Q个增强掩码及其权重（专利步骤S2）

    专利公式：
        M^(q) = M^(0) ⊙ B^(q), B_{t,n}^(q) ~ Bernoulli(1 - p_drop)
        w^(q) = exp(-||M^(q) - M^(0)||_1 / σ_c²)

    Args:
        mask: [batch, time, features] - 原始掩码 M^(0)
        Q: int - 增强样本数量
        p_drop: float - 丢弃概率
        sigma_c: float - 权重计算的带宽参数

    Returns:
        corrupted_masks: list of [batch, time, features] - Q个增强掩码
        weights: [Q] - 对应的权重
    """
    corrupted_masks = []
    weights = []

    for _ in range(Q):
        corrupted_mask, weight = bernoulli_corruption_with_weight(
            mask, p_drop=p_drop, sigma_c=sigma_c
        )
        corrupted_masks.append(corrupted_mask)
        weights.append(weight)

    return corrupted_masks, weights


def reconstruction_loss(x_true, x_pred, mask):
    """
    重建损失 L_recon (Algorithm 1, line 16)
    
    L_recon = (1/|M_i|) || (X_i - X̂_i) ⊙ M_i ||_F^2
    
    只在观测位置（M=1）计算均方误差
    
    Args:
        x_true: [batch, time, features] - 真实观测值
        x_pred: [batch, time, features] - 预测值
        mask: [batch, time, features] - 掩码矩阵 (1=observed, 0=missing)
        
    Returns:
        loss: scalar - 重建损失
    """
    # 只在观测位置计算误差
    diff = (x_true - x_pred) * mask
    
    # Frobenius 范数的平方
    squared_error = tf.reduce_sum(tf.square(diff))
    
    # 除以观测值的数量进行归一化
    n_observed = tf.reduce_sum(mask) + 1e-8  # 避免除零
    
    loss = squared_error / n_observed

    return loss


def missing_position_loss(x_pred, x_knn_space, x_knn_time, mask, alpha_space=0.5):
    """
    缺失位置伪标签损失 L_missing（方案1：解决"回归均值"问题）

    核心思想：
    - 缺失位置没有真实标签，但有KNN初始化值作为"伪标签"
    - 用空间KNN和时间KNN初始化值的加权组合监督缺失位置的预测
    - 这样缺失位置也有梯度信号，不会退化为均值解

    L_missing = (1/|1-M|) || (X_pred - X_pseudo) ⊙ (1-M) ||_F^2

    其中 X_pseudo = α * X_space_init + (1-α) * X_time_init

    Args:
        x_pred: [batch, time, features] - 模型预测值
        x_knn_space: [batch, time, features] - 空间KNN初始化值
        x_knn_time: [batch, time, features] - 时间KNN初始化值
        mask: [batch, time, features] - 掩码 (1=observed, 0=missing)
        alpha_space: float - 空间KNN的权重 (0-1)

    Returns:
        loss: scalar - 缺失位置损失
    """
    # 计算伪标签：空间和时间KNN的加权组合
    x_pseudo = alpha_space * x_knn_space + (1.0 - alpha_space) * x_knn_time

    # 缺失位置掩码
    missing_mask = 1.0 - mask

    # 只在缺失位置计算误差
    diff = (x_pred - x_pseudo) * missing_mask

    # Frobenius 范数的平方
    squared_error = tf.reduce_sum(tf.square(diff))

    # 除以缺失值的数量进行归一化
    n_missing = tf.reduce_sum(missing_mask) + 1e-8

    loss = squared_error / n_missing

    return loss


def consistency_loss(z_orig, z_corrupted_list, weights):
    """
    一致性损失 L_consist (Algorithm 1, line 17)

    专利公式（第36-40行）：
    L_consist = Σ_{k=1}^K w^(k) || Z_i^(k) - Z_i^orig ||_F^2 / Σ_{k=1}^K w^(k)

    注意：权重应该归一化，而不是简单平均。损坏越重，权重越小。

    Args:
        z_orig: [batch, time, latent] - 原始潜在表示
        z_corrupted_list: list of [batch, time, latent] - K个损坏版本的表示
        weights: [K] - 每个损坏版本的权重 w^(k)

    Returns:
        loss: scalar - 一致性损失
    """
    K = len(z_corrupted_list)

    if K == 0:
        return tf.constant(0.0, dtype=tf.float32)

    total_loss = tf.constant(0.0, dtype=tf.float32)
    for k, z_corrupted in enumerate(z_corrupted_list):
        # 计算与原始表示的距离
        diff = z_corrupted - z_orig
        # 使用 reduce_mean 进行规范化，避免损失过大
        mean_squared_error = tf.reduce_mean(tf.square(diff))

        # 应用权重
        w_k = float(weights[k]) if weights is not None else 1.0
        total_loss += w_k * mean_squared_error

    # ✅ 修复：权重归一化（而不是简单平均）
    # 专利要求：Σ w^(k) * MSE / Σ w^(k)
    if weights is not None:
        # 确保权重类型为tf.float32，避免类型不匹配
        weights_tf = tf.constant(weights, dtype=tf.float32)
        weight_sum = tf.reduce_sum(weights_tf) + 1e-8
        loss = total_loss / weight_sum
    else:
        # 如果没有权重，则简单平均
        loss = total_loss / tf.constant(K + 1e-8, dtype=tf.float32)

    return loss


def spatial_preservation_loss(z_i, z_neighbors_spatial, weights_spatial=None, mask=None):
    """
    空间邻域保持损失 L_space (Algorithm 1, line 18-19)

    专利公式（第231-234行）：
    L_space = Σ_{t=1}^T Σ_{s∈N_i^space} w_{i,s}^space || z_{i,t} - z_{i,s} ||_2^2

    ✅ 修复：使用原始核函数权重 w^space（来自部分距离策略），
             而不是在潜在空间中重新计算权重

    确保在嵌入空间中保持变量间的相似性关系

    Args:
        z_i: [batch, time, latent] - 当前样本的潜在表示
        z_neighbors_spatial: [batch, time, k, latent] - 空间近邻的表示
        weights_spatial: [batch, time, k] - 原始空间权重（可选）
        mask: [batch, time, features] - 掩码（不使用，保留向后兼容）

    Returns:
        loss: scalar - 空间保持损失
    """
    # z_i: [batch, time, latent]
    # z_neighbors: [batch, time, k, latent]

    # 扩展 z_i 以匹配 neighbors 的维度
    z_i_expanded = tf.expand_dims(z_i, axis=2)  # [batch, time, 1, latent]

    # 计算距离
    diff = z_neighbors_spatial - z_i_expanded  # [batch, time, k, latent]
    squared_distances = tf.reduce_sum(tf.square(diff), axis=-1)  # [batch, time, k]

    # ✅ 修复：使用原始权重（如果提供）
    if weights_spatial is not None:
        weights = weights_spatial  # 使用从部分距离策略得到的权重
    else:
        # 回退：使用距离的倒数，softmax 归一化
        weights = tf.nn.softmax(-squared_distances, axis=-1)  # [batch, time, k]

    # 加权距离
    weighted_distances = weights * squared_distances  # [batch, time, k]

    # 正确的归一化：对所有维度求平均，而不是求和
    # 这样可以保持损失与 L_recon 的数量级一致
    loss = tf.reduce_mean(weighted_distances)

    return loss


def temporal_preservation_loss(z_n, z_neighbors_temporal, weights_temporal=None, mask=None):
    """
    时间邻域保持损失 L_time (Algorithm 1, line 19-20)

    专利公式（第235-237行）：
    L_time = Σ_{n=1}^N Σ_{m∈N_n^time} w_{n,m}^time || z_{i,n} - z_{i,m} ||_2^2

    ✅ 修复：支持变量级表示
    - 当使用变量级映射时，z_n是变量级表示 [batch, features, latent]
    - 当使用传统方法时，z_n是时间步级表示 [batch, time, latent]

    确保在嵌入空间中保持时间序列的局部结构

    Args:
        z_n: [batch, time, latent] 或 [batch, features, latent] - 潜在表示
        z_neighbors_temporal: [batch, time, k, latent] 或 [batch, features, k, latent] - 时间近邻的表示
        weights_temporal: [batch, time, k] 或 [batch, features, k] - 原始时间权重（可选）
        mask: [batch, time, features] - 掩码（不使用，保留向后兼容）

    Returns:
        loss: scalar - 时间保持损失
    """
    if z_neighbors_temporal is None:
        # 如果没有时间邻居信息，返回0（例如使用变量级映射但未提供邻居信息）
        return tf.constant(0.0, dtype=tf.float32)

    # z_n: [batch, dimension, latent] (dimension可以是time或features)
    # z_neighbors: [batch, dimension, k, latent]

    z_n_expanded = tf.expand_dims(z_n, axis=2)  # [batch, dimension, 1, latent]

    # 计算距离
    diff = z_neighbors_temporal - z_n_expanded  # [batch, dimension, k, latent]
    squared_distances = tf.reduce_sum(tf.square(diff), axis=-1)  # [batch, dimension, k]

    # ✅ 修复：使用原始权重（如果提供）
    if weights_temporal is not None:
        weights = weights_temporal  # 使用从部分距离策略得到的权重
    else:
        # 回退：使用距离的倒数，softmax 归一化
        weights = tf.nn.softmax(-squared_distances, axis=-1)  # [batch, dimension, k]

    # 加权距离
    weighted_distances = weights * squared_distances

    # 正确的归一化：对所有维度求平均
    loss = tf.reduce_mean(weighted_distances)

    return loss


def total_loss(x_true, x_pred, mask, z_orig, z_corrupted_list,
               neighborhood_info=None,
               lambda1=1.0, lambda2=0.1, lambda3=0.1, lambda4=0.0,
               corruption_weights=None,
               x_knn_space=None, x_knn_time=None, alpha_space=0.5):
    """
    总损失函数 L_total (Algorithm 1, line 20-21)

    L_total = L_recon + λ1·L_consist + λ2·L_space + λ3·L_time

    注意：L_missing（伪标签损失）已被禁用（lambda4 默认为 0）。
    实验证明 KNN 伪标签会导致模型回归均值，拉低缺失位置 R²。

    Args:
        x_true: 真实数据
        x_pred: 重建数据
        mask: 掩码
        z_orig: 原始潜在表示
        z_corrupted_list: 损坏版本的潜在表示列表
        neighborhood_info: 邻域信息字典
        lambda1, lambda2, lambda3: 损失权重
        lambda4: [已禁用] 伪标签损失权重，默认为 0
        corruption_weights: 损坏版本的权重
        x_knn_space: [已禁用] 空间KNN初始化值
        x_knn_time: [已禁用] 时间KNN初始化值
        alpha_space: [已禁用] 空间KNN在伪标签中的权重

    Returns:
        total: 总损失
        losses_dict: 各组成部分的损失（用于监控）
    """
    # 1. 重建损失（仅观测位置）
    L_recon = reconstruction_loss(x_true, x_pred, mask)

    # 2. 一致性损失
    L_consist = consistency_loss(z_orig, z_corrupted_list, corruption_weights)

    # 3. 空间保持损失
    L_space = tf.constant(0.0, dtype=tf.float32)
    if neighborhood_info is not None and 'z_neighbors_space' in neighborhood_info:
        z_neighbors_spatial = neighborhood_info['z_neighbors_space']
        weights_spatial = neighborhood_info.get('weights_space', None)
        L_space = spatial_preservation_loss(
            z_orig, z_neighbors_spatial,
            weights_spatial=weights_spatial,
            mask=mask
        )

    # 4. 时间保持损失
    L_time = tf.constant(0.0, dtype=tf.float32)
    if neighborhood_info is not None:
        if 'z_var' in neighborhood_info and neighborhood_info['z_var'] is not None:
            z_var = neighborhood_info['z_var']
            z_var_neighbors = neighborhood_info.get('z_var_neighbors', None)
            weights_var = neighborhood_info.get('weights_var', None)

            if z_var_neighbors is not None:
                L_time = temporal_preservation_loss(
                    z_var, z_var_neighbors,
                    weights_temporal=weights_var,
                    mask=mask
                )
        elif 'z_neighbors_time' in neighborhood_info:
            z_neighbors_temporal = neighborhood_info['z_neighbors_time']
            weights_temporal = neighborhood_info.get('weights_time', None)

            if z_neighbors_temporal is not None:
                L_time = temporal_preservation_loss(
                    z_orig, z_neighbors_temporal,
                    weights_temporal=weights_temporal,
                    mask=mask
                )

    # 5. L_missing 已禁用（实验证明有害）
    # 保留变量以兼容旧代码，但始终为 0
    L_missing = tf.constant(0.0, dtype=tf.float32)

    # 总损失（不包含 L_missing）
    total = L_recon + lambda1 * L_consist + lambda2 * L_space + lambda3 * L_time

    # 返回详细信息用于监控
    losses_dict = {
        'total': total,
        'recon': L_recon,
        'consist': L_consist,
        'space': L_space,
        'time': L_time,
        'missing': L_missing  # 新增
    }

    return total, losses_dict


def bernoulli_corruption(mask, p_drop=0.1):
    """
    Bernoulli 损坏 (Algorithm 1, line 6)
    
    M_i^(k) ← M_i ⊙ Bernoulli(1 - p_drop)
    
    随机将部分观测值标记为缺失，用于训练时的鲁棒性增强
    
    Args:
        mask: [batch, time, features] - 原始掩码
        p_drop: float - 随机丢弃的概率
        
    Returns:
        corrupted_mask: [batch, time, features] - 损坏后的掩码
    """
    # 生成 Bernoulli 随机变量
    bernoulli_mask = tf.cast(
        tf.random.uniform(tf.shape(mask)) > p_drop, 
        tf.float32
    )
    
    # 只在原本有观测值的位置进行损坏
    corrupted_mask = mask * bernoulli_mask
    
    return corrupted_mask
