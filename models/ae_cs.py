"""
AE-CS: AutoEncoder based on Coherent denoising and Spatio-temporal
       neighborhood-preserving embedding

核心模型实现（符合专利步骤S2-S6）：
1. 第一编码器（一致性去噪编码器）: 零值填充输入
2. 第二编码器（空间邻域编码器）: 空间K近邻均值初始化输入
3. 第三编码器（时间邻域编码器）: 时间K近邻均值初始化输入
4. 解码器: 将融合后的潜在表示映射回原始数据空间
5. 门控网络: 基于缺失率的自适应融合
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model


# 自定义激活函数: 1 - e^(-x²)
# 这是一个高斯型激活函数，特点：
# - 输出范围: [0, 1)
# - x=0时输出0，|x|→∞时输出趋近1
# - 平滑、处处可导
@tf.keras.utils.register_keras_serializable(package='Custom', name='gaussian_activation')
def gaussian_activation(x):
    """高斯激活函数: f(x) = 1 - exp(-x²)"""
    return 1.0 - tf.exp(-tf.square(x))


def compute_spatial_knn_init(X, mask, k=5):
    """
    空间K近邻均值初始化（专利步骤S5）

    对每个缺失位置，用其空间邻域内观测值的加权均值进行填充
    观测位置保持原始值不变

    Args:
        X: [batch, time, features] - 原始数据（缺失位置可能为任意值）
        mask: [batch, time, features] - 掩码矩阵 (1=观测, 0=缺失)
        k: int - 邻域数量

    Returns:
        X_init: [batch, time, features] - 空间KNN初始化后的数据
    """
    batch_size, time_steps, n_features = X.shape
    X_np = X.numpy()
    M = mask.numpy()

    # 初始化：观测位置保持原始值，缺失位置先填充为0
    X_init = X_np * M  # 缺失位置为0

    for b in range(batch_size):
        X_b = X_np[b]    # [time, features] - 原始数据用于计算距离
        M_b = M[b]       # [time, features]

        # 对每个时间步，基于空间邻域（其他时间步）进行填充
        for t in range(time_steps):
            # 找到缺失的变量
            missing_vars = np.where(M_b[t, :] == 0)[0]

            if len(missing_vars) == 0:
                continue

            # 计算时间步t与其他时间步的空间距离
            distances = np.full(time_steps, np.inf)

            for s in range(time_steps):
                if s == t:
                    continue

                # 找到t和s共同观测的变量
                common_vars = np.where((M_b[t, :] == 1) & (M_b[s, :] == 1))[0]

                if len(common_vars) > 0:
                    # 归一化距离
                    diff = X_b[t, common_vars] - X_b[s, common_vars]
                    distances[s] = np.sqrt(np.sum(diff ** 2) / len(common_vars))

            # 找k个最近邻
            k_actual = min(k, int(np.sum(distances < np.inf)))
            if k_actual == 0:
                # 无法找到邻居，缺失位置保持为0
                continue

            k_indices = np.argsort(distances)[:k_actual]
            k_distances = distances[k_indices]

            # 高斯核权重
            sigma = np.median(k_distances[k_distances < np.inf]) + 1e-8
            weights = np.exp(-k_distances ** 2 / (sigma ** 2))
            weights = weights / (np.sum(weights) + 1e-8)

            # 对每个缺失变量进行填充
            for v in missing_vars:
                # 找到邻居中观测了变量v的时间步
                valid_neighbors = [k_indices[i] for i in range(k_actual)
                                   if M_b[k_indices[i], v] == 1]
                valid_weights = [weights[i] for i in range(k_actual)
                                 if M_b[k_indices[i], v] == 1]

                if len(valid_neighbors) > 0:
                    valid_weights = np.array(valid_weights)
                    valid_weights = valid_weights / (np.sum(valid_weights) + 1e-8)
                    X_init[b, t, v] = np.sum([valid_weights[i] * X_b[valid_neighbors[i], v]
                                               for i in range(len(valid_neighbors))])
                # 否则保持为0（已在初始化时设置）

    return tf.constant(X_init, dtype=tf.float32)


def compute_temporal_knn_init(X, mask, k=5):
    """
    时间K近邻均值初始化（专利步骤S5）

    对每个缺失位置，用其时间邻域内观测值的加权均值进行填充
    基于变量间相似性进行填充，观测位置保持原始值不变

    Args:
        X: [batch, time, features] - 原始数据
        mask: [batch, time, features] - 掩码矩阵
        k: int - 邻域数量

    Returns:
        X_init: [batch, time, features] - 时间KNN初始化后的数据
    """
    batch_size, time_steps, n_features = X.shape
    X_np = X.numpy()
    M = mask.numpy()

    # 初始化：观测位置保持原始值，缺失位置先填充为0
    X_init = X_np * M  # 缺失位置为0

    for b in range(batch_size):
        X_b = X_np[b]    # [time, features] - 原始数据用于计算距离
        M_b = M[b]       # [time, features]

        # 计算变量间的距离矩阵
        var_distances = np.full((n_features, n_features), np.inf)

        for j in range(n_features):
            for m in range(n_features):
                if j == m:
                    continue

                # 找到共同观测的时间点
                common_times = np.where((M_b[:, j] == 1) & (M_b[:, m] == 1))[0]

                if len(common_times) > 0:
                    diff = X_b[common_times, j] - X_b[common_times, m]
                    var_distances[j, m] = np.sqrt(np.sum(diff ** 2) / len(common_times))

        # 对每个变量，找k个最相似的变量
        for v in range(n_features):
            k_actual = min(k, int(np.sum(var_distances[v, :] < np.inf)))
            if k_actual == 0:
                continue

            neighbor_vars = np.argsort(var_distances[v, :])[:k_actual]
            neighbor_dists = var_distances[v, neighbor_vars]

            # 高斯核权重
            sigma = np.median(neighbor_dists[neighbor_dists < np.inf]) + 1e-8
            var_weights = np.exp(-neighbor_dists ** 2 / (sigma ** 2))
            var_weights = var_weights / (np.sum(var_weights) + 1e-8)

            # 对变量v的每个缺失时间点进行填充
            missing_times = np.where(M_b[:, v] == 0)[0]

            for t in missing_times:
                # 找到该时间点上被观测的邻居变量
                valid_neighbors = [neighbor_vars[i] for i in range(k_actual)
                                   if M_b[t, neighbor_vars[i]] == 1]
                valid_weights = [var_weights[i] for i in range(k_actual)
                                 if M_b[t, neighbor_vars[i]] == 1]

                if len(valid_neighbors) > 0:
                    valid_weights = np.array(valid_weights)
                    valid_weights = valid_weights / (np.sum(valid_weights) + 1e-8)
                    X_init[b, t, v] = np.sum([valid_weights[i] * X_b[t, valid_neighbors[i]]
                                               for i in range(len(valid_neighbors))])
                # 否则保持为0（已在初始化时设置）

    return tf.constant(X_init, dtype=tf.float32)


class Encoder(Model):
    """
    编码器网络 f_θ
    
    输入: X ⊙ M (observed values) 和 M (mask matrix)
    输出: Z (latent representation)
    
    架构: 使用 GRU 或 LSTM 处理时间序列
    """
    def __init__(self, latent_dim=64, hidden_units=128, name='encoder',
                 dropout_rate=0.2, l2_reg=0.001):
        super(Encoder, self).__init__(name=name)

        self.latent_dim = latent_dim
        self.hidden_units = hidden_units

        # LSTM layers for temporal encoding (replaced GRU due to cuDNN stability issues)
        self.gru1 = layers.LSTM(hidden_units, return_sequences=True, name='lstm1')
        self.gru2 = layers.LSTM(hidden_units, return_sequences=True, name='lstm2')

        # Dropout for regularization
        self.dropout1 = layers.Dropout(dropout_rate, name='dropout1')
        self.dropout2 = layers.Dropout(dropout_rate, name='dropout2')

        # Fully connected layer to latent space with L2 regularization
        # 实验1：移除潜在层高斯激活，改用线性激活（None）
        # 原代码: activation=gaussian_activation
        self.dense_latent = layers.Dense(
            latent_dim,
            activation=None,  # 线性激活
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='latent'
        )

        # Batch normalization
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.bn2 = layers.BatchNormalization(name='bn2')
        
    def call(self, x, mask, training=False):
        """
        Args:
            x: [batch_size, time_steps, n_features] - 输入数据
            mask: [batch_size, time_steps, n_features] - 掩码矩阵 (1=观测, 0=缺失)
            training: bool - 是否为训练模式

        Returns:
            z: [batch_size, time_steps, latent_dim] - 潜在表示
        """
        # 关键修复：将缺失位置的值置为0，防止数据泄露
        # 原代码: x_masked = tf.concat([x, mask], axis=-1)  ← 模型能看到缺失位置的真实值
        # 修复后: 先用掩码过滤，确保缺失位置为0
        x_observed = x * mask  # 缺失位置 (mask=0) 的值变为0
        x_masked = tf.concat([x_observed, mask], axis=-1)

        # GRU encoding
        h1 = self.gru1(x_masked, training=training)
        h1 = self.bn1(h1, training=training)
        h1 = self.dropout1(h1, training=training)

        h2 = self.gru2(h1, training=training)
        h2 = self.bn2(h2, training=training)
        h2 = self.dropout2(h2, training=training)

        # Project to latent space
        z = self.dense_latent(h2)

        return z


class Decoder(Model):
    """
    解码器网络 g_φ
    
    输入: Z^fused (融合后的潜在表示)
    输出: X̂ (重建的完整数据)
    
    架构: 使用 GRU 或 LSTM 进行时间序列重建
    """
    def __init__(self, n_features, hidden_units=128, name='decoder',
                 dropout_rate=0.2, l2_reg=0.001):
        super(Decoder, self).__init__(name=name)

        self.n_features = n_features
        self.hidden_units = hidden_units

        # LSTM layers for temporal decoding (replaced GRU due to cuDNN stability issues)
        self.gru1 = layers.LSTM(hidden_units, return_sequences=True, name='lstm1')
        self.gru2 = layers.LSTM(hidden_units, return_sequences=True, name='lstm2')

        # Dropout for regularization
        self.dropout1 = layers.Dropout(dropout_rate, name='dropout1')
        self.dropout2 = layers.Dropout(dropout_rate, name='dropout2')

        # Output layer with L2 regularization
        self.dense_out = layers.Dense(
            n_features,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='output'
        )

        # Batch normalization
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.bn2 = layers.BatchNormalization(name='bn2')
        
    def call(self, z, training=False):
        """
        Args:
            z: [batch_size, time_steps, latent_dim] - 潜在表示
            training: bool - 是否为训练模式
            
        Returns:
            x_hat: [batch_size, time_steps, n_features] - 重建的数据
        """
        # GRU decoding
        h1 = self.gru1(z, training=training)
        h1 = self.bn1(h1, training=training)
        h1 = self.dropout1(h1, training=training)

        h2 = self.gru2(h1, training=training)
        h2 = self.bn2(h2, training=training)
        h2 = self.dropout2(h2, training=training)

        # Reconstruct to original feature space
        x_hat = self.dense_out(h2)

        return x_hat


class GatingNetwork(Model):
    """
    门控网络 - 用于计算自适应融合权重 α (符合专利步骤S5)

    专利公式:
        α = softmax(FC([GAP(Z_orig), GAP(Z_space), GAP(Z_time), ρ]))

    关键改进:
        1. 添加全局平均池化(GAP)操作
        2. 添加缺失率ρ作为先验输入
        3. 输出全局权重[batch, 3]而非逐时间步权重[batch, time, 3]

    这使模型能够根据数据稀疏程度自适应调整融合策略:
        - 数据完整(ρ小) → α₁大 → 优先使用原始特征
        - 数据稀疏(ρ大) → α₂、α₃大 → 依赖邻域推断
    """
    def __init__(self, latent_dim=64, name='gating_network',
                 dropout_rate=0.2, l2_reg=0.001):
        super(GatingNetwork, self).__init__(name=name)

        self.latent_dim = latent_dim

        # 输入维度: latent*3 (三个GAP) + 1 (缺失率ρ)
        # Attention layers with L2 regularization
        # 实验2结论：gaussian激活在门控网络中表现更好（R²=0.345 vs tanh的0.171）
        self.dense1 = layers.Dense(
            latent_dim * 2,
            activation=gaussian_activation,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='dense1'
        )
        self.dense2 = layers.Dense(
            latent_dim,
            activation=gaussian_activation,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='dense2'
        )

        # Dropout for regularization
        self.dropout = layers.Dropout(dropout_rate, name='dropout')

        # Output 3 weights (for orig, space, time)
        self.dense_alpha = layers.Dense(3, activation='softmax', name='alpha')

    def call(self, z_orig, z_space, z_time, missing_rate, training=False):
        """
        Args:
            z_orig: [batch_size, time_steps, latent_dim] - 原始表示
            z_space: [batch_size, time_steps, latent_dim] - 空间表示
            z_time: [batch_size, time_steps, latent_dim] - 时间表示
            missing_rate: [batch_size] or scalar - 缺失率ρ (每个样本的缺失率)
            training: bool - 是否为训练模式

        Returns:
            alpha: [batch_size, 3] - 全局融合权重
            z_fused: [batch_size, time_steps, latent_dim] - 融合后的表示
        """
        # 步骤1: 全局平均池化 (GAP) - 将时间维度压缩
        gap_orig = tf.reduce_mean(z_orig, axis=1)    # [batch, latent]
        gap_space = tf.reduce_mean(z_space, axis=1)  # [batch, latent]
        gap_time = tf.reduce_mean(z_time, axis=1)    # [batch, latent]

        # 步骤2: 拼接GAP结果和缺失率ρ
        # 确保missing_rate是正确的形状
        if isinstance(missing_rate, (int, float)):
            # 标量缺失率，扩展到batch维度
            batch_size = tf.shape(z_orig)[0]
            missing_rate = tf.fill([batch_size], float(missing_rate))
        elif len(missing_rate.shape) == 0:
            # 0维张量，扩展到batch维度
            batch_size = tf.shape(z_orig)[0]
            missing_rate = tf.fill([batch_size], missing_rate)

        missing_rate_expanded = tf.expand_dims(missing_rate, axis=-1)  # [batch, 1]
        combined = tf.concat([gap_orig, gap_space, gap_time, missing_rate_expanded], axis=-1)
        # combined shape: [batch, latent*3 + 1]

        # 步骤3: 计算全局注意力权重
        h1 = self.dense1(combined)  # [batch, latent*2]
        h2 = self.dense2(h1)        # [batch, latent]
        h2 = self.dropout(h2, training=training)
        alpha = self.dense_alpha(h2)  # [batch, 3]

        # 步骤4: 扩展权重到时间维度并融合
        # alpha: [batch, 3] -> [batch, 1, 3] -> 广播到 [batch, time, latent]
        alpha_expanded = tf.expand_dims(alpha, axis=1)  # [batch, 1, 3]

        # 提取三个权重分量
        alpha_1 = tf.expand_dims(alpha_expanded[:, :, 0], axis=-1)  # [batch, 1, 1]
        alpha_2 = tf.expand_dims(alpha_expanded[:, :, 1], axis=-1)
        alpha_3 = tf.expand_dims(alpha_expanded[:, :, 2], axis=-1)

        # 融合：广播机制会自动将[batch,1,1]扩展到[batch,time,latent]
        z_fused = alpha_1 * z_orig + alpha_2 * z_space + alpha_3 * z_time

        return alpha, z_fused


class AECS(Model):
    """
    完整的 AE-CS 模型（符合专利步骤S2-S6）

    专利核心架构 - 三编码器一解码器：
    1. 第一编码器（一致性去噪编码器）: 零值填充输入，提取 Z_orig
    2. 第二编码器（空间邻域编码器）: 空间K近邻均值初始化输入，提取 Z_space
    3. 第三编码器（时间邻域编码器）: 时间K近邻均值初始化输入，提取 Z_time
    4. 解码器: 将融合后的 Z_fused 映射回原始数据空间
    5. 门控网络: 基于缺失率ρ的自适应融合

    关键改进（相对于旧版本）：
    - Z_space 和 Z_time 现在来自独立编码器，而不是邻域聚合
    - 实现了三种差异化输入初始化策略
    - 邻域模块仅用于计算损失函数中的邻域保持损失
    """
    def __init__(self, n_features, latent_dim=64, hidden_units=128,
                 k_spatial=5, k_temporal=5, use_partial_distance=True, use_variable_mapping=True,
                 dropout_rate=0.2, l2_reg=0.001, name='ae_cs'):
        super(AECS, self).__init__(name=name)

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.k_spatial = k_spatial
        self.k_temporal = k_temporal

        # ========== 专利步骤S2：构建三编码器-单解码器网络框架 ==========

        # 第一编码器（一致性去噪编码器）：输入为零值填充的数据
        self.encoder_orig = Encoder(
            latent_dim, hidden_units,
            dropout_rate=dropout_rate, l2_reg=l2_reg,
            name='encoder_orig'
        )

        # 第二编码器（空间邻域编码器）：输入为空间K近邻均值初始化的数据
        self.encoder_space = Encoder(
            latent_dim, hidden_units,
            dropout_rate=dropout_rate, l2_reg=l2_reg,
            name='encoder_space'
        )

        # 第三编码器（时间邻域编码器）：输入为时间K近邻均值初始化的数据
        self.encoder_time = Encoder(
            latent_dim, hidden_units,
            dropout_rate=dropout_rate, l2_reg=l2_reg,
            name='encoder_time'
        )

        # 解码器：将融合后的潜在表示映射回原始数据空间
        self.decoder = Decoder(
            n_features, hidden_units,
            dropout_rate=dropout_rate, l2_reg=l2_reg
        )

        # 门控网络：自适应融合
        self.gating_network = GatingNetwork(
            latent_dim,
            dropout_rate=dropout_rate, l2_reg=l2_reg
        )

        # 邻域模块（用于计算邻域保持损失，不用于生成Z_space和Z_time）
        from .neighborhood import NeighborhoodModule
        self.neighborhood_module = NeighborhoodModule(
            k_spatial, k_temporal, use_partial_distance, use_variable_mapping
        )

        # 缓存初始化后的数据，避免重复计算
        self._cached_x_space_init = None
        self._cached_x_time_init = None
        self._cache_valid = False

    def _prepare_inputs(self, x, mask):
        """
        准备三个编码器的输入（专利步骤S5的输入初始化策略）

        Args:
            x: [batch, time, features] - 原始数据
            mask: [batch, time, features] - 掩码

        Returns:
            x_zero: 零值填充的输入（第一编码器）
            x_space_init: 空间KNN均值初始化的输入（第二编码器）
            x_time_init: 时间KNN均值初始化的输入（第三编码器）
        """
        # 第一编码器输入：零值填充（缺失位置为0）
        x_zero = x * mask

        # 第二编码器输入：空间K近邻均值初始化
        x_space_init = compute_spatial_knn_init(x, mask, k=self.k_spatial)

        # 第三编码器输入：时间K近邻均值初始化
        x_time_init = compute_temporal_knn_init(x, mask, k=self.k_temporal)

        return x_zero, x_space_init, x_time_init

    def call(self, x, mask, training=False, return_all=False):
        """
        完整的前向传播（符合专利步骤S3-S6）

        Args:
            x: [batch_size, time_steps, n_features] - 输入数据
            mask: [batch_size, time_steps, n_features] - 掩码
            training: bool - 训练模式
            return_all: bool - 是否返回所有中间结果

        Returns:
            如果return_all=False:
                x_filled: 掩码融合后的完整数据（推理阶段）或 x_hat（训练阶段）
            如果return_all=True:
                outputs_dict: 包含所有中间结果的字典
        """
        # ========== 步骤S1：计算整体缺失率ρ ==========
        missing_rate = 1.0 - tf.reduce_mean(mask, axis=[1, 2])  # [batch]

        # ========== 步骤S5：差异化输入初始化 ==========
        x_zero, x_space_init, x_time_init = self._prepare_inputs(x, mask)

        # ========== 步骤S3：三路编码 ==========

        # 第一编码器：一致性去噪编码，输入零值填充数据
        z_orig = self.encoder_orig(x_zero, mask, training=training)

        # 第二编码器：空间邻域编码，输入空间KNN初始化数据
        z_space = self.encoder_space(x_space_init, mask, training=training)

        # 第三编码器：时间邻域编码，输入时间KNN初始化数据
        z_time = self.encoder_time(x_time_init, mask, training=training)

        # ========== 步骤S4：计算邻域信息（用于损失函数）==========
        # 注意：这里的邻域计算仅用于计算空间/时间保持损失
        # Z_space和Z_time已经由独立编码器生成
        _, _, neighborhood_info = self.neighborhood_module.compute_neighborhood_embeddings(
            x, z_orig, mask
        )

        # ========== 步骤S4：自适应融合 ==========
        alpha, z_fused = self.gating_network(
            z_orig, z_space, z_time, missing_rate, training=training
        )

        # ========== 解码重构 ==========
        x_hat = self.decoder(z_fused, training=training)

        # ========== 步骤S6：掩码融合输出 ==========
        if training:
            x_filled = x_hat
        else:
            x_filled = mask * x + (1.0 - mask) * x_hat

        if return_all:
            return {
                'x_hat': x_hat,
                'x_filled': x_filled,
                'z_orig': z_orig,
                'z_space': z_space,
                'z_time': z_time,
                'z_fused': z_fused,
                'alpha': alpha,
                'missing_rate': missing_rate,
                'neighborhood_info': neighborhood_info,
                # 额外信息用于调试
                'x_space_init': x_space_init,
                'x_time_init': x_time_init
            }
        else:
            return x_filled

    def encode(self, x, mask, training=False):
        """使用第一编码器编码（零值填充输入）"""
        x_zero = x * mask
        return self.encoder_orig(x_zero, mask, training=training)

    def encode_all(self, x, mask, training=False):
        """使用所有三个编码器编码，返回三路潜在表示"""
        x_zero, x_space_init, x_time_init = self._prepare_inputs(x, mask)
        z_orig = self.encoder_orig(x_zero, mask, training=training)
        z_space = self.encoder_space(x_space_init, mask, training=training)
        z_time = self.encoder_time(x_time_init, mask, training=training)
        return z_orig, z_space, z_time

    def decode(self, z, training=False):
        """仅解码"""
        return self.decoder(z, training=training)

    def compute_neighborhoods(self, x, z_orig, mask):
        """
        计算邻域信息（用于损失函数）

        注意：在新架构中，Z_space和Z_time由独立编码器生成，
        此方法仅返回邻域权重和索引用于计算邻域保持损失

        Args:
            x: [batch, time, features] - 原始观测数据
            z_orig: [batch, time, latent] - 原始潜在表示
            mask: [batch, time, features] - 掩码

        Returns:
            z_space_agg, z_time_agg, info - 邻域聚合结果和详细信息
        """
        return self.neighborhood_module.compute_neighborhood_embeddings(x, z_orig, mask)

    def fuse_representations(self, z_orig, z_space, z_time, missing_rate, training=False):
        """
        融合三种表示

        Args:
            z_orig: [batch, time, latent] - 第一编码器输出
            z_space: [batch, time, latent] - 第二编码器输出
            z_time: [batch, time, latent] - 第三编码器输出
            missing_rate: [batch] or scalar - 缺失率ρ
            training: bool

        Returns:
            alpha: [batch, 3] - 融合权重
            z_fused: [batch, time, latent] - 融合后的表示
        """
        return self.gating_network(z_orig, z_space, z_time, missing_rate, training=training)


# ========== 保持向后兼容的旧版AECS ==========
class AECS_Legacy(Model):
    """
    [已废弃] 旧版 AE-CS 模型（单编码器 + 邻域聚合）

    保留此类仅为向后兼容。新代码应使用 AECS 类。

    与新版AECS的区别：
    - 旧版：1个编码器，Z_space和Z_time通过邻域聚合得到
    - 新版：3个独立编码器，分别生成Z_orig、Z_space、Z_time（符合专利）
    """
    def __init__(self, n_features, latent_dim=64, hidden_units=128,
                 k_spatial=5, k_temporal=5, use_partial_distance=True, use_variable_mapping=True,
                 dropout_rate=0.2, l2_reg=0.001, name='ae_cs_legacy'):
        super(AECS_Legacy, self).__init__(name=name)

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.k_spatial = k_spatial
        self.k_temporal = k_temporal

        # 单编码器
        self.encoder = Encoder(latent_dim, hidden_units, dropout_rate=dropout_rate, l2_reg=l2_reg)
        self.decoder = Decoder(n_features, hidden_units, dropout_rate=dropout_rate, l2_reg=l2_reg)
        self.gating_network = GatingNetwork(latent_dim, dropout_rate=dropout_rate, l2_reg=l2_reg)

        from .neighborhood import NeighborhoodModule
        self.neighborhood_module = NeighborhoodModule(k_spatial, k_temporal, use_partial_distance, use_variable_mapping)

    def call(self, x, mask, training=False, return_all=False):
        """旧版前向传播（单编码器 + 邻域聚合）"""
        missing_rate = 1.0 - tf.reduce_mean(mask, axis=[1, 2])

        # 单编码器
        z_orig = self.encoder(x, mask, training=training)

        # 邻域聚合得到Z_space和Z_time（不符合专利）
        z_space, z_time, neighborhood_info = self.neighborhood_module.compute_neighborhood_embeddings(
            x, z_orig, mask
        )

        alpha, z_fused = self.gating_network(z_orig, z_space, z_time, missing_rate, training=training)
        x_hat = self.decoder(z_fused, training=training)

        if training:
            x_filled = x_hat
        else:
            x_filled = mask * x + (1.0 - mask) * x_hat

        if return_all:
            return {
                'x_hat': x_hat,
                'x_filled': x_filled,
                'z_orig': z_orig,
                'z_space': z_space,
                'z_time': z_time,
                'z_fused': z_fused,
                'alpha': alpha,
                'missing_rate': missing_rate,
                'neighborhood_info': neighborhood_info
            }
        else:
            return x_filled
