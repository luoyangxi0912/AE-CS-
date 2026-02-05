"""
Neighborhood-Preserving Embedding Module

实现论文中的 Algorithm 1 的步骤 9-12:
- 空间邻域搜索 (Spatial Neighborhood)
- 时间邻域搜索 (Temporal Neighborhood)
- 加权聚合 (Weighted Aggregation)

使用FAISS加速k-NN搜索
"""

import gc
import tensorflow as tf
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using TensorFlow-based k-NN search (slower).")


class FAISSKNNSearcher:
    """
    使用FAISS进行快速k-NN搜索

    FAISS (Facebook AI Similarity Search) 是一个用于高效相似性搜索的库
    支持GPU加速，比纯TensorFlow实现快100倍以上
    """
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.index = None

    def build_index(self, vectors: np.ndarray, nlist=100):
        """
        构建FAISS索引

        Args:
            vectors: [n_samples, dim] - 向量数据库
            nlist: IVF索引的聚类中心数（影响速度/精度tradeoff）
        """
        n_samples, dim = vectors.shape

        if not FAISS_AVAILABLE:
            return None

        # 使用IVF（Inverted File）索引以加速搜索
        # 对于小数据集使用FlatL2，大数据集使用IVF
        if n_samples < 10000:
            # Flat索引：精确搜索，适合小规模数据
            self.index = faiss.IndexFlatL2(dim)
        else:
            # IVF索引：近似搜索，适合大规模数据
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)

            # 训练IVF索引
            self.index.train(vectors)

        # 添加向量到索引
        self.index.add(vectors)

        return self.index

    def search(self, queries: np.ndarray, k: int):
        """
        搜索k个最近邻

        Args:
            queries: [n_queries, dim] - 查询向量
            k: 近邻数量

        Returns:
            distances: [n_queries, k] - 距离
            indices: [n_queries, k] - 索引
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        distances, indices = self.index.search(queries, k)

        return distances, indices


def find_knn_spatial_with_partial_distance(X, mask, k=5, sigma=None):
    """
    基于部分距离策略的空间邻域搜索 (符合专利步骤S4)

    专利要求：
    1. 识别共同观测集合 O_common = O_i ∩ O_j
    2. 只在共同观测的变量上计算距离
    3. 归一化距离: d = sqrt(sum((x_i - x_j)²) / |O_common|)
    4. 高斯核权重: w = exp(-d² / σ²)

    Args:
        X: [batch, time, features] - 原始观测数据（非潜在表示！）
        mask: [batch, time, features] - 掩码矩阵 (1=观测, 0=缺失)
        k: int - 邻域数量
        sigma: float - 高斯核带宽，如果为None则自动计算

    Returns:
        indices: [batch, time, k] - 邻居索引
        weights: [batch, time, k] - 邻居权重（已归一化）
    """
    batch_size, time_steps, n_features = X.shape

    indices_list = []
    weights_list = []

    # 对每个样本单独处理
    for b in range(batch_size):
        X_b = X[b].numpy()      # [time, features]
        M_b = mask[b].numpy()   # [time, features]

        # 步骤1: 计算共同观测掩码 [time, time, features]
        # common_mask[i,j,f] = M[i,f] * M[j,f]
        M_expanded_i = M_b[:, np.newaxis, :]  # [time, 1, features]
        M_expanded_j = M_b[np.newaxis, :, :]  # [1, time, features]
        common_mask = M_expanded_i * M_expanded_j  # [time, time, features]
        del M_expanded_i, M_expanded_j  # 立即释放

        # 步骤2: 计算共同观测数量 [time, time]
        n_common = np.sum(common_mask, axis=2)  # [time, time]

        # 步骤3: 计算差值的平方 [time, time, features]
        X_expanded_i = X_b[:, np.newaxis, :]  # [time, 1, features]
        X_expanded_j = X_b[np.newaxis, :, :]  # [1, time, features]
        diff = (X_expanded_i - X_expanded_j) * common_mask  # [time, time, features]
        del X_expanded_i, X_expanded_j, common_mask  # 立即释放

        squared_diff = diff ** 2
        del diff  # 立即释放

        # 步骤4: 计算归一化距离 [time, time]
        sum_squared_diff = np.sum(squared_diff, axis=2)  # [time, time]
        del squared_diff  # 立即释放

        distances = np.sqrt(sum_squared_diff / (n_common + 1e-8))  # [time, time]
        del sum_squared_diff  # 立即释放

        # 将对角线设为无穷大（排除自己）
        np.fill_diagonal(distances, float('inf'))

        # 将没有共同观测的位置设为无穷大
        distances[n_common == 0] = float('inf')
        del n_common  # 立即释放

        # 步骤5: 找到k个最近邻
        k_indices = np.argsort(distances, axis=1)[:, :k]  # [time, k]
        k_distances = np.take_along_axis(distances, k_indices, axis=1)  # [time, k]
        del distances  # 立即释放

        # 步骤6: 计算高斯核权重
        if sigma is None:
            # 自动确定带宽：使用有效距离的中位数
            valid_dists = k_distances[k_distances != float('inf')]
            if len(valid_dists) > 0:
                sigma_auto = np.median(valid_dists) + 1e-8
            else:
                sigma_auto = 1.0
        else:
            sigma_auto = sigma

        k_weights = np.exp(-k_distances ** 2 / (sigma_auto ** 2))
        del k_distances  # 立即释放

        # 归一化权重
        k_weights = k_weights / (np.sum(k_weights, axis=1, keepdims=True) + 1e-8)

        indices_list.append(k_indices)
        weights_list.append(k_weights)

        # 每个样本处理完后强制垃圾回收
        gc.collect()

    indices = tf.constant(np.stack(indices_list), dtype=tf.int32)
    weights = tf.constant(np.stack(weights_list), dtype=tf.float32)

    return indices, weights


def compute_spatial_neighborhood_features(z, indices, weights):
    """
    根据邻域索引和权重计算空间特征

    这个函数在潜在空间z上应用从原始空间X计算得到的邻域关系

    Args:
        z: [batch, time, latent] - 潜在表示
        indices: [batch, time, k] - 邻居索引（从部分距离搜索得到）
        weights: [batch, time, k] - 邻居权重（从部分距离搜索得到）

    Returns:
        z_space: [batch, time, latent] - 空间邻域特征
    """
    batch_size, time_steps, latent_dim = z.shape
    k = indices.shape[2]

    # 收集邻居的潜在表示
    z_neighbors = tf.gather(z, indices, batch_dims=1)  # [batch, time, k, latent]

    # 加权聚合
    weights_expanded = tf.expand_dims(weights, axis=-1)  # [batch, time, k, 1]
    z_space = tf.reduce_sum(z_neighbors * weights_expanded, axis=2)  # [batch, time, latent]

    return z_space


def find_knn_spatial_faiss(z, mask, k=5):
    """
    [已废弃 - 不符合专利要求] 在空间维度上进行 k-NN 搜索

    警告：此函数在潜在空间z上搜索，没有实现专利要求的"部分距离"策略
    请使用 find_knn_spatial_with_partial_distance 代替

    保留此函数仅为向后兼容

    Args:
        z: [batch, time, latent] - 潜在表示
        mask: [batch, time, features] - 掩码矩阵
        k: int - 近邻数量

    Returns:
        z_neighbors: [batch, time, k, latent] - k个空间近邻的表示
        indices: [batch, time, k] - 近邻索引
        weights: [batch, time, k] - 近邻权重
    """
    batch_size, time_steps, latent_dim = z.shape

    # 将batch和time维度展平，以便在所有样本中搜索空间近邻
    z_flat = tf.reshape(z, [-1, latent_dim])  # [batch*time, latent]
    mask_flat = tf.reshape(mask, [-1, mask.shape[-1]])  # [batch*time, features]

    # 转换为numpy进行FAISS搜索
    z_np = z_flat.numpy().astype(np.float32)

    if FAISS_AVAILABLE:
        # 使用FAISS搜索
        searcher = FAISSKNNSearcher()
        searcher.build_index(z_np)

        # 搜索k+1个邻居（包括自己）
        distances, indices = searcher.search(z_np, k+1)

        # 排除自己（第一个）
        distances = distances[:, 1:]  # [batch*time, k]
        indices = indices[:, 1:]  # [batch*time, k]
    else:
        # 回退到TensorFlow实现
        distances, indices = knn_search_tensorflow(z_flat, k)

    # 收集近邻的向量
    z_neighbors_flat = tf.gather(z_flat, indices)  # [batch*time, k, latent]

    # 计算权重（使用距离的负指数）
    distances_tf = tf.constant(distances, dtype=tf.float32)
    weights_flat = tf.nn.softmax(-distances_tf, axis=-1)  # [batch*time, k]

    # 重塑回原始维度
    z_neighbors = tf.reshape(z_neighbors_flat, [batch_size, time_steps, k, latent_dim])
    weights = tf.reshape(weights_flat, [batch_size, time_steps, k])
    indices_reshaped = tf.reshape(tf.constant(indices), [batch_size, time_steps, k])

    return z_neighbors, indices_reshaped, weights


def compute_temporal_neighborhood_with_mapping(X, mask, z, k=5, sigma=None):
    """
    实现完整的变量级→时间级映射（符合专利步骤S4 - 时间流形模块）

    专利要求：
    第一步：基于观测子集的变量间相似度计算
      - 定义每个变量的有效观测时间集合: T_j = {t | M_{t,j} = 1}
      - 计算变量间的归一化距离: d(j,m) = sqrt(Σ_{t∈T_common}(x_{t,j}-x_{t,m})²) / sqrt(|T_common|)

    第二步：确定时间邻域与亲和权重
      - 高斯核权重: w_{j,m} = exp(-d(j,m)² / σ²)
      - K近邻策略选择邻域变量

    第三步：变量级嵌入映射
      (1) 构造变量级潜在表示: z_var[j] = mean_{t ∈ T_j} z[t, :]
      (2) 变量级邻域聚合: z_var_agg[j] = Σ_{m ∈ N_j} w_{j,m} * z_var[m]
      (3) 映射回时间步特征矩阵: z_time[t] = mean_{j ∈ F_t} z_var_agg[j]

    Args:
        X: [batch, time, features] - 原始观测数据
        mask: [batch, time, features] - 掩码矩阵
        z: [batch, time, latent] - 潜在表示
        k: int - 邻域数量
        sigma: float - 高斯核带宽

    Returns:
        z_time: [batch, time, latent] - 时间流形特征
        z_var: [batch, features, latent] - 变量级潜在表示（用于计算L_time）
        z_var_neighbors: [batch, features, k, latent] - 变量级邻居表示
        weights_var: [batch, features, k] - 变量级邻居权重
    """
    batch_size, time_steps, n_features = X.shape
    _, _, latent_dim = z.shape

    z_time_list = []
    z_var_list = []  # 收集变量级表示
    z_var_neighbors_list = []  # 收集变量级邻居
    weights_var_list = []  # 收集变量级权重

    for b in range(batch_size):
        X_b = X[b].numpy()      # [time, features]
        M_b = mask[b].numpy()   # [time, features]
        z_b = z[b].numpy()      # [time, latent]

        # ========== 第一步：基于观测子集的变量间相似度计算 ==========

        # 计算变量间的距离矩阵 [features, features]
        var_distances = np.zeros((n_features, n_features))

        for j in range(n_features):
            for m in range(n_features):
                if j == m:
                    var_distances[j, m] = float('inf')
                    continue

                # 找到变量j和m的共同观测时间点
                common_times = M_b[:, j] * M_b[:, m]  # [time]
                n_common = np.sum(common_times)

                if n_common > 0:
                    # 只在共同观测时间点计算距离
                    x_j = X_b[:, j] * common_times
                    x_m = X_b[:, m] * common_times
                    diff = (x_j - x_m) ** 2
                    squared_dist = np.sum(diff)

                    # 归一化距离
                    var_distances[j, m] = np.sqrt(squared_dist / n_common)
                else:
                    var_distances[j, m] = float('inf')

        # ========== 第二步：确定时间邻域与亲和权重 ==========

        # 为每个变量找k个最近的变量邻居
        var_indices = np.argsort(var_distances, axis=1)[:, :k]  # [features, k]
        var_neighbor_dists = np.take_along_axis(var_distances, var_indices, axis=1)  # [features, k]

        # 计算权重
        if sigma is None:
            valid_dists = var_neighbor_dists[var_neighbor_dists != float('inf')]
            sigma_auto = np.median(valid_dists) + 1e-8 if len(valid_dists) > 0 else 1.0
        else:
            sigma_auto = sigma

        var_weights = np.exp(-var_neighbor_dists ** 2 / (sigma_auto ** 2))  # [features, k]
        var_weights = var_weights / (np.sum(var_weights, axis=1, keepdims=True) + 1e-8)  # 归一化

        # ========== 第三步：变量级嵌入映射 ==========

        # (1) 构造变量级潜在表示
        z_var = np.zeros((n_features, latent_dim))

        for j in range(n_features):
            # 找到变量j被观测到的时间步
            observed_times = np.where(M_b[:, j] > 0)[0]

            if len(observed_times) > 0:
                # 聚合这些时间步的潜在表示
                z_var[j] = np.mean(z_b[observed_times, :], axis=0)
            else:
                # 如果该变量从未被观测，用零向量
                z_var[j] = np.zeros(latent_dim)

        # (2) 变量级邻域聚合
        z_var_agg = np.zeros((n_features, latent_dim))

        # 收集变量级邻居表示（用于计算L_time）
        z_var_neighbors_b = np.zeros((n_features, k, latent_dim))

        for j in range(n_features):
            neighbor_vars = var_indices[j]  # [k]
            neighbor_weights = var_weights[j]  # [k]

            # 收集邻居表示
            for i, m in enumerate(neighbor_vars):
                z_var_neighbors_b[j, i, :] = z_var[m]

            # 加权聚合邻居变量的表示
            for i, m in enumerate(neighbor_vars):
                z_var_agg[j] += neighbor_weights[i] * z_var[m]

        # (3) 映射回时间步维度
        z_time_b = np.zeros((time_steps, latent_dim))

        for t in range(time_steps):
            # 找到时间步t被观测到的变量
            observed_vars = np.where(M_b[t, :] > 0)[0]

            if len(observed_vars) > 0:
                # 聚合这些变量的变量级表示
                z_time_b[t] = np.mean(z_var_agg[observed_vars, :], axis=0)
            else:
                # 如果该时间步没有观测，用零向量
                z_time_b[t] = np.zeros(latent_dim)

        z_time_list.append(z_time_b)
        z_var_list.append(z_var)
        z_var_neighbors_list.append(z_var_neighbors_b)
        weights_var_list.append(var_weights)

        # 释放中间变量并强制垃圾回收
        del X_b, M_b, z_b, var_distances, var_indices, var_neighbor_dists
        del z_var_agg, z_time_b
        gc.collect()

    z_time = tf.constant(np.stack(z_time_list), dtype=tf.float32)  # [batch, time, latent]
    z_var = tf.constant(np.stack(z_var_list), dtype=tf.float32)  # [batch, features, latent]
    z_var_neighbors = tf.constant(np.stack(z_var_neighbors_list), dtype=tf.float32)  # [batch, features, k, latent]
    weights_var = tf.constant(np.stack(weights_var_list), dtype=tf.float32)  # [batch, features, k]

    return z_time, z_var, z_var_neighbors, weights_var


def find_knn_temporal_faiss(z, mask, k=5):
    """
    [已废弃 - 不符合专利要求] 在时间维度上进行 k-NN 搜索

    警告：此函数在时间步之间做k-NN，没有实现专利要求的"变量级→时间级映射"
    请使用 compute_temporal_neighborhood_with_mapping 代替

    保留此函数仅为向后兼容

    原始说明：
    使用FAISS加速搜索过程
    对于AE-CS，时间邻域指的是同一变量在不同时刻之间的相似性

    Args:
        z: [batch, time, latent] - 潜在表示
        mask: [batch, time, features] - 掩码矩阵
        k: int - 近邻数量

    Returns:
        z_neighbors: [batch, time, k, latent] - k个时间近邻的表示
        indices: [batch, time, k] - 近邻索引
        weights: [batch, time, k] - 近邻权重
    """
    batch_size, time_steps, latent_dim = z.shape

    z_neighbors_list = []
    indices_list = []
    weights_list = []

    # 对batch中的每个样本单独处理
    for b in range(batch_size):
        z_sample = z[b]  # [time, latent]
        z_np = z_sample.numpy().astype(np.float32)

        if FAISS_AVAILABLE:
            # 使用FAISS搜索
            searcher = FAISSKNNSearcher()
            searcher.build_index(z_np)

            # 搜索k+1个邻居（包括自己）
            distances, indices = searcher.search(z_np, min(k+1, time_steps))

            # 排除自己（第一个）
            distances = distances[:, 1:]  # [time, k]
            indices = indices[:, 1:]  # [time, k]
        else:
            # 回退到TensorFlow实现
            distances, indices = knn_search_tensorflow(z_sample, k)

        # 收集近邻的向量
        z_neighbors_sample = tf.gather(z_sample, indices)  # [time, k, latent]

        # 计算权重
        distances_tf = tf.constant(distances, dtype=tf.float32)
        weights_sample = tf.nn.softmax(-distances_tf, axis=-1)  # [time, k]

        z_neighbors_list.append(z_neighbors_sample)
        indices_list.append(indices)
        weights_list.append(weights_sample)

    # 堆叠成batch
    z_neighbors = tf.stack(z_neighbors_list, axis=0)  # [batch, time, k, latent]
    indices = tf.stack([tf.constant(idx) for idx in indices_list], axis=0)  # [batch, time, k]
    weights = tf.stack(weights_list, axis=0)  # [batch, time, k]

    return z_neighbors, indices, weights


def knn_search_tensorflow(vectors, k):
    """
    使用TensorFlow实现k-NN搜索（回退方案）

    Args:
        vectors: [n, dim] - 向量
        k: 近邻数量

    Returns:
        distances: [n, k] - 距离
        indices: [n, k] - 索引
    """
    # 计算所有向量对之间的欧氏距离
    # [n, 1, dim] - [1, n, dim] = [n, n, dim]
    vectors_expanded = tf.expand_dims(vectors, axis=1)
    vectors_tiled = tf.expand_dims(vectors, axis=0)

    # 计算欧氏距离
    diff = vectors_expanded - vectors_tiled  # [n, n, dim]
    distances_all = tf.reduce_sum(tf.square(diff), axis=-1)  # [n, n]

    # 找到k+1个最近邻（包括自己）
    k_actual = min(k+1, vectors.shape[0])
    distances, indices = tf.nn.top_k(-distances_all, k=k_actual)  # top_k找最大，所以取负
    distances = -distances  # 转回正值

    # 排除自己（第一个）
    distances = distances[:, 1:].numpy()
    indices = indices[:, 1:].numpy()

    return distances, indices


def weighted_aggregation(z_neighbors, weights):
    """
    加权聚合近邻表示

    Z_aggregated = Σ w_k * Z_k

    Args:
        z_neighbors: [batch, time, k, latent] - 近邻表示
        weights: [batch, time, k] - 权重

    Returns:
        z_aggregated: [batch, time, latent] - 聚合后的表示
    """
    weights_expanded = tf.expand_dims(weights, axis=-1)  # [batch, time, k, 1]
    z_aggregated = tf.reduce_sum(z_neighbors * weights_expanded, axis=2)  # [batch, time, latent]

    return z_aggregated


class NeighborhoodModule:
    """
    邻域模块 - 封装所有邻域相关操作（符合专利步骤S4）

    关键改进：
    1. 空间邻域使用部分距离策略（在原始数据空间X计算）- 问题2已修复
    2. 时间邻域使用变量级→时间级映射 - 问题3已修复

    实现论文Algorithm 1的步骤9-12
    """
    def __init__(self, k_spatial=5, k_temporal=5, use_partial_distance=True, use_variable_mapping=True):
        """
        Args:
            k_spatial: 空间邻域数量
            k_temporal: 时间邻域数量
            use_partial_distance: 是否使用部分距离策略（专利要求，问题2）
            use_variable_mapping: 是否使用变量级映射（专利要求，问题3）
        """
        self.k_spatial = k_spatial
        self.k_temporal = k_temporal
        self.use_partial_distance = use_partial_distance
        self.use_variable_mapping = use_variable_mapping

        if use_partial_distance:
            print("NeighborhoodModule: 使用部分距离策略（符合专利 - 问题2已修复）")
        else:
            print("NeighborhoodModule: 使用传统空间k-NN策略（不符合专利）")

        if use_variable_mapping:
            print("NeighborhoodModule: 使用变量级→时间级映射（符合专利 - 问题3已修复）")
        else:
            print("NeighborhoodModule: 使用传统时间k-NN策略（不符合专利）")

    def compute_neighborhood_embeddings(self, X, z_orig, mask):
        """
        计算空间和时间邻域嵌入

        Args:
            X: [batch, time, features] - 原始观测数据
            z_orig: [batch, time, latent] - 原始潜在表示
            mask: [batch, time, features] - 掩码

        Returns:
            z_space: [batch, time, latent] - 空间邻域表示
            z_time: [batch, time, latent] - 时间邻域表示
            info: dict - 包含近邻索引和权重的字典
        """
        # ========== 空间邻域（专利步骤S4 - 空间流形模块） ==========
        if self.use_partial_distance:
            # 使用部分距离策略（符合专利）
            # 步骤1: 在原始数据空间X上计算邻域索引和权重
            indices_space, weights_space = find_knn_spatial_with_partial_distance(
                X, mask, k=self.k_spatial
            )

            # 步骤2: 在潜在空间z上应用邻域权重
            z_space = compute_spatial_neighborhood_features(
                z_orig, indices_space, weights_space
            )

            # 为了兼容性，也构造 z_neighbors_space
            z_neighbors_space = tf.gather(z_orig, indices_space, batch_dims=1)
        else:
            # 使用传统FAISS方法（不符合专利，仅用于对比）
            if FAISS_AVAILABLE:
                z_neighbors_space, indices_space, weights_space = find_knn_spatial_faiss(
                    z_orig, mask, k=self.k_spatial
                )
                z_space = weighted_aggregation(z_neighbors_space, weights_space)
            else:
                # 简化实现：使用时间维度的平均
                z_neighbors_space = tf.expand_dims(z_orig, axis=2)
                z_neighbors_space = tf.tile(z_neighbors_space, [1, 1, self.k_spatial, 1])
                weights_space = tf.ones([z_orig.shape[0], z_orig.shape[1], self.k_spatial]) / self.k_spatial
                z_space = weighted_aggregation(z_neighbors_space, weights_space)

        # ========== 时间邻域（专利步骤S4 - 时间流形模块） ==========
        if self.use_variable_mapping:
            # 使用变量级→时间级映射（符合专利 - 问题3修复）
            z_time, z_var, z_var_neighbors, weights_var = compute_temporal_neighborhood_with_mapping(
                X, mask, z_orig, k=self.k_temporal
            )

            # 设置虚拟的时间步级邻居信息（兼容性）
            z_neighbors_time = None
            indices_time = None
            weights_time = weights_var  # 但保留变量级权重用于L_time
        else:
            # 使用传统FAISS方法（不符合专利，仅用于对比）
            if FAISS_AVAILABLE:
                z_neighbors_time, indices_time, weights_time = find_knn_temporal_faiss(
                    z_orig, mask, k=self.k_temporal
                )
            else:
                # 简化实现：使用滑动窗口
                z_neighbors_time = tf.expand_dims(z_orig, axis=2)
                z_neighbors_time = tf.tile(z_neighbors_time, [1, 1, self.k_temporal, 1])
                weights_time = tf.ones([z_orig.shape[0], z_orig.shape[1], self.k_temporal]) / self.k_temporal
                indices_time = None

            z_time = weighted_aggregation(z_neighbors_time, weights_time)

            # 不使用变量级映射时，这些为None
            z_var = None
            z_var_neighbors = None
            weights_var = None

        # 返回聚合后的表示和详细信息
        info = {
            'z_neighbors_space': z_neighbors_space,
            'z_neighbors_time': z_neighbors_time,  # 可能是None（变量级映射时）
            'weights_space': weights_space,
            'weights_time': weights_time,
            'indices_space': indices_space if self.use_partial_distance else None,
            'indices_time': indices_time if FAISS_AVAILABLE else None,
            # 变量级信息（用于计算L_time - 专利第231-237行）
            'z_var': z_var if self.use_variable_mapping else None,
            'z_var_neighbors': z_var_neighbors if self.use_variable_mapping else None,
            'weights_var': weights_var if self.use_variable_mapping else None
        }

        return z_space, z_time, info
