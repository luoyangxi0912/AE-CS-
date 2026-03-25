"""
Neighborhood-Preserving Embedding Module

瀹炵幇璁烘枃涓殑 Algorithm 1 鐨勬楠?9-12:
- 绌洪棿閭诲煙鎼滅储 (Spatial Neighborhood)
- 鏃堕棿閭诲煙鎼滅储 (Temporal Neighborhood)
- 鍔犳潈鑱氬悎 (Weighted Aggregation)

浣跨敤FAISS鍔犻€焝-NN鎼滅储
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
    浣跨敤FAISS杩涜蹇€焝-NN鎼滅储

    FAISS (Facebook AI Similarity Search) 鏄竴涓敤浜庨珮鏁堢浉浼兼€ф悳绱㈢殑搴?
    鏀寔GPU鍔犻€燂紝姣旂函TensorFlow瀹炵幇蹇?00鍊嶄互涓?
    """
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.index = None

    def build_index(self, vectors: np.ndarray, nlist=100):
        """
        鏋勫缓FAISS绱㈠紩

        Args:
            vectors: [n_samples, dim] - 鍚戦噺鏁版嵁搴?
            nlist: IVF绱㈠紩鐨勮仛绫讳腑蹇冩暟锛堝奖鍝嶉€熷害/绮惧害tradeoff锛?
        """
        n_samples, dim = vectors.shape

        if not FAISS_AVAILABLE:
            return None

        # 浣跨敤IVF锛圛nverted File锛夌储寮曚互鍔犻€熸悳绱?
        # 瀵逛簬灏忔暟鎹泦浣跨敤FlatL2锛屽ぇ鏁版嵁闆嗕娇鐢↖VF
        if n_samples < 10000:
            # Flat绱㈠紩锛氱簿纭悳绱紝閫傚悎灏忚妯℃暟鎹?
            self.index = faiss.IndexFlatL2(dim)
        else:
            # IVF绱㈠紩锛氳繎浼兼悳绱紝閫傚悎澶ц妯℃暟鎹?
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)

            # 璁粌IVF绱㈠紩
            self.index.train(vectors)

        # 娣诲姞鍚戦噺鍒扮储寮?
        self.index.add(vectors)

        return self.index

    def search(self, queries: np.ndarray, k: int):
        """
        鎼滅储k涓渶杩戦偦

        Args:
            queries: [n_queries, dim] - 鏌ヨ鍚戦噺
            k: 杩戦偦鏁伴噺

        Returns:
            distances: [n_queries, k] - 璺濈
            indices: [n_queries, k] - 绱㈠紩
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        distances, indices = self.index.search(queries, k)

        return distances, indices


def find_knn_spatial_with_partial_distance(X, mask, k=5, sigma=None):
    """
    鍩轰簬閮ㄥ垎璺濈绛栫暐鐨勭┖闂撮偦鍩熸悳绱?(绗﹀悎涓撳埄姝ラS4)

    涓撳埄瑕佹眰锛?
    1. 璇嗗埆鍏卞悓瑙傛祴闆嗗悎 O_common = O_i 鈭?O_j
    2. 鍙湪鍏卞悓瑙傛祴鐨勫彉閲忎笂璁＄畻璺濈
    3. 褰掍竴鍖栬窛绂? d = sqrt(sum((x_i - x_j)虏) / |O_common|)
    4. 楂樻柉鏍告潈閲? w = exp(-d虏 / 蟽虏)

    Args:
        X: [batch, time, features] - 鍘熷瑙傛祴鏁版嵁锛堥潪娼滃湪琛ㄧず锛侊級
        mask: [batch, time, features] - 鎺╃爜鐭╅樀 (1=瑙傛祴, 0=缂哄け)
        k: int - 閭诲煙鏁伴噺
        sigma: float - 楂樻柉鏍稿甫瀹斤紝濡傛灉涓篘one鍒欒嚜鍔ㄨ绠?

    Returns:
        indices: [batch, time, k] - 閭诲眳绱㈠紩
        weights: [batch, time, k] - 閭诲眳鏉冮噸锛堝凡褰掍竴鍖栵級
    """
    batch_size, time_steps, n_features = X.shape

    indices_list = []
    weights_list = []

    # 瀵规瘡涓牱鏈崟鐙鐞?
    for b in range(batch_size):
        X_b = X[b].numpy()      # [time, features]
        M_b = mask[b].numpy()   # [time, features]

        # 姝ラ1: 璁＄畻鍏卞悓瑙傛祴鎺╃爜 [time, time, features]
        # common_mask[i,j,f] = M[i,f] * M[j,f]
        M_expanded_i = M_b[:, np.newaxis, :]  # [time, 1, features]
        M_expanded_j = M_b[np.newaxis, :, :]  # [1, time, features]
        common_mask = M_expanded_i * M_expanded_j  # [time, time, features]
        del M_expanded_i, M_expanded_j  # 绔嬪嵆閲婃斁

        # 姝ラ2: 璁＄畻鍏卞悓瑙傛祴鏁伴噺 [time, time]
        n_common = np.sum(common_mask, axis=2)  # [time, time]

        # 姝ラ3: 璁＄畻宸€肩殑骞虫柟 [time, time, features]
        X_expanded_i = X_b[:, np.newaxis, :]  # [time, 1, features]
        X_expanded_j = X_b[np.newaxis, :, :]  # [1, time, features]
        diff = (X_expanded_i - X_expanded_j) * common_mask  # [time, time, features]
        del X_expanded_i, X_expanded_j, common_mask  # 绔嬪嵆閲婃斁

        squared_diff = diff ** 2
        del diff  # 绔嬪嵆閲婃斁

        # 姝ラ4: 璁＄畻褰掍竴鍖栬窛绂?[time, time]
        sum_squared_diff = np.sum(squared_diff, axis=2)  # [time, time]
        del squared_diff  # 绔嬪嵆閲婃斁

        distances = np.sqrt(sum_squared_diff / (n_common + 1e-8))  # [time, time]
        del sum_squared_diff  # 绔嬪嵆閲婃斁

        # 灏嗗瑙掔嚎璁句负鏃犵┓澶э紙鎺掗櫎鑷繁锛?
        np.fill_diagonal(distances, float('inf'))

        # 灏嗘病鏈夊叡鍚岃娴嬬殑浣嶇疆璁句负鏃犵┓澶?
        distances[n_common == 0] = float('inf')
        del n_common  # 绔嬪嵆閲婃斁

        # 姝ラ5: 鎵惧埌k涓渶杩戦偦
        k_indices = np.argsort(distances, axis=1)[:, :k]  # [time, k]
        k_distances = np.take_along_axis(distances, k_indices, axis=1)  # [time, k]
        del distances  # 绔嬪嵆閲婃斁

        # 姝ラ6: 璁＄畻楂樻柉鏍告潈閲?
        if sigma is None:
            # 鑷姩纭畾甯﹀锛氫娇鐢ㄦ湁鏁堣窛绂荤殑涓綅鏁?
            valid_dists = k_distances[k_distances != float('inf')]
            if len(valid_dists) > 0:
                sigma_auto = np.median(valid_dists) + 1e-8
            else:
                sigma_auto = 1.0
        else:
            sigma_auto = sigma

        k_weights = np.exp(-k_distances ** 2 / (sigma_auto ** 2))
        del k_distances  # 绔嬪嵆閲婃斁

        # 褰掍竴鍖栨潈閲?
        k_weights = k_weights / (np.sum(k_weights, axis=1, keepdims=True) + 1e-8)

        indices_list.append(k_indices)
        weights_list.append(k_weights)

        # 姣忎釜鏍锋湰澶勭悊瀹屽悗寮哄埗鍨冨溇鍥炴敹
        gc.collect()

    indices = tf.constant(np.stack(indices_list), dtype=tf.int32)
    weights = tf.constant(np.stack(weights_list), dtype=tf.float32)

    return indices, weights


def compute_spatial_neighborhood_features(z, indices, weights):
    """
    鏍规嵁閭诲煙绱㈠紩鍜屾潈閲嶈绠楃┖闂寸壒寰?

    杩欎釜鍑芥暟鍦ㄦ綔鍦ㄧ┖闂磟涓婂簲鐢ㄤ粠鍘熷绌洪棿X璁＄畻寰楀埌鐨勯偦鍩熷叧绯?

    Args:
        z: [batch, time, latent] - 娼滃湪琛ㄧず
        indices: [batch, time, k] - 閭诲眳绱㈠紩锛堜粠閮ㄥ垎璺濈鎼滅储寰楀埌锛?
        weights: [batch, time, k] - 閭诲眳鏉冮噸锛堜粠閮ㄥ垎璺濈鎼滅储寰楀埌锛?

    Returns:
        z_space: [batch, time, latent] - 绌洪棿閭诲煙鐗瑰緛
    """
    batch_size, time_steps, latent_dim = z.shape
    k = indices.shape[2]

    # 鏀堕泦閭诲眳鐨勬綔鍦ㄨ〃绀?
    z_neighbors = tf.gather(z, indices, batch_dims=1)  # [batch, time, k, latent]

    # 鍔犳潈鑱氬悎
    weights_expanded = tf.expand_dims(weights, axis=-1)  # [batch, time, k, 1]
    z_space = tf.reduce_sum(z_neighbors * weights_expanded, axis=2)  # [batch, time, latent]

    return z_space


def find_knn_spatial_faiss(z, mask, k=5):
    """
    [宸插簾寮?- 涓嶇鍚堜笓鍒╄姹俔 鍦ㄧ┖闂寸淮搴︿笂杩涜 k-NN 鎼滅储

    璀﹀憡锛氭鍑芥暟鍦ㄦ綔鍦ㄧ┖闂磟涓婃悳绱紝娌℃湁瀹炵幇涓撳埄瑕佹眰鐨?閮ㄥ垎璺濈"绛栫暐
    璇蜂娇鐢?find_knn_spatial_with_partial_distance 浠ｆ浛

    淇濈暀姝ゅ嚱鏁颁粎涓哄悜鍚庡吋瀹?

    Args:
        z: [batch, time, latent] - 娼滃湪琛ㄧず
        mask: [batch, time, features] - 鎺╃爜鐭╅樀
        k: int - 杩戦偦鏁伴噺

    Returns:
        z_neighbors: [batch, time, k, latent] - k涓┖闂磋繎閭荤殑琛ㄧず
        indices: [batch, time, k] - 杩戦偦绱㈠紩
        weights: [batch, time, k] - 杩戦偦鏉冮噸
    """
    batch_size, time_steps, latent_dim = z.shape

    # 灏哹atch鍜宼ime缁村害灞曞钩锛屼互渚垮湪鎵€鏈夋牱鏈腑鎼滅储绌洪棿杩戦偦
    z_flat = tf.reshape(z, [-1, latent_dim])  # [batch*time, latent]
    mask_flat = tf.reshape(mask, [-1, mask.shape[-1]])  # [batch*time, features]

    # 杞崲涓簄umpy杩涜FAISS鎼滅储
    z_np = z_flat.numpy().astype(np.float32)

    if FAISS_AVAILABLE:
        # 浣跨敤FAISS鎼滅储
        searcher = FAISSKNNSearcher()
        searcher.build_index(z_np)

        # 鎼滅储k+1涓偦灞咃紙鍖呮嫭鑷繁锛?
        distances, indices = searcher.search(z_np, k+1)

        # 鎺掗櫎鑷繁锛堢涓€涓級
        distances = distances[:, 1:]  # [batch*time, k]
        indices = indices[:, 1:]  # [batch*time, k]
    else:
        # 鍥為€€鍒癟ensorFlow瀹炵幇
        distances, indices = knn_search_tensorflow(z_flat, k)

    # 鏀堕泦杩戦偦鐨勫悜閲?
    z_neighbors_flat = tf.gather(z_flat, indices)  # [batch*time, k, latent]

    # 璁＄畻鏉冮噸锛堜娇鐢ㄨ窛绂荤殑璐熸寚鏁帮級
    distances_tf = tf.constant(distances, dtype=tf.float32)
    weights_flat = tf.nn.softmax(-distances_tf, axis=-1)  # [batch*time, k]

    # 閲嶅鍥炲師濮嬬淮搴?
    z_neighbors = tf.reshape(z_neighbors_flat, [batch_size, time_steps, k, latent_dim])
    weights = tf.reshape(weights_flat, [batch_size, time_steps, k])
    indices_reshaped = tf.reshape(tf.constant(indices), [batch_size, time_steps, k])

    return z_neighbors, indices_reshaped, weights


def compute_temporal_neighborhood_with_mapping(X, mask, z, k=5, sigma=None):
    """
    Temporal neighborhood mapping with gradient-preserving z-flow for L_time.
    """
    batch_size, time_steps, n_features = X.shape
    _, _, latent_dim = z.shape

    z_time_list = []
    z_var_list = []
    z_var_neighbors_list = []
    weights_var_list = []

    for b in range(batch_size):
        X_b = X[b].numpy()
        M_b = mask[b].numpy()
        z_b = z[b]  # keep as Tensor for gradients

        var_distances = np.zeros((n_features, n_features), dtype=np.float32)
        for j in range(n_features):
            for m in range(n_features):
                if j == m:
                    var_distances[j, m] = float("inf")
                    continue
                common_times = M_b[:, j] * M_b[:, m]
                n_common = np.sum(common_times)
                if n_common > 0:
                    x_j = X_b[:, j] * common_times
                    x_m = X_b[:, m] * common_times
                    squared_dist = np.sum((x_j - x_m) ** 2)
                    var_distances[j, m] = np.sqrt(squared_dist / n_common)
                else:
                    var_distances[j, m] = float("inf")

        var_indices = np.argsort(var_distances, axis=1)[:, :k]
        var_neighbor_dists = np.take_along_axis(var_distances, var_indices, axis=1)

        if sigma is None:
            valid_dists = var_neighbor_dists[np.isfinite(var_neighbor_dists)]
            sigma_auto = np.median(valid_dists) + 1e-8 if len(valid_dists) > 0 else 1.0
        else:
            sigma_auto = sigma

        var_weights = np.exp(-(var_neighbor_dists ** 2) / (sigma_auto ** 2))
        var_weights = np.where(np.isfinite(var_neighbor_dists), var_weights, 0.0)
        var_weights = var_weights / (np.sum(var_weights, axis=1, keepdims=True) + 1e-8)

        var_indices_tf = tf.convert_to_tensor(var_indices, dtype=tf.int32)
        var_weights_tf = tf.convert_to_tensor(var_weights, dtype=z.dtype)

        z_var_rows = []
        for j in range(n_features):
            observed_times = np.where(M_b[:, j] > 0)[0]
            if len(observed_times) > 0:
                observed_times_tf = tf.convert_to_tensor(observed_times, dtype=tf.int32)
                z_var_j = tf.reduce_mean(tf.gather(z_b, observed_times_tf, axis=0), axis=0)
            else:
                z_var_j = tf.zeros([latent_dim], dtype=z.dtype)
            z_var_rows.append(z_var_j)
        z_var_b = tf.stack(z_var_rows, axis=0)

        z_var_neighbors_b = tf.gather(z_var_b, var_indices_tf, axis=0)
        z_var_agg = tf.reduce_sum(
            tf.expand_dims(var_weights_tf, axis=-1) * z_var_neighbors_b,
            axis=1
        )

        z_time_rows = []
        for t in range(time_steps):
            observed_vars = np.where(M_b[t, :] > 0)[0]
            if len(observed_vars) > 0:
                observed_vars_tf = tf.convert_to_tensor(observed_vars, dtype=tf.int32)
                z_time_t = tf.reduce_mean(tf.gather(z_var_agg, observed_vars_tf, axis=0), axis=0)
            else:
                z_time_t = tf.zeros([latent_dim], dtype=z.dtype)
            z_time_rows.append(z_time_t)
        z_time_b = tf.stack(z_time_rows, axis=0)

        z_time_list.append(z_time_b)
        z_var_list.append(z_var_b)
        z_var_neighbors_list.append(z_var_neighbors_b)
        weights_var_list.append(var_weights_tf)

    z_time = tf.stack(z_time_list, axis=0)
    z_var = tf.stack(z_var_list, axis=0)
    z_var_neighbors = tf.stack(z_var_neighbors_list, axis=0)
    weights_var = tf.stack(weights_var_list, axis=0)

    return z_time, z_var, z_var_neighbors, weights_var

def find_knn_temporal_faiss(z, mask, k=5):
    """
    [宸插簾寮?- 涓嶇鍚堜笓鍒╄姹俔 鍦ㄦ椂闂寸淮搴︿笂杩涜 k-NN 鎼滅储

    璀﹀憡锛氭鍑芥暟鍦ㄦ椂闂存涔嬮棿鍋歬-NN锛屾病鏈夊疄鐜颁笓鍒╄姹傜殑"鍙橀噺绾р啋鏃堕棿绾ф槧灏?
    璇蜂娇鐢?compute_temporal_neighborhood_with_mapping 浠ｆ浛

    淇濈暀姝ゅ嚱鏁颁粎涓哄悜鍚庡吋瀹?

    鍘熷璇存槑锛?
    浣跨敤FAISS鍔犻€熸悳绱㈣繃绋?
    瀵逛簬AE-CS锛屾椂闂撮偦鍩熸寚鐨勬槸鍚屼竴鍙橀噺鍦ㄤ笉鍚屾椂鍒讳箣闂寸殑鐩镐技鎬?

    Args:
        z: [batch, time, latent] - 娼滃湪琛ㄧず
        mask: [batch, time, features] - 鎺╃爜鐭╅樀
        k: int - 杩戦偦鏁伴噺

    Returns:
        z_neighbors: [batch, time, k, latent] - k涓椂闂磋繎閭荤殑琛ㄧず
        indices: [batch, time, k] - 杩戦偦绱㈠紩
        weights: [batch, time, k] - 杩戦偦鏉冮噸
    """
    batch_size, time_steps, latent_dim = z.shape

    z_neighbors_list = []
    indices_list = []
    weights_list = []

    # 瀵筨atch涓殑姣忎釜鏍锋湰鍗曠嫭澶勭悊
    for b in range(batch_size):
        z_sample = z[b]  # [time, latent]
        z_np = z_sample.numpy().astype(np.float32)

        if FAISS_AVAILABLE:
            # 浣跨敤FAISS鎼滅储
            searcher = FAISSKNNSearcher()
            searcher.build_index(z_np)

            # 鎼滅储k+1涓偦灞咃紙鍖呮嫭鑷繁锛?
            distances, indices = searcher.search(z_np, min(k+1, time_steps))

            # 鎺掗櫎鑷繁锛堢涓€涓級
            distances = distances[:, 1:]  # [time, k]
            indices = indices[:, 1:]  # [time, k]
        else:
            # 鍥為€€鍒癟ensorFlow瀹炵幇
            distances, indices = knn_search_tensorflow(z_sample, k)

        # 鏀堕泦杩戦偦鐨勫悜閲?
        z_neighbors_sample = tf.gather(z_sample, indices)  # [time, k, latent]

        # 璁＄畻鏉冮噸
        distances_tf = tf.constant(distances, dtype=tf.float32)
        weights_sample = tf.nn.softmax(-distances_tf, axis=-1)  # [time, k]

        z_neighbors_list.append(z_neighbors_sample)
        indices_list.append(indices)
        weights_list.append(weights_sample)

    # 鍫嗗彔鎴恇atch
    z_neighbors = tf.stack(z_neighbors_list, axis=0)  # [batch, time, k, latent]
    indices = tf.stack([tf.constant(idx) for idx in indices_list], axis=0)  # [batch, time, k]
    weights = tf.stack(weights_list, axis=0)  # [batch, time, k]

    return z_neighbors, indices, weights


def knn_search_tensorflow(vectors, k):
    """
    浣跨敤TensorFlow瀹炵幇k-NN鎼滅储锛堝洖閫€鏂规锛?

    Args:
        vectors: [n, dim] - 鍚戦噺
        k: 杩戦偦鏁伴噺

    Returns:
        distances: [n, k] - 璺濈
        indices: [n, k] - 绱㈠紩
    """
    # 璁＄畻鎵€鏈夊悜閲忓涔嬮棿鐨勬姘忚窛绂?
    # [n, 1, dim] - [1, n, dim] = [n, n, dim]
    vectors_expanded = tf.expand_dims(vectors, axis=1)
    vectors_tiled = tf.expand_dims(vectors, axis=0)

    # 璁＄畻娆ф皬璺濈
    diff = vectors_expanded - vectors_tiled  # [n, n, dim]
    distances_all = tf.reduce_sum(tf.square(diff), axis=-1)  # [n, n]

    # 鎵惧埌k+1涓渶杩戦偦锛堝寘鎷嚜宸憋級
    k_actual = min(k+1, vectors.shape[0])
    distances, indices = tf.nn.top_k(-distances_all, k=k_actual)  # top_k鎵炬渶澶э紝鎵€浠ュ彇璐?
    distances = -distances  # 杞洖姝ｅ€?

    # 鎺掗櫎鑷繁锛堢涓€涓級
    distances = distances[:, 1:].numpy()
    indices = indices[:, 1:].numpy()

    return distances, indices


def weighted_aggregation(z_neighbors, weights):
    """
    鍔犳潈鑱氬悎杩戦偦琛ㄧず

    Z_aggregated = 危 w_k * Z_k

    Args:
        z_neighbors: [batch, time, k, latent] - 杩戦偦琛ㄧず
        weights: [batch, time, k] - 鏉冮噸

    Returns:
        z_aggregated: [batch, time, latent] - 鑱氬悎鍚庣殑琛ㄧず
    """
    weights_expanded = tf.expand_dims(weights, axis=-1)  # [batch, time, k, 1]
    z_aggregated = tf.reduce_sum(z_neighbors * weights_expanded, axis=2)  # [batch, time, latent]

    return z_aggregated


class NeighborhoodModule:
    """
    閭诲煙妯″潡 - 灏佽鎵€鏈夐偦鍩熺浉鍏虫搷浣滐紙绗﹀悎涓撳埄姝ラS4锛?

    鍏抽敭鏀硅繘锛?
    1. 绌洪棿閭诲煙浣跨敤閮ㄥ垎璺濈绛栫暐锛堝湪鍘熷鏁版嵁绌洪棿X璁＄畻锛? 闂2宸蹭慨澶?
    2. 鏃堕棿閭诲煙浣跨敤鍙橀噺绾р啋鏃堕棿绾ф槧灏?- 闂3宸蹭慨澶?

    瀹炵幇璁烘枃Algorithm 1鐨勬楠?-12
    """
    def __init__(self, k_spatial=5, k_temporal=5, use_partial_distance=True, use_variable_mapping=True):
        """
        Args:
            k_spatial: 绌洪棿閭诲煙鏁伴噺
            k_temporal: 鏃堕棿閭诲煙鏁伴噺
            use_partial_distance: 鏄惁浣跨敤閮ㄥ垎璺濈绛栫暐锛堜笓鍒╄姹傦紝闂2锛?
            use_variable_mapping: 鏄惁浣跨敤鍙橀噺绾ф槧灏勶紙涓撳埄瑕佹眰锛岄棶棰?锛?
        """
        self.k_spatial = k_spatial
        self.k_temporal = k_temporal
        self.use_partial_distance = use_partial_distance
        self.use_variable_mapping = use_variable_mapping

        if use_partial_distance:
            print("NeighborhoodModule: partial distance strategy (patent-compliant)")
        else:
            print("NeighborhoodModule: using legacy spatial k-NN strategy (not patent-compliant)")

        if use_variable_mapping:
            print("NeighborhoodModule: variable-level temporal mapping (patent-compliant)")
        else:
            print("NeighborhoodModule: using legacy temporal k-NN strategy (not patent-compliant)")

    def compute_neighborhood_embeddings(self, X, z_orig, mask):
        """
        璁＄畻绌洪棿鍜屾椂闂撮偦鍩熷祵鍏?

        Args:
            X: [batch, time, features] - 鍘熷瑙傛祴鏁版嵁
            z_orig: [batch, time, latent] - 鍘熷娼滃湪琛ㄧず
            mask: [batch, time, features] - 鎺╃爜

        Returns:
            z_space: [batch, time, latent] - 绌洪棿閭诲煙琛ㄧず
            z_time: [batch, time, latent] - 鏃堕棿閭诲煙琛ㄧず
            info: dict - 鍖呭惈杩戦偦绱㈠紩鍜屾潈閲嶇殑瀛楀吀
        """
        # ========== 绌洪棿閭诲煙锛堜笓鍒╂楠4 - 绌洪棿娴佸舰妯″潡锛?==========
        if self.use_partial_distance:
            # 浣跨敤閮ㄥ垎璺濈绛栫暐锛堢鍚堜笓鍒╋級
            # 姝ラ1: 鍦ㄥ師濮嬫暟鎹┖闂碭涓婅绠楅偦鍩熺储寮曞拰鏉冮噸
            indices_space, weights_space = find_knn_spatial_with_partial_distance(
                X, mask, k=self.k_spatial
            )

            # 姝ラ2: 鍦ㄦ綔鍦ㄧ┖闂磟涓婂簲鐢ㄩ偦鍩熸潈閲?
            z_space = compute_spatial_neighborhood_features(
                z_orig, indices_space, weights_space
            )

            # 涓轰簡鍏煎鎬э紝涔熸瀯閫?z_neighbors_space
            z_neighbors_space = tf.gather(z_orig, indices_space, batch_dims=1)
        else:
            # 浣跨敤浼犵粺FAISS鏂规硶锛堜笉绗﹀悎涓撳埄锛屼粎鐢ㄤ簬瀵规瘮锛?
            if FAISS_AVAILABLE:
                z_neighbors_space, indices_space, weights_space = find_knn_spatial_faiss(
                    z_orig, mask, k=self.k_spatial
                )
                z_space = weighted_aggregation(z_neighbors_space, weights_space)
            else:
                # 绠€鍖栧疄鐜帮細浣跨敤鏃堕棿缁村害鐨勫钩鍧?
                z_neighbors_space = tf.expand_dims(z_orig, axis=2)
                z_neighbors_space = tf.tile(z_neighbors_space, [1, 1, self.k_spatial, 1])
                weights_space = tf.ones([z_orig.shape[0], z_orig.shape[1], self.k_spatial]) / self.k_spatial
                z_space = weighted_aggregation(z_neighbors_space, weights_space)

        # ========== 鏃堕棿閭诲煙锛堜笓鍒╂楠4 - 鏃堕棿娴佸舰妯″潡锛?==========
        if self.use_variable_mapping:
            # 浣跨敤鍙橀噺绾р啋鏃堕棿绾ф槧灏勶紙绗﹀悎涓撳埄 - 闂3淇锛?
            z_time, z_var, z_var_neighbors, weights_var = compute_temporal_neighborhood_with_mapping(
                X, mask, z_orig, k=self.k_temporal
            )

            # 璁剧疆铏氭嫙鐨勬椂闂存绾ч偦灞呬俊鎭紙鍏煎鎬э級
            z_neighbors_time = None
            indices_time = None
            weights_time = weights_var  # 浣嗕繚鐣欏彉閲忕骇鏉冮噸鐢ㄤ簬L_time
        else:
            # 浣跨敤浼犵粺FAISS鏂规硶锛堜笉绗﹀悎涓撳埄锛屼粎鐢ㄤ簬瀵规瘮锛?
            if FAISS_AVAILABLE:
                z_neighbors_time, indices_time, weights_time = find_knn_temporal_faiss(
                    z_orig, mask, k=self.k_temporal
                )
            else:
                # 绠€鍖栧疄鐜帮細浣跨敤婊戝姩绐楀彛
                z_neighbors_time = tf.expand_dims(z_orig, axis=2)
                z_neighbors_time = tf.tile(z_neighbors_time, [1, 1, self.k_temporal, 1])
                weights_time = tf.ones([z_orig.shape[0], z_orig.shape[1], self.k_temporal]) / self.k_temporal
                indices_time = None

            z_time = weighted_aggregation(z_neighbors_time, weights_time)

            # 涓嶄娇鐢ㄥ彉閲忕骇鏄犲皠鏃讹紝杩欎簺涓篘one
            z_var = None
            z_var_neighbors = None
            weights_var = None

        # 杩斿洖鑱氬悎鍚庣殑琛ㄧず鍜岃缁嗕俊鎭?
        info = {
            'z_neighbors_space': z_neighbors_space,
            'z_neighbors_time': z_neighbors_time,  # 鍙兘鏄疦one锛堝彉閲忕骇鏄犲皠鏃讹級
            'weights_space': weights_space,
            'weights_time': weights_time,
            'indices_space': indices_space if self.use_partial_distance else None,
            'indices_time': indices_time if FAISS_AVAILABLE else None,
            # 鍙橀噺绾т俊鎭紙鐢ㄤ簬璁＄畻L_time - 涓撳埄绗?31-237琛岋級
            'z_var': z_var if self.use_variable_mapping else None,
            'z_var_neighbors': z_var_neighbors if self.use_variable_mapping else None,
            'weights_var': weights_var if self.use_variable_mapping else None
        }

        return z_space, z_time, info
