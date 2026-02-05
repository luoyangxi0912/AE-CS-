"""
循环插补测试脚本

比较单次推理与循环插补的效果
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入模型和循环插补模块
from models import AECS, compute_spatial_knn_init, compute_temporal_knn_init
from models.iterative_imputation import IterativeImputer, adaptive_iterative_impute


def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    """加载并预处理数据 (与训练脚本保持一致)"""
    # 加载数据
    df = pd.read_csv(filepath, encoding='gbk')

    # 移除非数值列 (与data/preprocessor.py一致)
    cols_to_drop = ['序号', '日期']
    cols_to_keep = [col for col in df.columns if col not in cols_to_drop]
    df = df[cols_to_keep]

    # 转换为float类型
    X = df.values.astype(np.float32)
    n_features = X.shape[1]
    print(f"   数据特征数: {n_features}")

    # 划分训练集和测试集
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler, n_features


def create_missing_data(X, missing_rate=0.3, random_state=42):
    """创建人工缺失数据"""
    np.random.seed(random_state)
    mask = np.random.binomial(1, 1 - missing_rate, X.shape).astype(np.float32)
    X_observed = X * mask
    return X_observed, mask


def compute_rmse(X_pred, X_true, mask):
    """计算缺失位置的RMSE"""
    missing_mask = 1.0 - mask
    n_missing = np.sum(missing_mask)
    if n_missing == 0:
        return 0.0
    mse = np.sum((X_pred - X_true) ** 2 * missing_mask) / n_missing
    return np.sqrt(mse)


def main():
    print("=" * 60)
    print("循环插补测试 - 比较单次推理与迭代精化效果")
    print("=" * 60)

    # 加载数据
    data_path = "hangmei_90_拼接好的.csv"
    print(f"\n1. 加载数据: {data_path}")
    X_train, X_test, scaler, n_features = load_and_preprocess_data(data_path)
    print(f"   训练集形状: {X_train.shape}")
    print(f"   测试集形状: {X_test.shape}")
    latent_dim = 16
    hidden_units = 64

    # 创建模型
    print("\n2. 创建AECS模型...")
    model = AECS(
        n_features=n_features,
        latent_dim=latent_dim,
        hidden_units=hidden_units,
        k_spatial=5,
        k_temporal=5
    )

    # 加载最佳模型权重
    checkpoint_path = "checkpoints/best_model.weights.h5"
    print(f"   加载模型权重: {checkpoint_path}")

    # 构建模型（需要先做一次前向传播）
    dummy_X = tf.zeros((1, 1, n_features))
    dummy_mask = tf.ones((1, 1, n_features))
    _ = model(dummy_X, dummy_mask, training=False)

    # 加载权重
    model.load_weights(checkpoint_path)
    print("   模型权重加载成功!")

    # 测试不同缺失率
    missing_rates = [0.2, 0.3, 0.4, 0.5]

    print("\n" + "=" * 60)
    print("3. 测试不同缺失率下的插补效果")
    print("=" * 60)

    for missing_rate in missing_rates:
        print(f"\n>>> 缺失率: {missing_rate:.0%}")
        print("-" * 50)

        # 创建缺失数据
        X_observed, mask = create_missing_data(X_test, missing_rate)

        # 转换为张量 [batch=1, time=N, features]
        X_tf = tf.constant(X_observed.reshape(1, -1, n_features), dtype=tf.float32)
        mask_tf = tf.constant(mask.reshape(1, -1, n_features), dtype=tf.float32)
        X_true_tf = tf.constant(X_test.reshape(1, -1, n_features), dtype=tf.float32)

        # 方法1: 单次推理
        print("\n   [方法1] 单次推理:")
        X_pred_single = model(X_tf, mask_tf, training=False)
        X_imputed_single = mask_tf * X_tf + (1.0 - mask_tf) * X_pred_single
        rmse_single = compute_rmse(
            X_imputed_single.numpy().reshape(-1, n_features),
            X_test,
            mask
        )
        print(f"   RMSE (单次推理): {rmse_single:.6f}")

        # 方法2: 循环插补（不重计算KNN）
        print("\n   [方法2] 循环插补 (不重计算KNN):")
        imputer_simple = IterativeImputer(
            model,
            max_iters=5,
            recompute_knn=False,
            verbose=False
        )
        X_imputed_simple, history_simple = imputer_simple.impute(X_tf, mask_tf, X_true_tf)
        rmse_simple = compute_rmse(
            X_imputed_simple.numpy().reshape(-1, n_features),
            X_test,
            mask
        )
        print(f"   RMSE (简单迭代): {rmse_simple:.6f}")
        print(f"   迭代次数: {history_simple['n_iters']}, 收敛: {history_simple['converged']}")

        # 方法3: 循环插补（重计算KNN）
        print("\n   [方法3] 循环插补 (重计算KNN):")
        imputer_full = IterativeImputer(
            model,
            max_iters=5,
            recompute_knn=True,
            verbose=False
        )
        X_imputed_full, history_full = imputer_full.impute(X_tf, mask_tf, X_true_tf)
        rmse_full = compute_rmse(
            X_imputed_full.numpy().reshape(-1, n_features),
            X_test,
            mask
        )
        print(f"   RMSE (完整迭代): {rmse_full:.6f}")
        print(f"   迭代次数: {history_full['n_iters']}, 收敛: {history_full['converged']}")

        # 方法4: 自适应循环插补
        print("\n   [方法4] 自适应循环插补:")
        X_imputed_adaptive, history_adaptive = adaptive_iterative_impute(
            model, X_tf, mask_tf, X_true_tf,
            base_iters=3, high_missing_iters=8,
            missing_threshold=0.35
        )
        rmse_adaptive = compute_rmse(
            X_imputed_adaptive.numpy().reshape(-1, n_features),
            X_test,
            mask
        )
        print(f"   RMSE (自适应): {rmse_adaptive:.6f}")

        # 汇总
        print("\n   === 汇总 ===")
        print(f"   单次推理:     RMSE = {rmse_single:.6f}")
        print(f"   简单迭代:     RMSE = {rmse_simple:.6f} (改善: {(rmse_single-rmse_simple)/rmse_single*100:+.2f}%)")
        print(f"   完整迭代:     RMSE = {rmse_full:.6f} (改善: {(rmse_single-rmse_full)/rmse_single*100:+.2f}%)")
        print(f"   自适应迭代:   RMSE = {rmse_adaptive:.6f} (改善: {(rmse_single-rmse_adaptive)/rmse_single*100:+.2f}%)")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
