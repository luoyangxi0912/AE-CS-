#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估和预测分析脚本

功能:
1. 加载训练好的最佳模型
2. 在测试集上进行预测
3. 计算各种评估指标 (MSE, MAE, RMSE, R², MAPE)
4. 可视化预测结果
5. 分析误差分布
6. 识别预测困难的特征
"""

import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# 导入自定义模块
from data import AECSDataLoader
from models.ae_cs import AECS


def load_model_and_data(checkpoint_dir, data_path):
    """
    加载模型和数据

    Args:
        checkpoint_dir: 检查点目录
        data_path: 数据文件路径

    Returns:
        model: 加载的模型
        test_dataset: 测试数据集
        config: 配置信息
    """
    checkpoint_dir = Path(checkpoint_dir)

    # 1. 加载配置
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("=" * 80)
    print("加载配置")
    print("=" * 80)
    print(json.dumps(config, indent=2, ensure_ascii=False))

    # 2. 初始化数据加载器
    print("\n" + "=" * 80)
    print("加载数据")
    print("=" * 80)

    data_loader = AECSDataLoader(
        batch_size=config['batch_size'],
        shuffle_train=False,  # 评估时不打乱
        seed=config['seed']
    )
    data_loader.preprocessor.data_path = Path(data_path)
    data_loader.preprocessor.window_size = config['window_size']

    # 准备数据
    datasets = data_loader.prepare(
        missing_rate=config['missing_rate'],
        missing_type=config['missing_type'],
        train_ratio=0.7,
        val_ratio=0.15
    )

    test_dataset = datasets['test']

    print(f"测试集样本数: {len(test_dataset)}")

    # 3. 初始化模型
    print("\n" + "=" * 80)
    print("初始化模型")
    print("=" * 80)

    model = AECS(
        n_features=config['n_features'],
        latent_dim=config['latent_dim'],
        hidden_units=config['hidden_units'],
        k_spatial=config['k_spatial'],
        k_temporal=config['k_temporal'],
        dropout_rate=config.get('dropout_rate', 0.2),
        l2_reg=config.get('l2_reg', 0.001)
    )

    # 4. 加载权重
    best_model_path = checkpoint_dir / 'best_model.weights.h5'

    # 需要先进行一次前向传播来创建权重
    dummy_batch = next(iter(test_dataset.get_dataset()))
    _ = model(dummy_batch[0], dummy_batch[1], training=False)

    # 加载权重
    model.load_weights(str(best_model_path))
    print(f"已加载模型权重: {best_model_path}")

    return model, test_dataset, config


def predict_on_test_set(model, test_dataset):
    """
    在测试集上进行预测

    Args:
        model: 训练好的模型
        test_dataset: 测试数据集

    Returns:
        predictions: 预测值
        ground_truth: 真实值
        masks: 掩码
        inputs: 输入数据
    """
    print("\n" + "=" * 80)
    print("在测试集上进行预测")
    print("=" * 80)

    predictions = []
    ground_truth = []
    masks = []
    inputs = []

    test_data = test_dataset.get_dataset()

    for X, M in test_data:
        # 预测
        X_pred = model(X, M, training=False)

        predictions.append(X_pred.numpy())
        ground_truth.append(X.numpy())
        masks.append(M.numpy())
        inputs.append((X * M).numpy())  # 输入的观测值

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    masks = np.concatenate(masks, axis=0)
    inputs = np.concatenate(inputs, axis=0)

    print(f"预测形状: {predictions.shape}")
    print(f"真实值形状: {ground_truth.shape}")

    return predictions, ground_truth, masks, inputs


def calculate_metrics(predictions, ground_truth, masks):
    """
    计算评估指标

    Args:
        predictions: 预测值 [n_samples, time_steps, n_features]
        ground_truth: 真实值
        masks: 掩码 (1=观测, 0=缺失)

    Returns:
        metrics_dict: 各种指标的字典
    """
    print("\n" + "=" * 80)
    print("计算评估指标")
    print("=" * 80)

    # 只在缺失位置计算指标
    missing_mask = (1 - masks).astype(bool)

    y_true_missing = ground_truth[missing_mask]
    y_pred_missing = predictions[missing_mask]

    # 在观测位置也计算指标作为对比
    observed_mask = masks.astype(bool)
    y_true_observed = ground_truth[observed_mask]
    y_pred_observed = predictions[observed_mask]

    metrics = {}

    # 缺失位置的指标
    metrics['missing'] = {
        'mse': mean_squared_error(y_true_missing, y_pred_missing),
        'rmse': np.sqrt(mean_squared_error(y_true_missing, y_pred_missing)),
        'mae': mean_absolute_error(y_true_missing, y_pred_missing),
        'r2': r2_score(y_true_missing, y_pred_missing),
        'mape': np.mean(np.abs((y_true_missing - y_pred_missing) / (y_true_missing + 1e-8))) * 100
    }

    # 观测位置的指标（作为对比）
    metrics['observed'] = {
        'mse': mean_squared_error(y_true_observed, y_pred_observed),
        'rmse': np.sqrt(mean_squared_error(y_true_observed, y_pred_observed)),
        'mae': mean_absolute_error(y_true_observed, y_pred_observed),
        'r2': r2_score(y_true_observed, y_pred_observed),
        'mape': np.mean(np.abs((y_true_observed - y_pred_observed) / (y_true_observed + 1e-8))) * 100
    }

    # 全局指标
    metrics['overall'] = {
        'mse': mean_squared_error(ground_truth.flatten(), predictions.flatten()),
        'rmse': np.sqrt(mean_squared_error(ground_truth.flatten(), predictions.flatten())),
        'mae': mean_absolute_error(ground_truth.flatten(), predictions.flatten()),
        'r2': r2_score(ground_truth.flatten(), predictions.flatten()),
        'mape': np.mean(np.abs((ground_truth - predictions) / (ground_truth + 1e-8))) * 100
    }

    print("\n缺失位置的指标 (主要关注):")
    print(f"  MSE:  {metrics['missing']['mse']:.6f}")
    print(f"  RMSE: {metrics['missing']['rmse']:.6f}")
    print(f"  MAE:  {metrics['missing']['mae']:.6f}")
    print(f"  R2:   {metrics['missing']['r2']:.6f}")
    print(f"  MAPE: {metrics['missing']['mape']:.2f}%")

    print("\n观测位置的指标 (作为对比):")
    print(f"  MSE:  {metrics['observed']['mse']:.6f}")
    print(f"  RMSE: {metrics['observed']['rmse']:.6f}")
    print(f"  MAE:  {metrics['observed']['mae']:.6f}")
    print(f"  R2:   {metrics['observed']['r2']:.6f}")
    print(f"  MAPE: {metrics['observed']['mape']:.2f}%")

    return metrics


def analyze_per_feature(predictions, ground_truth, masks, save_dir):
    """
    按特征分析预测性能

    Args:
        predictions: 预测值 [n_samples, time_steps, n_features]
        ground_truth: 真实值
        masks: 掩码
        save_dir: 保存目录
    """
    print("\n" + "=" * 80)
    print("按特征分析预测性能")
    print("=" * 80)

    n_features = predictions.shape[2]
    missing_mask = (1 - masks).astype(bool)

    feature_metrics = []

    for i in range(n_features):
        # 获取该特征的缺失位置
        feature_missing = missing_mask[:, :, i]

        if feature_missing.sum() == 0:
            continue

        y_true = ground_truth[:, :, i][feature_missing]
        y_pred = predictions[:, :, i][feature_missing]

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        feature_metrics.append({
            'feature': i,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'n_missing': feature_missing.sum()
        })

    feature_df = pd.DataFrame(feature_metrics)
    feature_df = feature_df.sort_values('mae', ascending=False)

    print("\n按MAE排序的特征性能 (前10个最差):")
    print(feature_df.head(10).to_string(index=False))

    print("\n按MAE排序的特征性能 (前10个最好):")
    print(feature_df.tail(10).to_string(index=False))

    # 保存完整结果
    feature_df.to_csv(save_dir / 'feature_performance.csv', index=False)
    print(f"\n特征性能已保存到: {save_dir / 'feature_performance.csv'}")

    return feature_df


def visualize_predictions(predictions, ground_truth, masks, save_dir, n_samples=5):
    """
    可视化预测结果

    Args:
        predictions: 预测值
        ground_truth: 真实值
        masks: 掩码
        save_dir: 保存目录
        n_samples: 可视化样本数
    """
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 预测vs真实值散点图 (缺失位置)
    missing_mask = (1 - masks).astype(bool)
    y_true_missing = ground_truth[missing_mask]
    y_pred_missing = predictions[missing_mask]

    plt.figure(figsize=(10, 10))
    plt.scatter(y_true_missing, y_pred_missing, alpha=0.1, s=1)
    plt.plot([y_true_missing.min(), y_true_missing.max()],
             [y_true_missing.min(), y_true_missing.max()],
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Prediction vs Truth (Missing Values Only)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'prediction_vs_truth_scatter.png', dpi=300)
    plt.close()
    print(f"  保存: {save_dir / 'prediction_vs_truth_scatter.png'}")

    # 2. 误差分布直方图
    errors = y_pred_missing - y_true_missing

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution', fontsize=14)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(np.abs(errors), bins=100, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Absolute Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Absolute Error Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'error_distribution.png', dpi=300)
    plt.close()
    print(f"  保存: {save_dir / 'error_distribution.png'}")

    # 3. 时间序列可视化 (随机选择几个样本)
    indices = np.random.choice(len(predictions), size=min(n_samples, len(predictions)), replace=False)

    for idx in indices:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        # 随机选择4个特征
        feature_indices = np.random.choice(predictions.shape[2], size=4, replace=False)

        for ax, feat_idx in zip(axes, feature_indices):
            time_steps = np.arange(predictions.shape[1])

            # 真实值
            ax.plot(time_steps, ground_truth[idx, :, feat_idx],
                   'b-', label='Ground Truth', linewidth=2)

            # 预测值
            ax.plot(time_steps, predictions[idx, :, feat_idx],
                   'r--', label='Prediction', linewidth=2)

            # 标记缺失位置
            missing_positions = np.where(masks[idx, :, feat_idx] == 0)[0]
            if len(missing_positions) > 0:
                ax.scatter(missing_positions,
                          ground_truth[idx, missing_positions, feat_idx],
                          c='green', s=50, marker='x', label='Missing (Truth)', zorder=5)
                ax.scatter(missing_positions,
                          predictions[idx, missing_positions, feat_idx],
                          c='orange', s=50, marker='o', label='Missing (Pred)', zorder=5)

            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f'Feature {feat_idx}', fontsize=12)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f'timeseries_sample_{idx}.png', dpi=300)
        plt.close()
        print(f"  保存: {save_dir / f'timeseries_sample_{idx}.png'}")

    print("\n所有可视化图表已生成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估AE-CS模型')
    parser.add_argument('--checkpoint_dir', type=str,
                       default=r'./checkpoints/regularized',
                       help='检查点目录')
    parser.add_argument('--data_path', type=str,
                       default=r'D:\数据补全\hangmei_90_拼接好的.csv',
                       help='数据文件路径')
    parser.add_argument('--output_dir', type=str,
                       default=r'./results/evaluation',
                       help='结果保存目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型和数据
    model, test_dataset, config = load_model_and_data(
        args.checkpoint_dir,
        args.data_path
    )

    # 2. 预测
    predictions, ground_truth, masks, inputs = predict_on_test_set(
        model,
        test_dataset
    )

    # 3. 计算指标
    metrics = calculate_metrics(predictions, ground_truth, masks)

    # 保存指标
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"\n指标已保存到: {metrics_path}")

    # 4. 按特征分析
    feature_df = analyze_per_feature(
        predictions,
        ground_truth,
        masks,
        output_dir
    )

    # 5. 可视化
    visualize_predictions(
        predictions,
        ground_truth,
        masks,
        output_dir,
        n_samples=5
    )

    # 6. 生成总结报告
    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)
    print(f"\n所有结果已保存到: {output_dir}")
    print("\n生成的文件:")
    print(f"  - metrics.json: 评估指标")
    print(f"  - feature_performance.csv: 各特征性能")
    print(f"  - prediction_vs_truth_scatter.png: 预测vs真实值散点图")
    print(f"  - error_distribution.png: 误差分布")
    print(f"  - timeseries_sample_*.png: 时间序列样本可视化")


if __name__ == '__main__':
    main()
