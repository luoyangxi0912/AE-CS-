#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from data import AECSDataLoader
from models.ae_cs import AECS


def load_model_and_data(checkpoint_dir, data_path):
    checkpoint_dir = Path(checkpoint_dir)

    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print('=' * 80)
    print('Load config')
    print('=' * 80)
    print(json.dumps(config, indent=2, ensure_ascii=False))

    print('\n' + '=' * 80)
    print('Load data')
    print('=' * 80)

    data_loader = AECSDataLoader(
        batch_size=config.get('batch_size', 8),
        shuffle_train=False,
        seed=config.get('seed', 42)
    )
    data_loader.preprocessor.data_path = Path(data_path)
    data_loader.preprocessor.window_size = config.get('window_size', 48)

    datasets = data_loader.prepare(
        missing_rate=config.get('missing_rate', 0.2),
        missing_type=config.get('missing_type', 'MCAR'),
        train_ratio=0.7,
        val_ratio=0.15
    )

    test_dataset = datasets['test']

    n_test = getattr(test_dataset, 'n_samples', None)
    if n_test is None:
        try:
            n_test = len(test_dataset)
        except TypeError:
            n_test = 'unknown'
    print(f'Test samples: {n_test}')

    print('\n' + '=' * 80)
    print('Init model')
    print('=' * 80)

    model = AECS(
        n_features=config.get('n_features', 44),
        latent_dim=config.get('latent_dim', 32),
        hidden_units=config.get('hidden_units', 128),
        k_spatial=config.get('k_spatial', 5),
        k_temporal=config.get('k_temporal', 5),
        dropout_rate=config.get('dropout_rate', 0.1),
        l2_reg=config.get('l2_reg', 0.0005)
    )

    best_model_path = checkpoint_dir / 'best_model.weights.h5'

    dummy_batch = next(iter(test_dataset.get_dataset()))
    _ = model(dummy_batch[0], dummy_batch[1], training=False)

    model.load_weights(str(best_model_path))
    print(f'Loaded model weights: {best_model_path}')

    return model, test_dataset, config


def predict_on_test_set(model, test_dataset):
    print('\n' + '=' * 80)
    print('Predict on test set')
    print('=' * 80)

    predictions = []
    ground_truth = []
    masks = []
    inputs = []

    for X, M in test_dataset.get_dataset():
        X_pred = model(X, M, training=False)
        predictions.append(X_pred.numpy())
        ground_truth.append(X.numpy())
        masks.append(M.numpy())
        inputs.append((X * M).numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    masks = np.concatenate(masks, axis=0)
    inputs = np.concatenate(inputs, axis=0)

    print(f'Prediction shape: {predictions.shape}')
    print(f'Ground truth shape: {ground_truth.shape}')

    return predictions, ground_truth, masks, inputs


def calculate_metrics(predictions, ground_truth, masks):
    print('\n' + '=' * 80)
    print('Calculate metrics')
    print('=' * 80)

    missing_mask = (1 - masks).astype(bool)
    observed_mask = masks.astype(bool)

    y_true_missing = ground_truth[missing_mask]
    y_pred_missing = predictions[missing_mask]

    y_true_observed = ground_truth[observed_mask]
    y_pred_observed = predictions[observed_mask]

    metrics = {
        'missing': {
            'mse': mean_squared_error(y_true_missing, y_pred_missing),
            'rmse': np.sqrt(mean_squared_error(y_true_missing, y_pred_missing)),
            'mae': mean_absolute_error(y_true_missing, y_pred_missing),
            'r2': r2_score(y_true_missing, y_pred_missing),
            'mape': np.mean(np.abs((y_true_missing - y_pred_missing) / (y_true_missing + 1e-8))) * 100
        },
        'observed': {
            'mse': mean_squared_error(y_true_observed, y_pred_observed),
            'rmse': np.sqrt(mean_squared_error(y_true_observed, y_pred_observed)),
            'mae': mean_absolute_error(y_true_observed, y_pred_observed),
            'r2': r2_score(y_true_observed, y_pred_observed),
            'mape': np.mean(np.abs((y_true_observed - y_pred_observed) / (y_true_observed + 1e-8))) * 100
        },
        'overall': {
            'mse': mean_squared_error(ground_truth.flatten(), predictions.flatten()),
            'rmse': np.sqrt(mean_squared_error(ground_truth.flatten(), predictions.flatten())),
            'mae': mean_absolute_error(ground_truth.flatten(), predictions.flatten()),
            'r2': r2_score(ground_truth.flatten(), predictions.flatten()),
            'mape': np.mean(np.abs((ground_truth - predictions) / (ground_truth + 1e-8))) * 100
        }
    }

    print('\nMissing positions (main focus):')
    print(f"  MSE:  {metrics['missing']['mse']:.6f}")
    print(f"  RMSE: {metrics['missing']['rmse']:.6f}")
    print(f"  MAE:  {metrics['missing']['mae']:.6f}")
    print(f"  R2:   {metrics['missing']['r2']:.6f}")
    print(f"  MAPE: {metrics['missing']['mape']:.2f}%")

    print('\nObserved positions (reference):')
    print(f"  MSE:  {metrics['observed']['mse']:.6f}")
    print(f"  RMSE: {metrics['observed']['rmse']:.6f}")
    print(f"  MAE:  {metrics['observed']['mae']:.6f}")
    print(f"  R2:   {metrics['observed']['r2']:.6f}")
    print(f"  MAPE: {metrics['observed']['mape']:.2f}%")

    return metrics


def analyze_per_feature(predictions, ground_truth, masks, save_dir):
    print('\n' + '=' * 80)
    print('Per-feature analysis')
    print('=' * 80)

    n_features = predictions.shape[2]
    missing_mask = (1 - masks).astype(bool)

    y_true_missing = ground_truth[missing_mask]
    y_pred_missing = predictions[missing_mask]

    plt.figure(figsize=(10, 10))
    plt.scatter(y_true_missing, y_pred_missing, alpha=0.1, s=1)
    plt.plot(
        [y_true_missing.min(), y_true_missing.max()],
        [y_true_missing.min(), y_true_missing.max()],
        'r--', lw=2, label='Perfect Prediction'
    )
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Prediction vs Truth (Missing Values Only)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'prediction_vs_truth_scatter.png', dpi=300)
    plt.close()
    print(f"  Saved: {save_dir / 'prediction_vs_truth_scatter.png'}")

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
    print(f"  Saved: {save_dir / 'error_distribution.png'}")

    rows = []
    for feat_idx in range(n_features):
        feat_missing = (masks[:, :, feat_idx] == 0)
        n_missing = int(np.sum(feat_missing))

        if n_missing == 0:
            rows.append({
                'feature_idx': feat_idx,
                'n_missing': 0,
                'mse': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan
            })
            continue

        y_true_f = ground_truth[:, :, feat_idx][feat_missing]
        y_pred_f = predictions[:, :, feat_idx][feat_missing]
        mse_f = mean_squared_error(y_true_f, y_pred_f)

        rows.append({
            'feature_idx': feat_idx,
            'n_missing': n_missing,
            'mse': mse_f,
            'rmse': float(np.sqrt(mse_f)),
            'mae': mean_absolute_error(y_true_f, y_pred_f),
            'r2': r2_score(y_true_f, y_pred_f),
            'mape': float(np.mean(np.abs((y_true_f - y_pred_f) / (y_true_f + 1e-8))) * 100)
        })

    feature_df = pd.DataFrame(rows).sort_values('rmse', ascending=False)
    feature_csv_path = save_dir / 'feature_performance.csv'
    feature_df.to_csv(feature_csv_path, index=False, encoding='utf-8-sig')
    print(f'  Saved: {feature_csv_path}')

    return feature_df


def visualize_predictions(predictions, ground_truth, masks, save_dir, n_samples=5):
    print('\n' + '=' * 80)
    print('Extra prediction visualization')
    print('=' * 80)

    n = min(n_samples, len(predictions))
    sample_indices = np.random.choice(len(predictions), size=n, replace=False)

    for rank, idx in enumerate(sample_indices, start=1):
        fig, ax = plt.subplots(figsize=(16, 5))

        x_true = ground_truth[idx].flatten()
        x_pred = predictions[idx].flatten()
        x_mask = masks[idx].flatten()
        t = np.arange(len(x_true))

        ax.plot(t, x_true, color='tab:blue', linewidth=1.4, label='Ground Truth')
        ax.plot(t, x_pred, color='tab:red', linewidth=1.1, alpha=0.85, label='Prediction')

        miss_pos = np.where(x_mask == 0)[0]
        if len(miss_pos) > 0:
            ax.scatter(miss_pos, x_true[miss_pos], s=8, c='tab:green', marker='x', label='Missing (Truth)')
            ax.scatter(miss_pos, x_pred[miss_pos], s=8, c='tab:orange', marker='o', label='Missing (Pred)')

        ax.set_title(f'Sample {idx} Flattened Series')
        ax.set_xlabel('Flattened Time-Feature Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        out_path = save_dir / f'prediction_sample_{rank}_idx_{idx}.png'
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate AE-CS model')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_v4', help='Checkpoint directory')
    parser.add_argument('--data_path', type=str, default=r'D:\数据补全\hangmei_90_拼接好的.csv', help='Data path')
    parser.add_argument('--output_dir', type=str, default='./results/eval_v4', help='Output directory')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of extra visualization samples')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, test_dataset, _ = load_model_and_data(args.checkpoint_dir, args.data_path)

    predictions, ground_truth, masks, _ = predict_on_test_set(model, test_dataset)

    metrics = calculate_metrics(predictions, ground_truth, masks)

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f'\nSaved metrics: {metrics_path}')

    _ = analyze_per_feature(predictions, ground_truth, masks, output_dir)

    visualize_predictions(
        predictions,
        ground_truth,
        masks,
        output_dir,
        n_samples=args.n_samples
    )

    print('\n' + '=' * 80)
    print('Evaluation finished')
    print('=' * 80)
    print(f'All outputs saved to: {output_dir}')
    print('Generated files:')
    print('  - metrics.json')
    print('  - feature_performance.csv')
    print('  - prediction_vs_truth_scatter.png')
    print('  - error_distribution.png')
    print('  - prediction_sample_*.png')


if __name__ == '__main__':
    main()
