#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速评估脚本 - 带GPU内存增长设置"""

import os
# 在导入TensorFlow之前设置
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
# 设置GPU内存增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import AECSDataLoader
from models.ae_cs import AECS

def main():
    checkpoint_dir = Path('checkpoints_v2')

    # Load config
    with open(checkpoint_dir / 'config.json', 'r') as f:
        config = json.load(f)

    print('Loading data...')
    data_loader = AECSDataLoader(batch_size=config['batch_size'], shuffle_train=False, seed=config['seed'])
    # 使用相对路径避免编码问题
    data_loader.preprocessor.data_path = Path('./hangmei_90_拼接好的.csv')
    data_loader.preprocessor.window_size = config['window_size']
    datasets = data_loader.prepare(missing_rate=config['missing_rate'], missing_type=config['missing_type'], train_ratio=0.7, val_ratio=0.15)
    test_dataset = datasets['test']

    print('Initializing model...')
    model = AECS(
        n_features=config['n_features'],
        latent_dim=config['latent_dim'],
        hidden_units=config['hidden_units'],
        k_spatial=config['k_spatial'],
        k_temporal=config['k_temporal'],
        dropout_rate=config.get('dropout_rate', 0.2),
        l2_reg=config.get('l2_reg', 0.001)
    )

    # Build model
    dummy_batch = next(iter(test_dataset.get_dataset()))
    _ = model(dummy_batch[0], dummy_batch[1], training=False)
    model.load_weights(str(checkpoint_dir / 'best_model.weights.h5'))
    print('Model loaded.')

    print('Predicting on test set...')
    predictions, ground_truth, masks = [], [], []
    for X, M in test_dataset.get_dataset():
        X_pred = model(X, M, training=False)
        predictions.append(X_pred.numpy())
        ground_truth.append(X.numpy())
        masks.append(M.numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    masks = np.concatenate(masks, axis=0)

    # Calculate metrics for missing positions
    missing_mask = (1 - masks).astype(bool)
    y_true_missing = ground_truth[missing_mask]
    y_pred_missing = predictions[missing_mask]

    # For observed positions
    observed_mask = masks.astype(bool)
    y_true_observed = ground_truth[observed_mask]
    y_pred_observed = predictions[observed_mask]

    print('')
    print('='*60)
    print('缺失位置的指标 (主要关注):')
    print('='*60)
    mse_m = mean_squared_error(y_true_missing, y_pred_missing)
    r2_m = r2_score(y_true_missing, y_pred_missing)
    mae_m = mean_absolute_error(y_true_missing, y_pred_missing)
    print(f'  MSE:  {mse_m:.6f}')
    print(f'  RMSE: {np.sqrt(mse_m):.6f}')
    print(f'  MAE:  {mae_m:.6f}')
    print(f'  R2:   {r2_m:.6f}')

    print('')
    print('='*60)
    print('观测位置的指标 (作为对比):')
    print('='*60)
    mse_o = mean_squared_error(y_true_observed, y_pred_observed)
    r2_o = r2_score(y_true_observed, y_pred_observed)
    mae_o = mean_absolute_error(y_true_observed, y_pred_observed)
    print(f'  MSE:  {mse_o:.6f}')
    print(f'  RMSE: {np.sqrt(mse_o):.6f}')
    print(f'  MAE:  {mae_o:.6f}')
    print(f'  R2:   {r2_o:.6f}')

if __name__ == '__main__':
    main()
