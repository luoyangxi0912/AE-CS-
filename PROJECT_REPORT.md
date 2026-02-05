# AE-CS 时间序列数据补全项目报告

**生成日期:** 2026-01-18
**项目名称:** AE-CS: AutoEncoder based on Coherent denoising and Spatio-temporal neighborhood-preserving embedding

---

## 目录

1. [实验概览](#1-实验概览)
2. [代码改动摘要](#2-代码改动摘要)
3. [数据集简述](#3-数据集简述)
4. [实验A：基准模型（ReLU激活）](#4-实验a基准模型relu激活)
5. [实验B：高斯激活+增强正则化](#5-实验b高斯激活增强正则化)
6. [评估一致性与消融实验](#6-评估一致性与消融实验)
7. [代码版本标识](#7-代码版本标识)
8. [评估结果对比](#8-评估结果对比)
9. [运行环境](#9-运行环境)
10. [核心代码附录](#10-核心代码附录)
11. [配置文件附录](#11-配置文件附录)

---

## 1. 实验概览

本报告记录了两次独立实验，使用不同的配置和代码版本：

| 项目 | 实验A（基准） | 实验B（高斯激活） |
|------|--------------|------------------|
| **检查点目录** | `checkpoints/` | `checkpoints/12h_run/` |
| **评估结果目录** | `results/latest_eval/` | `results/gaussian_eval/` |
| **激活函数** | ReLU (Dense层默认) | gaussian_activation |
| **hidden_units** | 128 | 256 |
| **latent_dim** | 32 | 64 |
| **batch_size** | 16 | 8 |
| **dropout_rate** | 0.1 | 0.3 |
| **l2_reg** | 0.0005 | 0.005 |
| **训练轮数** | 30 epochs | 20 epochs (早停) |
| **最佳epoch** | 19 | 5 |
| **最佳验证损失** | 7.2654 | 7.6081 |
| **测试集R²** | **0.297** | -0.782 |

---

## 2. 代码改动摘要

### 2.1 实验A使用的代码版本 (ae_cs.py)

```python
# Encoder.dense_latent - 无激活函数
self.dense_latent = layers.Dense(
    latent_dim,
    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    name='latent'
)

# GatingNetwork.dense1/dense2 - 使用ReLU
self.dense1 = layers.Dense(latent_dim * 2, activation='relu', ...)
self.dense2 = layers.Dense(latent_dim, activation='relu', ...)
```

### 2.2 实验B使用的代码版本 (ae_cs.py)

**新增自定义激活函数:**
```python
@tf.keras.utils.register_keras_serializable(package='Custom', name='gaussian_activation')
def gaussian_activation(x):
    """高斯激活函数: f(x) = 1 - exp(-x²)"""
    return 1.0 - tf.exp(-tf.square(x))
```

**修改位置:**
```python
# Encoder.dense_latent - 使用gaussian_activation
self.dense_latent = layers.Dense(
    latent_dim,
    activation=gaussian_activation,  # ← 新增
    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    name='latent'
)

# GatingNetwork.dense1/dense2 - ReLU → gaussian_activation
self.dense1 = layers.Dense(latent_dim * 2, activation=gaussian_activation, ...)
self.dense2 = layers.Dense(latent_dim, activation=gaussian_activation, ...)
```

### 2.3 两次实验共同的代码改动

| 文件 | 改动项 | 说明 |
|------|--------|------|
| ae_cs.py | GRU → LSTM | 解决cuDNN稳定性问题 |
| ae_cs.py | 数据泄露修复 | `x_observed = x * mask` 缺失位置置零 |
| neighborhood.py | 部分距离策略 | 实现专利要求的共同观测集合距离计算 |
| neighborhood.py | 变量级→时间级映射 | 实现专利步骤S4的时间流形模块 |
| losses.py | 权重归一化修复 | 修复类型不匹配问题 |
| train.py | 默认参数调整 | dropout 0.1→0.3, l2_reg 0.0005→0.005 |

---

## 3. 数据集简述

### 3.1 基本信息

| 项目 | 值 |
|------|-----|
| 数据文件 | `hangmei_90_拼接好的.csv` |
| 总行数 | 2793 |
| 特征数 | 44 (移除序号、日期后) |
| 数据类型 | 工业时间序列（传感器数据） |

### 3.2 缺失机制与比例

| 配置项 | 值 |
|--------|-----|
| missing_rate | 0.2 (20%) |
| missing_type | MCAR (Missing Completely At Random) |
| 实际训练集缺失率 | 19.90% |
| 实际验证集缺失率 | 19.40% |
| 实际测试集缺失率 | 19.96% |

### 3.3 数据切分方式

| 配置项 | 值 |
|--------|-----|
| 切分方式 | **时间顺序** (非随机) |
| train_ratio | 0.70 (70%) |
| val_ratio | 0.15 (15%) |
| test_ratio | 0.15 (15%) |
| 训练集 | 1955条 → 1908窗口 |
| 验证集 | 419条 → 372窗口 |
| 测试集 | 419条 → 372窗口 |

### 3.4 预处理统计 (prepare_data() 输出)

```
Loading data from: D:\数据补全\hangmei_90_拼接好的.csv
Loaded data shape: (2793, 44)
Features: 44
Data split before normalization: train=1955, val=419, test=419
Normalization check:
  Train: mean=-0.000000, std=1.000000  ← 只在训练集fit
  Val:   mean=-0.821307, std=4.318517  ← 分布偏移明显
  Test:  mean=0.524690, std=1.039409   ← 分布偏移明显
Created MCAR mask: target=20.00%, actual=19.90%
Created MCAR mask: target=20.00%, actual=19.40%
Created MCAR mask: target=20.00%, actual=19.96%
Created windows: train=1908, val=372, test=372
```

### 3.5 工况漂移分析

**存在严重的时序协变量偏移 (Covariate Shift):**

> **计算口径说明:** 漂移统计基于训练集 vs 测试集的均值/方差对比，按特征计算 z-score：
> ```
> drift = |mean_test - mean_train| / std_train
> ```

| 偏移程度 | 特征数量 | 占比 |
|----------|----------|------|
| 严重 (>1σ) | 10个 | 22.7% |
| 中等 (0.5-1σ) | 18个 | 40.9% |
| 轻微 (<0.5σ) | 16个 | 36.4% |

**偏移最严重的特征:**

| 特征索引 | 特征名 | 训练集均值 | 测试集均值 | 偏移量 |
|----------|--------|-----------|-----------|--------|
| 4 | TI10403 | 346.26 | 370.97 | **3.46σ** |
| 5 | TI10406A | 352.30 | 381.16 | **2.17σ** |
| 38 | TE20303 | 196.25 | 216.04 | 1.75σ |
| 11 | TI10414A | 394.02 | 412.00 | 1.70σ |

---

## 4. 实验A：基准模型（ReLU激活）

### 4.1 配置文件 (checkpoints/config.json)

```json
{
    "data_path": "D:\\数据补全\\hangmei_90_拼接好的.csv",
    "n_features": 44,
    "window_size": 48,
    "batch_size": 16,
    "latent_dim": 32,
    "hidden_units": 128,
    "k_spatial": 5,
    "k_temporal": 5,
    "p_drop": 0.1,
    "n_corrupted": 3,
    "lambda1": 1.0,
    "lambda2": 0.1,
    "lambda3": 0.1,
    "learning_rate": 0.001,
    "missing_rate": 0.2,
    "missing_type": "MCAR",
    "use_faiss": true,
    "dropout_rate": 0.1,
    "l2_reg": 0.0005,
    "seed": 42
}
```

### 4.2 运行命令

```bash
python train.py --epochs 30 --batch_size 16 --hidden_units 128 --latent_dim 32 --dropout_rate 0.1 --l2_reg 0.0005 --checkpoint_dir ./checkpoints
```

### 4.3 完整训练日志 (30 epochs)

| Epoch | Train Total | Train Recon | Train Consist | Train Space | Train Time | Val Total | Val Recon | Val Consist | Val Space | Val Time | LR |
|-------|-------------|-------------|---------------|-------------|------------|-----------|-----------|-------------|-----------|----------|-----|
| 1 | 1.2173 | 0.7923 | 0.2448 | 1.7847 | 0.0176 | 17.5472 | 17.5130 | 0.0118 | 0.2091 | 0.0139 | 0.001 |
| 2 | 0.5317 | 0.3814 | 0.0887 | 0.6076 | 0.0078 | 12.8004 | 12.7830 | 0.0053 | 0.1007 | 0.0201 | 0.001 |
| 3 | 0.3633 | 0.2902 | 0.0429 | 0.2972 | 0.0051 | 10.4190 | 10.4079 | 0.0035 | 0.0630 | 0.0131 | 0.001 |
| 4 | 0.2806 | 0.2415 | 0.0227 | 0.1612 | 0.0034 | 9.5533 | 9.5436 | 0.0030 | 0.0581 | 0.0089 | 0.001 |
| 5 | 0.2315 | 0.2082 | 0.0132 | 0.0988 | 0.0024 | 8.7520 | 8.7443 | 0.0024 | 0.0459 | 0.0078 | 0.001 |
| 6 | 0.1993 | 0.1841 | 0.0084 | 0.0665 | 0.0019 | 9.1551 | 9.1478 | 0.0022 | 0.0446 | 0.0060 | 0.001 |
| 7 | 0.1724 | 0.1615 | 0.0059 | 0.0492 | 0.0016 | 8.1136 | 8.1076 | 0.0019 | 0.0365 | 0.0051 | 0.001 |
| 8 | 0.1667 | 0.1574 | 0.0048 | 0.0435 | 0.0015 | 7.8183 | 7.8117 | 0.0020 | 0.0407 | 0.0050 | 0.001 |
| 9 | 0.1479 | 0.1404 | 0.0038 | 0.0363 | 0.0012 | 8.0538 | 8.0476 | 0.0018 | 0.0396 | 0.0041 | 0.001 |
| 10 | 0.1429 | 0.1361 | 0.0033 | 0.0335 | 0.0013 | 7.9189 | 7.9123 | 0.0020 | 0.0423 | 0.0042 | 0.001 |
| 11 | 0.1392 | 0.1327 | 0.0031 | 0.0325 | 0.0012 | 7.4868 | 7.4800 | 0.0022 | 0.0419 | 0.0045 | 0.001 |
| 12 | 0.1285 | 0.1215 | 0.0034 | 0.0354 | 0.0014 | 7.9506 | 7.9408 | 0.0032 | 0.0604 | 0.0048 | 0.001 |
| 13 | 0.1277 | 0.1212 | 0.0030 | 0.0333 | 0.0012 | 7.7033 | 7.6948 | 0.0023 | 0.0580 | 0.0041 | 0.001 |
| 14 | 0.1213 | 0.1156 | 0.0026 | 0.0300 | 0.0010 | 7.6085 | 7.6011 | 0.0022 | 0.0486 | 0.0032 | 0.001 |
| 15 | 0.1114 | 0.1062 | 0.0024 | 0.0281 | 0.0009 | 7.5255 | 7.5165 | 0.0024 | 0.0623 | 0.0039 | 0.001 |
| 16 | 0.1122 | 0.1072 | 0.0022 | 0.0266 | 0.0009 | 7.7453 | 7.7371 | 0.0022 | 0.0572 | 0.0034 | 0.0005 |
| 17 | 0.1010 | 0.0968 | 0.0019 | 0.0232 | 0.0007 | 7.5917 | 7.5849 | 0.0019 | 0.0469 | 0.0027 | 0.0005 |
| 18 | 0.1045 | 0.1006 | 0.0017 | 0.0213 | 0.0007 | 7.4820 | 7.4752 | 0.0018 | 0.0471 | 0.0029 | 0.0005 |
| **19** | **0.1039** | **0.1001** | **0.0016** | **0.0208** | **0.0006** | **7.2654** | **7.2591** | **0.0017** | **0.0427** | **0.0033** | **0.0005** |
| 20 | 0.1006 | 0.0968 | 0.0016 | 0.0208 | 0.0007 | 7.6175 | 7.6109 | 0.0019 | 0.0450 | 0.0028 | 0.0005 |
| 21 | 0.0994 | 0.0958 | 0.0015 | 0.0200 | 0.0006 | 7.3856 | 7.3794 | 0.0017 | 0.0432 | 0.0026 | 0.0005 |
| 22 | 0.0972 | 0.0938 | 0.0014 | 0.0189 | 0.0006 | 7.3245 | 7.3179 | 0.0017 | 0.0464 | 0.0027 | 0.0005 |
| 23 | 0.0920 | 0.0887 | 0.0014 | 0.0184 | 0.0006 | 7.7738 | 7.7680 | 0.0016 | 0.0401 | 0.0025 | 0.0005 |
| 24 | 0.0946 | 0.0912 | 0.0014 | 0.0191 | 0.0006 | 7.3251 | 7.3191 | 0.0017 | 0.0408 | 0.0026 | 0.00025 |
| 25 | 0.0918 | 0.0887 | 0.0013 | 0.0178 | 0.0005 | 7.4808 | 7.4751 | 0.0016 | 0.0393 | 0.0022 | 0.00025 |
| 26 | 0.0927 | 0.0898 | 0.0012 | 0.0161 | 0.0005 | 7.3821 | 7.3770 | 0.0014 | 0.0350 | 0.0021 | 0.00025 |
| 27 | 0.0894 | 0.0864 | 0.0012 | 0.0164 | 0.0005 | 7.3629 | 7.3576 | 0.0015 | 0.0365 | 0.0022 | 0.00025 |
| 28 | 0.0874 | 0.0845 | 0.0012 | 0.0163 | 0.0005 | 7.4162 | 7.4107 | 0.0015 | 0.0377 | 0.0023 | 0.00025 |
| 29 | 0.0895 | 0.0868 | 0.0011 | 0.0154 | 0.0005 | 7.6576 | 7.6523 | 0.0014 | 0.0365 | 0.0020 | 0.000125 |
| 30 | 0.0874 | 0.0848 | 0.0011 | 0.0146 | 0.0004 | 7.6092 | 7.6043 | 0.0013 | 0.0341 | 0.0019 | 0.000125 |

**训练结果:**
- 最佳验证损失: **7.2654** (Epoch 19)
- 训练完成，无早停触发

### 4.4 学习率变化

```
Epoch 1-15:  0.001000
Epoch 16-23: 0.000500
Epoch 24-28: 0.000250
Epoch 29-30: 0.000125
```

### 4.5 评估结果 (results/latest_eval/metrics.json)

| 位置 | MSE | RMSE | MAE | R² | MAPE |
|------|-----|------|-----|-----|------|
| **缺失位置** | 0.7775 | 0.8818 | 0.6303 | **0.2969** | 210.49% |
| 观测位置 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0% |
| 整体 | 0.1559 | 0.3949 | 0.1264 | 0.8582 | 42.21% |

---

## 5. 实验B：高斯激活+增强正则化

### 5.1 配置文件 (checkpoints/12h_run/config.json)

```json
{
    "data_path": "D:\\数据补全\\hangmei_90_拼接好的.csv",
    "n_features": 44,
    "window_size": 48,
    "batch_size": 8,
    "latent_dim": 64,
    "hidden_units": 256,
    "k_spatial": 5,
    "k_temporal": 5,
    "p_drop": 0.1,
    "n_corrupted": 3,
    "lambda1": 1.0,
    "lambda2": 0.1,
    "lambda3": 0.1,
    "learning_rate": 0.001,
    "missing_rate": 0.2,
    "missing_type": "MCAR",
    "use_faiss": true,
    "dropout_rate": 0.3,
    "l2_reg": 0.005,
    "seed": 42
}
```

### 5.2 运行命令

```bash
python train.py --epochs 200 --batch_size 8 --hidden_units 256 --latent_dim 64 --dropout_rate 0.3 --l2_reg 0.005 --checkpoint_dir ./checkpoints/12h_run
```

### 5.3 完整训练日志 (20 epochs，早停)

| Epoch | Train Total | Train Recon | Train Consist | Train Space | Train Time | Val Total | Val Recon | Val Consist | Val Space | Val Time | LR |
|-------|-------------|-------------|---------------|-------------|------------|-----------|-----------|-------------|-----------|----------|-----|
| 1 | 1.4249 | 1.3321 | 0.0364 | 0.5604 | 0.0038 | 19.2274 | 19.1577 | 0.0278 | 0.4152 | 0.0036 | 0.001 |
| 2 | 0.9906 | 0.9750 | 0.0062 | 0.0928 | 0.0006 | 20.3436 | 20.3189 | 0.0096 | 0.1492 | 0.0014 | 0.001 |
| 3 | 0.6472 | 0.6236 | 0.0094 | 0.1412 | 0.0010 | 13.6025 | 13.5938 | 0.0027 | 0.0583 | 0.0015 | 0.001 |
| 4 | 0.5373 | 0.5180 | 0.0078 | 0.1140 | 0.0008 | 12.4467 | 12.4078 | 0.0164 | 0.2228 | 0.0021 | 0.001 |
| **5** | **0.5328** | **0.5148** | **0.0074** | **0.1056** | **0.0007** | **7.6081** | **7.5977** | **0.0039** | **0.0633** | **0.0012** | **0.001** |
| 6 | 0.4912 | 0.4711 | 0.0083 | 0.1174 | 0.0008 | 10.1623 | 10.1514 | 0.0031 | 0.0768 | 0.0012 | 0.001 |
| 7 | 0.4104 | 0.3947 | 0.0065 | 0.0904 | 0.0007 | 8.7577 | 8.7486 | 0.0034 | 0.0567 | 0.0012 | 0.001 |
| 8 | 0.3884 | 0.3758 | 0.0053 | 0.0724 | 0.0006 | 7.9997 | 7.9905 | 0.0035 | 0.0561 | 0.0012 | 0.001 |
| 9 | 0.3805 | 0.3700 | 0.0044 | 0.0609 | 0.0005 | 8.8994 | 8.8872 | 0.0046 | 0.0740 | 0.0011 | 0.001 |
| 10 | 0.3578 | 0.3476 | 0.0043 | 0.0585 | 0.0005 | 10.5657 | 10.5583 | 0.0024 | 0.0493 | 0.0012 | 0.0005 |
| 11 | 0.3550 | 0.3429 | 0.0051 | 0.0694 | 0.0006 | 7.8748 | 7.8617 | 0.0055 | 0.0743 | 0.0014 | 0.0005 |
| 12 | 0.3280 | 0.3168 | 0.0047 | 0.0642 | 0.0005 | 7.9352 | 7.9264 | 0.0032 | 0.0551 | 0.0011 | 0.0005 |
| 13 | 0.3192 | 0.3092 | 0.0042 | 0.0575 | 0.0005 | 7.7665 | 7.7575 | 0.0030 | 0.0577 | 0.0016 | 0.0005 |
| 14 | 0.3132 | 0.3020 | 0.0046 | 0.0648 | 0.0005 | 7.9637 | 7.9537 | 0.0032 | 0.0665 | 0.0014 | 0.0005 |
| 15 | 0.3021 | 0.2909 | 0.0046 | 0.0654 | 0.0005 | 8.1417 | 8.1304 | 0.0036 | 0.0757 | 0.0012 | 0.00025 |
| 16 | 0.2953 | 0.2834 | 0.0050 | 0.0686 | 0.0006 | 8.1587 | 8.1495 | 0.0030 | 0.0604 | 0.0015 | 0.00025 |
| 17 | 0.2887 | 0.2776 | 0.0046 | 0.0644 | 0.0005 | 7.9776 | 7.9637 | 0.0048 | 0.0892 | 0.0016 | 0.00025 |
| 18 | 0.2941 | 0.2832 | 0.0045 | 0.0631 | 0.0005 | 7.9492 | 7.9401 | 0.0032 | 0.0582 | 0.0013 | 0.00025 |
| 19 | 0.2928 | 0.2820 | 0.0045 | 0.0634 | 0.0005 | 8.1715 | 8.1632 | 0.0029 | 0.0530 | 0.0013 | 0.00025 |
| 20 | 0.2810 | 0.2707 | 0.0043 | 0.0600 | 0.0005 | 8.0534 | 8.0444 | 0.0032 | 0.0567 | 0.0012 | 0.000125 |

**训练结果:**
- 最佳验证损失: **7.6081** (Epoch 5)
- 早停触发: 验证损失未改善 (15/15)

### 5.4 终端输出 (最后一个epoch)

```
Training: 100%|█████████████████████████████████████████████████| 239/239 [04:47<00:00,  1.20s/it, loss=0.4753, recon=0.4595]
Validating...
  Train Loss: 0.2810 (recon: 0.2707, consist: 0.0043, space: 0.0600, time: 0.0005)
  Val Loss:   8.0534 (recon: 8.0444, consist: 0.0032, space: 0.0567, time: 0.0012)
  Learning Rate: 0.000250
  验证损失未改善 (15/15)
  [LR Scheduler] 学习率降低: 0.000250 -> 0.000125

早停触发! 最佳模型在 epoch 5

================================================================================
训练完成!
最佳验证损失: 7.6081 (Epoch 5)
================================================================================
```

### 5.5 学习率变化

```
Epoch 1-9:   0.001000
Epoch 10-14: 0.000500
Epoch 15-19: 0.000250
Epoch 20:    0.000125
```

### 5.6 评估结果 (results/gaussian_eval/metrics.json)

| 位置 | MSE | RMSE | MAE | R² | MAPE |
|------|-----|------|-----|-----|------|
| **缺失位置** | 1.9708 | 1.4039 | 1.0475 | **-0.7823** | 460.15% |
| 观测位置 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0% |
| 整体 | 0.3952 | 0.6287 | 0.2101 | 0.6407 | 92.28% |

---

## 6. 评估一致性与消融实验

### 6.1 evaluate.py 参数一致性验证

**结论: 评估参数与训练配置完全一致**

evaluate.py 通过动态加载检查点目录中的 `config.json` 来初始化模型，确保评估时使用与训练时相同的参数：

```python
# evaluate.py 第87-95行
model = AECS(
    n_features=config['n_features'],
    latent_dim=config['latent_dim'],          # ← 从config.json读取
    hidden_units=config['hidden_units'],       # ← 从config.json读取
    k_spatial=config['k_spatial'],
    k_temporal=config['k_temporal'],
    dropout_rate=config.get('dropout_rate', 0.2),  # ← 从config.json读取
    l2_reg=config.get('l2_reg', 0.001)             # ← 从config.json读取
)
```

**模型权重加载路径确认:**

| 实验 | 评估命令 | 实际加载的权重文件 |
|------|----------|-------------------|
| 实验A | `--checkpoint_dir ./checkpoints` | `checkpoints/best_model.weights.h5` |
| 实验B | `--checkpoint_dir ./checkpoints/12h_run` | `checkpoints/12h_run/best_model.weights.h5` |

```python
# evaluate.py 第98行
best_model_path = checkpoint_dir / 'best_model.weights.h5'
```

**参数一致性验证表:**

| 参数 | 实验A训练config | 实验A评估使用 | 实验B训练config | 实验B评估使用 |
|------|----------------|---------------|----------------|---------------|
| latent_dim | 32 | 32 ✓ | 64 | 64 ✓ |
| hidden_units | 128 | 128 ✓ | 256 | 256 ✓ |
| dropout_rate | 0.1 | 0.1 ✓ | 0.3 | 0.3 ✓ |
| l2_reg | 0.0005 | 0.0005 ✓ | 0.005 | 0.005 ✓ |
| k_spatial | 5 | 5 ✓ | 5 | 5 ✓ |
| k_temporal | 5 | 5 ✓ | 5 | 5 ✓ |

### 6.2 邻域消融实验

**结论: 本项目未执行邻域消融实验 (AE-only baseline)**

经代码库搜索，未发现以下内容：
- 任何包含 "ablation" 的文件或代码
- 禁用邻域分支 (disable neighbor) 的配置选项
- 纯AE重建 (AE-only) 的实验结果

**建议后续补充的消融实验:**

| 实验编号 | 配置 | 目的 |
|----------|------|------|
| Ablation-1 | λ2=0, λ3=0 (禁用L_space和L_time) | 验证邻域损失的贡献 |
| Ablation-2 | 移除 z_space 融合 (仅用 z_orig + z_time) | 验证空间邻域的贡献 |
| Ablation-3 | 移除 z_time 融合 (仅用 z_orig + z_space) | 验证时间邻域的贡献 |
| Ablation-4 | 仅用 z_orig (纯AE，无邻域) | 建立基线 |

---

## 7. 代码版本标识

### 7.1 版本控制状态

**本项目未使用Git版本控制**，因此无提交ID (commit hash)。以下通过显式代码改动清单标识实验A/B对应的代码状态。

### 7.2 实验A对应的代码状态 (ReLU版本)

**ae_cs.py 关键代码片段:**

```python
# Encoder.dense_latent - 无激活函数 (线性)
self.dense_latent = layers.Dense(
    latent_dim,
    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    name='latent'
)

# GatingNetwork.dense1 - 使用ReLU
self.dense1 = layers.Dense(
    latent_dim * 2,
    activation='relu',  # ← ReLU
    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    name='dense1'
)

# GatingNetwork.dense2 - 使用ReLU
self.dense2 = layers.Dense(
    latent_dim,
    activation='relu',  # ← ReLU
    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    name='dense2'
)
```

**train.py 默认参数:**
```python
parser.add_argument('--dropout_rate', type=float, default=0.1, ...)
parser.add_argument('--l2_reg', type=float, default=0.0005, ...)
```

### 7.3 实验B对应的代码状态 (高斯激活版本)

**ae_cs.py 完整改动清单:**

| 行号 | 改动类型 | 改动内容 |
|------|----------|----------|
| 16-24 | 新增 | 添加 `gaussian_activation` 函数定义 |
| 55 | 修改 | Encoder.dense_latent: `activation=gaussian_activation` |
| 181 | 修改 | GatingNetwork.dense1: `activation=gaussian_activation` |
| 187 | 修改 | GatingNetwork.dense2: `activation=gaussian_activation` |

**新增代码 (ae_cs.py 第16-24行):**
```python
@tf.keras.utils.register_keras_serializable(package='Custom', name='gaussian_activation')
def gaussian_activation(x):
    """高斯激活函数: f(x) = 1 - exp(-x²)"""
    return 1.0 - tf.exp(-tf.square(x))
```

**train.py 默认参数改动:**
```python
parser.add_argument('--dropout_rate', type=float, default=0.3, ...)   # 0.1 → 0.3
parser.add_argument('--l2_reg', type=float, default=0.005, ...)       # 0.0005 → 0.005
```

### 7.4 代码状态对照表

| 组件 | 实验A (ReLU) | 实验B (高斯激活) |
|------|-------------|-----------------|
| gaussian_activation函数 | 不存在 | 存在 (第16-24行) |
| Encoder.dense_latent激活 | None (线性) | gaussian_activation |
| GatingNetwork.dense1激活 | 'relu' | gaussian_activation |
| GatingNetwork.dense2激活 | 'relu' | gaussian_activation |
| Decoder激活 | None (线性) | None (线性) - 未修改 |
| train.py dropout_rate | 0.1 | 0.3 |
| train.py l2_reg | 0.0005 | 0.005 |

### 7.5 当前代码库状态

**当前代码库为实验B版本**（包含gaussian_activation）。

如需复现实验A，需将以下代码还原：

```python
# ae_cs.py - 移除gaussian_activation函数定义 (第16-24行)
# ae_cs.py - Encoder.dense_latent: 移除 activation=gaussian_activation
# ae_cs.py - GatingNetwork.dense1: activation='relu'
# ae_cs.py - GatingNetwork.dense2: activation='relu'
# train.py - --dropout_rate default=0.1
# train.py - --l2_reg default=0.0005
```

---

## 8. 评估结果对比

### 8.1 缺失位置指标对比

| 指标 | 实验A (ReLU) | 实验B (高斯) | 变化 |
|------|-------------|--------------|------|
| **R²** | **0.2969** | -0.7823 | ❌ -1.08 |
| **MAE** | **0.6303** | 1.0475 | ❌ +66% |
| **RMSE** | **0.8818** | 1.4039 | ❌ +59% |
| MAPE | 210.49% | 460.15% | ❌ +119% |

## 8.2 结论

**实验B（高斯激活+增强正则化）性能严重下降，原因分析:**

1. **高斯激活函数特性问题**
   - 输出范围 [0, 1)，压缩了特征表达能力
   - x=0时输出0，可能导致信息丢失

2. **模型容量与正则化矛盾**
   - 增大模型(hidden_units 128→256)的同时增强正则化(dropout 0.1→0.3)
   - 两者相互抵消，导致学习能力下降

3. **早停过早触发**
   - 最佳epoch在第5轮，模型可能未充分收敛

---

## 9. 运行环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 10/11 |
| Python版本 | 3.10.11 |
| TensorFlow版本 | 2.10.0 |
| CUDA版本 | 11.x |
| cuDNN版本 | 8.6.0 |
| GPU型号 | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU显存 | 8GB (可用约5.5GB) |

---

## 10. 核心代码附录

### 10.1 models/ae_cs.py (实验B版本，含gaussian_activation)

```python
"""
AE-CS: AutoEncoder based on Coherent denoising and Spatio-temporal
       neighborhood-preserving embedding

核心模型实现，包括：
1. Encoder (编码器): f_θ
2. Decoder (解码器): g_φ
3. Gating Network (门控网络): 用于自适应融合
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


class Encoder(Model):
    """
    编码器网络 f_θ

    输入: X ⊙ M (observed values) 和 M (mask matrix)
    输出: Z (latent representation)

    架构: 使用 LSTM 处理时间序列 (原GRU因cuDNN问题替换)
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
        # 使用自定义高斯激活函数: 1 - e^(-x²)
        self.dense_latent = layers.Dense(
            latent_dim,
            activation=gaussian_activation,
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

        # LSTM encoding
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

    架构: 使用 LSTM 进行时间序列重建
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
        # LSTM decoding
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
    """
    def __init__(self, latent_dim=64, name='gating_network',
                 dropout_rate=0.2, l2_reg=0.001):
        super(GatingNetwork, self).__init__(name=name)

        self.latent_dim = latent_dim

        # 输入维度: latent*3 (三个GAP) + 1 (缺失率ρ)
        # Attention layers with L2 regularization
        # 使用自定义高斯激活函数替代ReLU
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
        # 步骤1: 全局平均池化 (GAP) - 将时间维度压缩
        gap_orig = tf.reduce_mean(z_orig, axis=1)    # [batch, latent]
        gap_space = tf.reduce_mean(z_space, axis=1)  # [batch, latent]
        gap_time = tf.reduce_mean(z_time, axis=1)    # [batch, latent]

        # 步骤2: 拼接GAP结果和缺失率ρ
        if isinstance(missing_rate, (int, float)):
            batch_size = tf.shape(z_orig)[0]
            missing_rate = tf.fill([batch_size], float(missing_rate))
        elif len(missing_rate.shape) == 0:
            batch_size = tf.shape(z_orig)[0]
            missing_rate = tf.fill([batch_size], missing_rate)

        missing_rate_expanded = tf.expand_dims(missing_rate, axis=-1)  # [batch, 1]
        combined = tf.concat([gap_orig, gap_space, gap_time, missing_rate_expanded], axis=-1)

        # 步骤3: 计算全局注意力权重
        h1 = self.dense1(combined)
        h2 = self.dense2(h1)
        h2 = self.dropout(h2, training=training)
        alpha = self.dense_alpha(h2)  # [batch, 3]

        # 步骤4: 扩展权重到时间维度并融合
        alpha_expanded = tf.expand_dims(alpha, axis=1)
        alpha_1 = tf.expand_dims(alpha_expanded[:, :, 0], axis=-1)
        alpha_2 = tf.expand_dims(alpha_expanded[:, :, 1], axis=-1)
        alpha_3 = tf.expand_dims(alpha_expanded[:, :, 2], axis=-1)

        z_fused = alpha_1 * z_orig + alpha_2 * z_space + alpha_3 * z_time

        return alpha, z_fused


class AECS(Model):
    """完整的 AE-CS 模型"""
    def __init__(self, n_features, latent_dim=64, hidden_units=128,
                 k_spatial=5, k_temporal=5, use_partial_distance=True, use_variable_mapping=True,
                 dropout_rate=0.2, l2_reg=0.001, name='ae_cs'):
        super(AECS, self).__init__(name=name)

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.k_spatial = k_spatial
        self.k_temporal = k_temporal

        # Core components with regularization
        self.encoder = Encoder(latent_dim, hidden_units, dropout_rate=dropout_rate, l2_reg=l2_reg)
        self.decoder = Decoder(n_features, hidden_units, dropout_rate=dropout_rate, l2_reg=l2_reg)
        self.gating_network = GatingNetwork(latent_dim, dropout_rate=dropout_rate, l2_reg=l2_reg)

        # 导入邻域模块
        from .neighborhood import NeighborhoodModule
        self.neighborhood_module = NeighborhoodModule(k_spatial, k_temporal, use_partial_distance, use_variable_mapping)

    def call(self, x, mask, training=False, return_all=False):
        # Step 0: 计算整体缺失率ρ
        missing_rate = 1.0 - tf.reduce_mean(mask, axis=[1, 2])

        # Step 1: Encode to get original representation
        z_orig = self.encoder(x, mask, training=training)

        # Step 2: Compute spatial and temporal neighborhood embeddings
        z_space, z_time, neighborhood_info = self.neighborhood_module.compute_neighborhood_embeddings(
            x, z_orig, mask
        )

        # Step 3: Fuse representations using gating network
        alpha, z_fused = self.gating_network(z_orig, z_space, z_time, missing_rate, training=training)

        # Step 4: Decode
        x_hat = self.decoder(z_fused, training=training)

        # Step 5: 掩码融合输出
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

    def encode(self, x, mask, training=False):
        """仅编码"""
        return self.encoder(x, mask, training=training)

    def decode(self, z, training=False):
        """仅解码"""
        return self.decoder(z, training=training)
```

*(注: 完整的 neighborhood.py, losses.py, preprocessor.py, dataset.py, train.py 代码请参考源文件)*

---

## 11. 配置文件附录

### 11.1 实验A配置 (checkpoints/config.json)

```json
{
    "data_path": "D:\\数据补全\\hangmei_90_拼接好的.csv",
    "n_features": 44,
    "window_size": 48,
    "batch_size": 16,
    "latent_dim": 32,
    "hidden_units": 128,
    "k_spatial": 5,
    "k_temporal": 5,
    "p_drop": 0.1,
    "n_corrupted": 3,
    "lambda1": 1.0,
    "lambda2": 0.1,
    "lambda3": 0.1,
    "learning_rate": 0.001,
    "missing_rate": 0.2,
    "missing_type": "MCAR",
    "use_faiss": true,
    "dropout_rate": 0.1,
    "l2_reg": 0.0005,
    "seed": 42
}
```

### 11.2 实验B配置 (checkpoints/12h_run/config.json)

```json
{
    "data_path": "D:\\数据补全\\hangmei_90_拼接好的.csv",
    "n_features": 44,
    "window_size": 48,
    "batch_size": 8,
    "latent_dim": 64,
    "hidden_units": 256,
    "k_spatial": 5,
    "k_temporal": 5,
    "p_drop": 0.1,
    "n_corrupted": 3,
    "lambda1": 1.0,
    "lambda2": 0.1,
    "lambda3": 0.1,
    "learning_rate": 0.001,
    "missing_rate": 0.2,
    "missing_type": "MCAR",
    "use_faiss": true,
    "dropout_rate": 0.3,
    "l2_reg": 0.005,
    "seed": 42
}
```

### 11.3 实验A训练状态 (checkpoints/training_state.json) - 完整

```json
{
    "epoch": 30,
    "best_val_loss": 7.265402533424397,
    "train_loss_history": [
        {"total": 1.2173, "recon": 0.7923, "consist": 0.2448, "space": 1.7847, "time": 0.0176},
        {"total": 0.5317, "recon": 0.3814, "consist": 0.0887, "space": 0.6076, "time": 0.0078},
        {"total": 0.3633, "recon": 0.2902, "consist": 0.0429, "space": 0.2972, "time": 0.0051},
        {"total": 0.2806, "recon": 0.2415, "consist": 0.0227, "space": 0.1612, "time": 0.0034},
        {"total": 0.2315, "recon": 0.2082, "consist": 0.0132, "space": 0.0988, "time": 0.0024},
        {"total": 0.1993, "recon": 0.1841, "consist": 0.0084, "space": 0.0665, "time": 0.0019},
        {"total": 0.1724, "recon": 0.1615, "consist": 0.0059, "space": 0.0492, "time": 0.0016},
        {"total": 0.1667, "recon": 0.1574, "consist": 0.0048, "space": 0.0435, "time": 0.0015},
        {"total": 0.1479, "recon": 0.1404, "consist": 0.0038, "space": 0.0363, "time": 0.0012},
        {"total": 0.1429, "recon": 0.1361, "consist": 0.0033, "space": 0.0335, "time": 0.0013},
        {"total": 0.1392, "recon": 0.1327, "consist": 0.0031, "space": 0.0325, "time": 0.0012},
        {"total": 0.1285, "recon": 0.1215, "consist": 0.0034, "space": 0.0354, "time": 0.0014},
        {"total": 0.1277, "recon": 0.1212, "consist": 0.0030, "space": 0.0333, "time": 0.0012},
        {"total": 0.1213, "recon": 0.1156, "consist": 0.0026, "space": 0.0300, "time": 0.0010},
        {"total": 0.1114, "recon": 0.1062, "consist": 0.0024, "space": 0.0281, "time": 0.0009},
        {"total": 0.1122, "recon": 0.1072, "consist": 0.0022, "space": 0.0266, "time": 0.0009},
        {"total": 0.1010, "recon": 0.0968, "consist": 0.0019, "space": 0.0232, "time": 0.0007},
        {"total": 0.1045, "recon": 0.1006, "consist": 0.0017, "space": 0.0213, "time": 0.0007},
        {"total": 0.1039, "recon": 0.1001, "consist": 0.0016, "space": 0.0208, "time": 0.0006},
        {"total": 0.1006, "recon": 0.0968, "consist": 0.0016, "space": 0.0208, "time": 0.0007},
        {"total": 0.0994, "recon": 0.0958, "consist": 0.0015, "space": 0.0200, "time": 0.0006},
        {"total": 0.0972, "recon": 0.0938, "consist": 0.0014, "space": 0.0189, "time": 0.0006},
        {"total": 0.0920, "recon": 0.0887, "consist": 0.0014, "space": 0.0184, "time": 0.0006},
        {"total": 0.0946, "recon": 0.0912, "consist": 0.0014, "space": 0.0191, "time": 0.0006},
        {"total": 0.0918, "recon": 0.0887, "consist": 0.0013, "space": 0.0178, "time": 0.0005},
        {"total": 0.0927, "recon": 0.0898, "consist": 0.0012, "space": 0.0161, "time": 0.0005},
        {"total": 0.0894, "recon": 0.0864, "consist": 0.0012, "space": 0.0164, "time": 0.0005},
        {"total": 0.0874, "recon": 0.0845, "consist": 0.0012, "space": 0.0163, "time": 0.0005},
        {"total": 0.0895, "recon": 0.0868, "consist": 0.0011, "space": 0.0154, "time": 0.0005},
        {"total": 0.0874, "recon": 0.0848, "consist": 0.0011, "space": 0.0146, "time": 0.0004}
    ],
    "val_loss_history": [
        {"total": 17.5472, "recon": 17.5130, "consist": 0.0118, "space": 0.2091, "time": 0.0139},
        {"total": 12.8004, "recon": 12.7830, "consist": 0.0053, "space": 0.1007, "time": 0.0201},
        {"total": 10.4190, "recon": 10.4079, "consist": 0.0035, "space": 0.0630, "time": 0.0131},
        {"total": 9.5533, "recon": 9.5436, "consist": 0.0030, "space": 0.0581, "time": 0.0089},
        {"total": 8.7520, "recon": 8.7443, "consist": 0.0024, "space": 0.0459, "time": 0.0078},
        {"total": 9.1551, "recon": 9.1478, "consist": 0.0022, "space": 0.0446, "time": 0.0060},
        {"total": 8.1136, "recon": 8.1076, "consist": 0.0019, "space": 0.0365, "time": 0.0051},
        {"total": 7.8183, "recon": 7.8117, "consist": 0.0020, "space": 0.0407, "time": 0.0050},
        {"total": 8.0538, "recon": 8.0476, "consist": 0.0018, "space": 0.0396, "time": 0.0041},
        {"total": 7.9189, "recon": 7.9123, "consist": 0.0020, "space": 0.0423, "time": 0.0042},
        {"total": 7.4868, "recon": 7.4800, "consist": 0.0022, "space": 0.0419, "time": 0.0045},
        {"total": 7.9506, "recon": 7.9408, "consist": 0.0032, "space": 0.0604, "time": 0.0048},
        {"total": 7.7033, "recon": 7.6948, "consist": 0.0023, "space": 0.0580, "time": 0.0041},
        {"total": 7.6085, "recon": 7.6011, "consist": 0.0022, "space": 0.0486, "time": 0.0032},
        {"total": 7.5255, "recon": 7.5165, "consist": 0.0024, "space": 0.0623, "time": 0.0039},
        {"total": 7.7453, "recon": 7.7371, "consist": 0.0022, "space": 0.0572, "time": 0.0034},
        {"total": 7.5917, "recon": 7.5849, "consist": 0.0019, "space": 0.0469, "time": 0.0027},
        {"total": 7.4820, "recon": 7.4752, "consist": 0.0018, "space": 0.0471, "time": 0.0029},
        {"total": 7.2654, "recon": 7.2591, "consist": 0.0017, "space": 0.0427, "time": 0.0033},
        {"total": 7.6175, "recon": 7.6109, "consist": 0.0019, "space": 0.0450, "time": 0.0028},
        {"total": 7.3856, "recon": 7.3794, "consist": 0.0017, "space": 0.0432, "time": 0.0026},
        {"total": 7.3245, "recon": 7.3179, "consist": 0.0017, "space": 0.0464, "time": 0.0027},
        {"total": 7.7738, "recon": 7.7680, "consist": 0.0016, "space": 0.0401, "time": 0.0025},
        {"total": 7.3251, "recon": 7.3191, "consist": 0.0017, "space": 0.0408, "time": 0.0026},
        {"total": 7.4808, "recon": 7.4751, "consist": 0.0016, "space": 0.0393, "time": 0.0022},
        {"total": 7.3821, "recon": 7.3770, "consist": 0.0014, "space": 0.0350, "time": 0.0021},
        {"total": 7.3629, "recon": 7.3576, "consist": 0.0015, "space": 0.0365, "time": 0.0022},
        {"total": 7.4162, "recon": 7.4107, "consist": 0.0015, "space": 0.0377, "time": 0.0023},
        {"total": 7.6576, "recon": 7.6523, "consist": 0.0014, "space": 0.0365, "time": 0.0020},
        {"total": 7.6092, "recon": 7.6043, "consist": 0.0013, "space": 0.0341, "time": 0.0019}
    ],
    "lr_history": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.000125, 0.000125]
}
```

---

*报告生成完毕*

**已补充内容:**
- 第6节：评估参数一致性验证、邻域消融实验状态说明
- 第7节：代码版本标识（实验A/B代码状态对照）
- 配置冲突已纠正，完整训练日志已提供，明确区分两次实验
