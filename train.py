#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本 - AE-CS模型

实现专利中的完整训练流程：
- 三编码器-单解码器架构
- 一致性去噪机制（多掩码增强 + 加权一致性约束）
- 时空双重邻域保持
- 自适应融合机制
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json

# ===== GPU内存配置 (必须在导入其他TF模块之前!) =====
# 设置环境变量限制GPU内存
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # 禁用自动增长
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # 使用异步分配器

import tensorflow as tf

GPU_MEMORY_LIMIT_MB = 1024

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置硬性内存限制，防止 OOM 和系统死机
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_MB)]
        )
        print(f"[OK] GPU内存硬限制已设置: {GPU_MEMORY_LIMIT_MB}MB")
    except RuntimeError as e:
        print(f"[ERROR] GPU配置失败（可能TensorFlow已初始化）: {e}")
        print("[INFO] 请确保此脚本是第一个导入TensorFlow的模块")
else:
    print("[INFO] 未检测到GPU，将使用CPU训练")
# ======================================

import numpy as np
from tqdm import tqdm

# 导入自定义模块
from data import AECSDataLoader
from models.ae_cs import AECS
from models.losses import total_loss
from data.preprocessor import BernoulliCorruptor

# 导入内存管理工具
from utils.memory_utils import MemoryManager, setup_memory_management, force_memory_cleanup


class AECSTrainer:
    """
    AE-CS模型训练器

    实现论文Algorithm 1
    """
    def __init__(self,
                 data_path: str,
                 n_features: int = 44,
                 window_size: int = 48,
                 batch_size: int = 32,
                 latent_dim: int = 64,
                 hidden_units: int = 128,
                 k_spatial: int = 5,
                 k_temporal: int = 5,
                 p_drop: float = 0.1,
                 n_corrupted: int = 3,
                 lambda1: float = 1.0,
                 lambda2: float = 0.1,
                 lambda3: float = 0.1,
                 lambda4: float = 0.0,
                 learning_rate: float = 1e-3,
                 missing_rate: float = 0.2,
                 missing_type: str = 'MCAR',
                 use_faiss: bool = True,
                 dropout_rate: float = 0.2,
                 l2_reg: float = 0.001,
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs',
                 seed: int = 42,
                 use_lr_scheduler: bool = True,
                 lr_patience: int = 5,
                 lr_factor: float = 0.5,
                 min_lr: float = 1e-6):
        """
        初始化训练器

        Args:
            data_path: 数据文件路径
            n_features: 特征数量
            window_size: 时间窗口大小
            batch_size: 批大小
            latent_dim: 潜在空间维度
            hidden_units: 隐藏单元数
            k_spatial: 空间邻域数量
            k_temporal: 时间邻域数量
            p_drop: Bernoulli损坏概率
            n_corrupted: 损坏版本数量K
            lambda1, lambda2, lambda3: 损失函数权重
            learning_rate: 初始学习率
            missing_rate: 缺失率
            missing_type: 缺失类型
            use_faiss: 是否使用FAISS加速
            dropout_rate: Dropout概率
            l2_reg: L2正则化系数
            checkpoint_dir: 检查点保存目录
            log_dir: 日志保存目录
            seed: 随机种子
            use_lr_scheduler: 是否使用学习率调度器
            lr_patience: 学习率调度器的耐心值
            lr_factor: 学习率衰减因子
            min_lr: 最小学习率
        """
        self.data_path = data_path
        self.n_features = n_features
        self.window_size = window_size
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.k_spatial = k_spatial
        self.k_temporal = k_temporal
        self.p_drop = p_drop
        self.n_corrupted = n_corrupted
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.learning_rate = learning_rate
        self.missing_rate = missing_rate
        self.missing_type = missing_type
        self.use_faiss = use_faiss
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.seed = seed

        # 学习率调度器参数
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr

        # 创建目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # 初始化组件
        self.model = None
        self.optimizer = None
        self.data_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.corruptor = None

        # 训练状态
        self.current_epoch = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')

        # 学习率调度器状态
        self.lr_patience_counter = 0
        self.current_lr = learning_rate
        self.lr_history = []

        # 内存管理器（防止碎片化）
        self.memory_manager = None

    def setup(self):
        """
        设置训练环境
        """
        print("=" * 80)
        print("设置训练环境")
        print("=" * 80)

        # 1. 初始化数据加载器
        print("\n1. 初始化数据加载器...")
        self.data_loader = AECSDataLoader(
            batch_size=self.batch_size,
            shuffle_train=True,
            seed=self.seed
        )
        self.data_loader.preprocessor.data_path = Path(self.data_path)
        self.data_loader.preprocessor.window_size = self.window_size

        # 准备数据
        datasets = self.data_loader.prepare(
            missing_rate=self.missing_rate,
            missing_type=self.missing_type,
            train_ratio=0.7,
            val_ratio=0.15
        )

        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']
        self.test_dataset = datasets['test']

        print(f"  训练集: {len(self.train_dataset)} 样本")
        print(f"  验证集: {len(self.val_dataset)} 样本")
        print(f"  测试集: {len(self.test_dataset)} 样本")

        # 2. 初始化模型
        print("\n2. 初始化模型...")
        self.model = AECS(
            n_features=self.n_features,
            latent_dim=self.latent_dim,
            hidden_units=self.hidden_units,
            k_spatial=self.k_spatial,
            k_temporal=self.k_temporal,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg
        )

        print(f"  模型参数（三编码器-单解码器架构）:")
        print(f"    - 第一编码器：一致性去噪编码器（零值填充输入）")
        print(f"    - 第二编码器：空间邻域编码器（空间KNN初始化）")
        print(f"    - 第三编码器：时间邻域编码器（时间KNN初始化）")
        print(f"    - 特征数: {self.n_features}")
        print(f"    - 潜在维度: {self.latent_dim}")
        print(f"    - 隐藏单元: {self.hidden_units}")
        print(f"    - 空间邻域: {self.k_spatial}")
        print(f"    - 时间邻域: {self.k_temporal}")
        print(f"    - FAISS加速: {self.use_faiss}")
        print(f"    - Dropout率: {self.dropout_rate}")
        print(f"    - L2正则化: {self.l2_reg}")

        # 3. 初始化优化器
        print("\n3. 初始化优化器...")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        print(f"  学习率: {self.learning_rate}")

        # 4. 初始化Bernoulli损坏器
        print("\n4. 初始化Bernoulli损坏器...")
        self.corruptor = BernoulliCorruptor(p_drop=self.p_drop)
        print(f"  丢弃概率: {self.p_drop}")
        print(f"  损坏版本数: {self.n_corrupted}")

        # 5. 保存配置
        self.save_config()

        # 6. 初始化内存管理器（防止碎片化）
        print("\n5. 初始化内存管理器...")
        setup_memory_management()
        self.memory_manager = MemoryManager(cleanup_interval=25, verbose=True)
        print("  内存清理间隔: 每25步")

        print("\n" + "=" * 80)
        print("训练环境设置完成!")
        print("=" * 80)

    def save_config(self):
        """保存训练配置"""
        config = {
            'data_path': str(self.data_path),
            'n_features': self.n_features,
            'window_size': self.window_size,
            'batch_size': self.batch_size,
            'latent_dim': self.latent_dim,
            'hidden_units': self.hidden_units,
            'k_spatial': self.k_spatial,
            'k_temporal': self.k_temporal,
            'p_drop': self.p_drop,
            'n_corrupted': self.n_corrupted,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3,
            'lambda4': self.lambda4,
            'learning_rate': self.learning_rate,
            'missing_rate': self.missing_rate,
            'missing_type': self.missing_type,
            'use_faiss': self.use_faiss,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'seed': self.seed
        }

        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"\n配置已保存到: {config_path}")

    def train_step(self, X, M):
        """
        单步训练 - 实现Algorithm 1的核心循环

        注意: 此方法不能使用@tf.function装饰器，因为:
        1. FAISS k-NN搜索需要numpy数组
        2. BernoulliCorruptor使用numpy操作
        虽然没有@tf.function会略微降低性能(~10-20%)，但FAISS加速已经提供了显著的整体性能提升

        Args:
            X: [batch, time, features] - 输入数据
            M: [batch, time, features] - 掩码

        Returns:
            losses_dict: 损失字典
        """
        with tf.GradientTape() as tape:
            # Step 1: 前向传播，获取所有中间结果
            outputs = self.model(X, M, training=True, return_all=True)

            x_hat = outputs['x_hat']
            z_orig = outputs['z_orig']
            neighborhood_info = outputs['neighborhood_info']
            x_knn_space = outputs['x_space_init']
            x_knn_time = outputs['x_time_init']

            # Step 2: 生成K个损坏版本并编码
            z_corrupted_list = []
            corrupted_masks = self.corruptor.corrupt(M.numpy(), n_versions=self.n_corrupted, seed=None)

            for corrupted_mask in corrupted_masks:
                corrupted_mask_tf = tf.constant(corrupted_mask, dtype=tf.float32)
                z_corrupted = self.model.encode(X, corrupted_mask_tf, training=True)
                z_corrupted_list.append(z_corrupted)

            # Step 3: 计算损坏版本的权重
            weights = self.corruptor.compute_weights(M.numpy(), corrupted_masks)

            # Step 4: 计算总损失
            loss, losses_dict = total_loss(
                x_true=X,
                x_pred=x_hat,
                mask=M,
                z_orig=z_orig,
                z_corrupted_list=z_corrupted_list,
                neighborhood_info=neighborhood_info,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
                lambda3=self.lambda3,
                lambda4=self.lambda4,
                corruption_weights=weights,
                x_knn_space=x_knn_space,
                x_knn_time=x_knn_time
            )

        # Step 5: 计算梯度并更新
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return losses_dict

    def validate(self):
        """
        验证循环

        Returns:
            avg_loss: 平均验证损失
            losses_dict: 损失详情
        """
        val_dataset = self.val_dataset.get_dataset()

        total_losses = {
            'total': 0.0,
            'recon': 0.0,
            'consist': 0.0,
            'space': 0.0,
            'time': 0.0,
            'missing': 0.0
        }
        n_batches = 0

        for X, M in val_dataset:
            # 前向传播
            outputs = self.model(X, M, training=False, return_all=True)

            x_hat = outputs['x_hat']
            z_orig = outputs['z_orig']
            neighborhood_info = outputs['neighborhood_info']
            x_knn_space = outputs['x_space_init']
            x_knn_time = outputs['x_time_init']

            # 生成损坏版本
            z_corrupted_list = []
            corrupted_masks = self.corruptor.corrupt(M.numpy(), n_versions=self.n_corrupted, seed=42)

            for corrupted_mask in corrupted_masks:
                corrupted_mask_tf = tf.constant(corrupted_mask, dtype=tf.float32)
                z_corrupted = self.model.encode(X, corrupted_mask_tf, training=False)
                z_corrupted_list.append(z_corrupted)

            # 计算权重
            weights = self.corruptor.compute_weights(M.numpy(), corrupted_masks)

            # 计算损失
            _, losses_dict = total_loss(
                x_true=X,
                x_pred=x_hat,
                mask=M,
                z_orig=z_orig,
                z_corrupted_list=z_corrupted_list,
                neighborhood_info=neighborhood_info,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
                lambda3=self.lambda3,
                lambda4=self.lambda4,
                corruption_weights=weights,
                x_knn_space=x_knn_space,
                x_knn_time=x_knn_time
            )

            # 累积损失
            for key in total_losses.keys():
                total_losses[key] += losses_dict[key].numpy()

            n_batches += 1

        # 计算平均损失
        avg_losses = {key: val / n_batches for key, val in total_losses.items()}

        return avg_losses['total'], avg_losses

    def train(self, epochs: int = 100, early_stopping_patience: int = 10):
        """
        训练循环

        Args:
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
        """
        print("\n" + "=" * 80)
        print(f"开始训练 - {epochs} epochs")
        print("=" * 80)

        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # 训练阶段
            print(f"\nEpoch {self.current_epoch}/{epochs}")
            train_dataset = self.train_dataset.get_dataset()

            epoch_losses = {
                'total': 0.0,
                'recon': 0.0,
                'consist': 0.0,
                'space': 0.0,
                'time': 0.0,
                'missing': 0.0
            }
            n_batches = 0

            # 训练循环
            progress_bar = tqdm(train_dataset, desc="Training")
            for X, M in progress_bar:
                losses_dict = self.train_step(X, M)

                # 累积损失
                for key in epoch_losses.keys():
                    epoch_losses[key] += losses_dict[key].numpy()

                n_batches += 1

                # 内存管理（防止碎片化）
                if self.memory_manager:
                    self.memory_manager.step()

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{losses_dict['total'].numpy():.4f}",
                    'recon': f"{losses_dict['recon'].numpy():.4f}"
                })

            # 计算平均损失
            avg_losses = {key: val / n_batches for key, val in epoch_losses.items()}
            self.train_loss_history.append(avg_losses)

            # 验证阶段
            print("Validating...")
            force_memory_cleanup()  # 验证前清理内存
            val_loss, val_losses = self.validate()
            force_memory_cleanup()  # 验证后清理内存
            self.val_loss_history.append(val_losses)

            # 打印损失
            print(f"  Train Loss: {avg_losses['total']:.4f} "
                  f"(recon: {avg_losses['recon']:.4f}, "
                  f"consist: {avg_losses['consist']:.4f}, "
                  f"space: {avg_losses['space']:.4f}, "
                  f"time: {avg_losses['time']:.4f}, "
                  f"missing: {avg_losses['missing']:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} "
                  f"(recon: {val_losses['recon']:.4f}, "
                  f"consist: {val_losses['consist']:.4f}, "
                  f"space: {val_losses['space']:.4f}, "
                  f"time: {val_losses['time']:.4f}, "
                  f"missing: {val_losses['missing']:.4f})")
            print(f"  Learning Rate: {self.current_lr:.6f}")

            # 检查是否是最佳模型
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = self.current_epoch
                patience_counter = 0
                improved = True

                # 保存最佳模型
                self.save_checkpoint(is_best=True)
                print(f"  [OK] 新的最佳验证损失! 模型已保存.")
            else:
                patience_counter += 1
                print(f"  验证损失未改善 ({patience_counter}/{early_stopping_patience})")

            # 学习率调度器
            if self.use_lr_scheduler:
                if not improved:
                    self.lr_patience_counter += 1
                    if self.lr_patience_counter >= self.lr_patience:
                        old_lr = self.current_lr
                        self.current_lr = max(self.current_lr * self.lr_factor, self.min_lr)
                        if self.current_lr < old_lr:
                            # 更新optimizer的学习率
                            self.optimizer.learning_rate.assign(self.current_lr)
                            print(f"  [LR Scheduler] 学习率降低: {old_lr:.6f} -> {self.current_lr:.6f}")
                            self.lr_patience_counter = 0
                else:
                    self.lr_patience_counter = 0

            # 记录学习率
            self.lr_history.append(self.current_lr)

            # 定期保存检查点
            if self.current_epoch % 10 == 0:
                self.save_checkpoint(is_best=False)

            # 早停检查
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发! 最佳模型在 epoch {best_epoch}")
                break

        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.4f} (Epoch {best_epoch})")
        print("=" * 80)

    def save_checkpoint(self, is_best=False):
        """
        保存检查点

        Args:
            is_best: 是否是最佳模型
        """
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.weights.h5'
        else:
            checkpoint_path = self.checkpoint_dir / f'model_epoch_{self.current_epoch}.weights.h5'

        # 保存权重（子类化模型不能保存为完整模型）
        self.model.save_weights(checkpoint_path)

        # 保存训练状态
        state = {
            'epoch': self.current_epoch,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'best_val_loss': self.best_val_loss,
            'lr_history': self.lr_history
        }

        state_path = self.checkpoint_dir / 'training_state.json'
        with open(state_path, 'w') as f:
            # 转换numpy类型为Python原生类型
            state_serializable = {
                'epoch': state['epoch'],
                'best_val_loss': float(state['best_val_loss']),
                'train_loss_history': [
                    {k: float(v) for k, v in losses.items()}
                    for losses in state['train_loss_history']
                ],
                'val_loss_history': [
                    {k: float(v) for k, v in losses.items()}
                    for losses in state['val_loss_history']
                ],
                'lr_history': [float(lr) for lr in state['lr_history']]
            }
            json.dump(state_serializable, f, indent=4)

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
        """
        # 加载权重（需要先初始化模型）
        if self.model is None:
            raise ValueError("Model not initialized. Call setup() first.")
        self.model.load_weights(checkpoint_path)
        print(f"模型权重已从 {checkpoint_path} 加载")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练AE-CS模型')
    parser.add_argument('--data_path', type=str, default=r'D:\数据补全\hangmei_90_拼接好的.csv',
                        help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='潜在空间维度')
    parser.add_argument('--hidden_units', type=int, default=128,
                        help='GRU隐藏单元数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True,
                        help='是否使用学习率调度器')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='学习率调度器的耐心值')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='学习率衰减因子')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='最小学习率')
    parser.add_argument('--missing_rate', type=float, default=0.2,
                        help='缺失率')
    parser.add_argument('--missing_type', type=str, default='MCAR',
                        choices=['MCAR', 'MAR', 'MNAR'],
                        help='缺失类型')
    parser.add_argument('--use_faiss', action='store_true', default=True,
                        help='是否使用FAISS加速')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='一致性损失权重')
    parser.add_argument('--lambda2', type=float, default=0.1,
                        help='空间损失权重')
    parser.add_argument('--lambda3', type=float, default=0.1,
                        help='时间损失权重')
    parser.add_argument('--lambda4', type=float, default=0.0,
                        help='[已禁用] 伪标签损失权重，默认为 0')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout概率 (增强正则化: 0.1→0.3)')
    parser.add_argument('--l2_reg', type=float, default=0.005,
                        help='L2正则化系数 (增强正则化: 0.0005→0.005)')

    args = parser.parse_args()

    # 初始化训练器
    trainer = AECSTrainer(
        data_path=args.data_path,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        missing_rate=args.missing_rate,
        missing_type=args.missing_type,
        use_faiss=args.use_faiss,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        use_lr_scheduler=args.use_lr_scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr
    )

    # 设置训练环境
    trainer.setup()

    # 开始训练
    trainer.train(epochs=args.epochs, early_stopping_patience=args.early_stopping_patience)


if __name__ == '__main__':
    main()
