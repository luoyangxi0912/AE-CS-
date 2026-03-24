"""
训练脚本 - AE-CS模型 (V9: 修复 delta≈0 问题)

架构: 三编码器-单解码器, KNN初始化, 门控融合
训练: 自监督掩码重建 + 一致性去噪 + 邻域保持损失

V9 关键修复:
  - L_recon 仅在额外掩盖的位置计算 (消除 80% 零信号干扰)
  - p_drop=0.5 (增加有用梯度信号占比)
  - stride=12 (增加训练样本)
  - delta clip [-2.0, 2.0] (放宽表达能力)

用法:
    python train_iterative.py --checkpoint_dir checkpoints_v9 --epochs 100
"""

# ===== 强制使用 CPU =====
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
print("[INFO] 强制使用 CPU 训练")

import numpy as np
import json
import argparse
from pathlib import Path

from models.ae_cs import AECS
from data import AECSDataLoader
from models.losses import (
    total_loss,
    generate_augmented_masks,
)


class AECSTrainerV2:
    """AE-CS 训练器 (V9: 修复 delta 学习信号)"""

    def __init__(self, model, train_dataset, val_dataset,
                 checkpoint_dir='checkpoints_v9',
                 learning_rate=0.001,
                 lambda1=0.5, lambda2=0.5, lambda3=0.5,
                 p_drop=0.2,
                 n_corrupted=3,
                 p_consist=0.1,
                 use_lr_scheduler=True,
                 lr_patience=5,
                 lr_factor=0.5,
                 min_lr=1e-6):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.current_lr = learning_rate

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.p_drop = p_drop
        self.n_corrupted = n_corrupted
        self.p_consist = p_consist

        self.use_lr_scheduler = use_lr_scheduler
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.lr_patience_counter = 0

        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0

    def train_step(self, X_batch, mask_batch):
        """单步训练 - V9: L_recon 仅在额外掩盖位置计算"""
        with tf.GradientTape() as tape:
            # Step 1: 自监督去噪 corruption
            bernoulli = tf.cast(
                tf.random.uniform(tf.shape(mask_batch)) > self.p_drop,
                tf.float32
            )
            corrupted_mask = mask_batch * bernoulli

            # Step 2: 前向传播 (用 corrupted_mask)
            outputs = self.model(
                X_batch, corrupted_mask, training=True, return_all=True
            )
            x_hat = outputs['x_hat']
            z_orig = outputs['z_orig']
            neighborhood_info = outputs['neighborhood_info']

            # Step 3: 一致性损失 (从原始 mask 生成)
            consistency_masks, consistency_weights = generate_augmented_masks(
                mask_batch, Q=self.n_corrupted,
                p_drop=self.p_consist
            )
            z_corrupted_list = []
            for cm in consistency_masks:
                z_corrupted = self.model.encode(X_batch, cm, training=True)
                z_corrupted_list.append(z_corrupted)

            # Step 4: V9 核心修复 — recon_mask 仅覆盖额外掩盖位置
            # 这些位置的 x_space_init = KNN填充 ≠ x_true → delta 目标非零
            # 消除了旧版 80% 零信号淹没 20% 有用信号的问题
            recon_mask = mask_batch * (1.0 - corrupted_mask)

            loss, losses_dict = total_loss(
                x_true=X_batch,
                x_pred=x_hat,
                mask=mask_batch,
                z_orig=z_orig,
                z_corrupted_list=z_corrupted_list,
                neighborhood_info=neighborhood_info,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
                lambda3=self.lambda3,
                corruption_weights=consistency_weights,
                recon_mask=recon_mask,
            )

        # 梯度裁剪 + 更新
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g
                     for g in gradients]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        return {k: float(v.numpy()) for k, v in losses_dict.items()}

    def validate(self):
        """验证 - 同样应用 V9 修复"""
        val_losses = []
        val_rng = tf.random.Generator.from_seed(12345)

        for X_batch, mask_batch in self.val_dataset.get_dataset():
            bernoulli = tf.cast(
                val_rng.uniform(tf.shape(mask_batch)) > self.p_drop,
                tf.float32
            )
            corrupted_mask = mask_batch * bernoulli

            outputs = self.model(
                X_batch, corrupted_mask, training=False, return_all=True
            )
            x_hat = outputs['x_hat']
            z_orig = outputs['z_orig']
            neighborhood_info = outputs['neighborhood_info']

            consistency_masks, consistency_weights = generate_augmented_masks(
                mask_batch, Q=self.n_corrupted,
                p_drop=self.p_consist
            )
            z_corrupted_list = []
            for cm in consistency_masks:
                z_corrupted = self.model.encode(X_batch, cm, training=False)
                z_corrupted_list.append(z_corrupted)

            # V9: recon_mask 仅额外掩盖位置
            recon_mask = mask_batch * (1.0 - corrupted_mask)

            _, losses_dict = total_loss(
                x_true=X_batch, x_pred=x_hat,
                mask=mask_batch,
                z_orig=z_orig,
                z_corrupted_list=z_corrupted_list,
                neighborhood_info=neighborhood_info,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
                lambda3=self.lambda3,
                corruption_weights=consistency_weights,
                recon_mask=recon_mask,
            )

            val_losses.append(
                {k: float(v.numpy()) for k, v in losses_dict.items()}
            )

        avg_loss = {
            k: np.mean([l[k] for l in val_losses])
            for k in val_losses[0].keys()
        }
        return avg_loss

    def train(self, epochs=100, early_stopping_patience=15, resume=False):
        """训练主循环"""
        start_epoch = 0
        patience_counter = 0
        best_epoch = 0

        if resume:
            start_epoch = self.resume_from_checkpoint()
            best_epoch = start_epoch

        print(f"\n{'='*70}")
        print(f"AE-CS V9 训练 (修复 delta 学习信号)")
        print(f"  p_drop={self.p_drop}, n_corrupted={self.n_corrupted}, "
              f"p_consist={self.p_consist}")
        print(f"  lambda1={self.lambda1}, lambda2={self.lambda2}, "
              f"lambda3={self.lambda3}")
        print(f"  LR scheduler: {self.use_lr_scheduler} "
              f"(patience={self.lr_patience}, factor={self.lr_factor})")
        print(f"{'='*70}")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch + 1
            epoch_losses = []

            train_ds = self.train_dataset.get_dataset()
            for batch_idx, (X_batch, mask_batch) in enumerate(train_ds):
                loss_dict = self.train_step(X_batch, mask_batch)
                epoch_losses.append(loss_dict)

                if (batch_idx + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1} Batch {batch_idx+1}: "
                          f"total={loss_dict['total']:.4f} "
                          f"recon={loss_dict['recon']:.4f}")

            train_loss = {
                k: np.mean([l[k] for l in epoch_losses])
                for k in epoch_losses[0].keys()
            }
            self.train_loss_history.append(train_loss)

            val_loss = self.validate()
            self.val_loss_history.append(val_loss)
            self.lr_history.append(float(self.optimizer.learning_rate.numpy()))

            print(f"Epoch {epoch+1}/{epochs}  LR={self.current_lr:.6f}")
            print(f"  Train: total={train_loss['total']:.4f}  "
                  f"recon={train_loss['recon']:.4f}  "
                  f"consist={train_loss['consist']:.4f}  "
                  f"space={train_loss['space']:.4f}  "
                  f"time={train_loss['time']:.4f}")
            print(f"  Val:   total={val_loss['total']:.4f}  "
                  f"recon={val_loss['recon']:.4f}  "
                  f"consist={val_loss['consist']:.4f}  "
                  f"space={val_loss['space']:.4f}  "
                  f"time={val_loss['time']:.4f}")

            improved = val_loss['recon'] < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss['recon']
                best_epoch = epoch + 1
                patience_counter = 0
                self.save_checkpoint(epoch + 1, is_best=True)
                print(f"  [BEST] val_recon={self.best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  未改善 ({patience_counter}/{early_stopping_patience})")

            if self.use_lr_scheduler:
                if not improved:
                    self.lr_patience_counter += 1
                    if self.lr_patience_counter >= self.lr_patience:
                        old_lr = self.current_lr
                        self.current_lr = max(
                            self.current_lr * self.lr_factor, self.min_lr
                        )
                        if self.current_lr < old_lr:
                            self.optimizer.learning_rate.assign(self.current_lr)
                            print(f"  [LR] {old_lr:.6f} -> {self.current_lr:.6f}")
                        self.lr_patience_counter = 0
                else:
                    self.lr_patience_counter = 0

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, is_best=False)

            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发! 最佳模型在 Epoch {best_epoch}")
                break

        print(f"\n训练完成! 最佳验证 recon 损失: {self.best_val_loss:.4f} "
              f"(Epoch {best_epoch})")
        return self.best_val_loss

    def save_checkpoint(self, epoch, is_best=False):
        def to_native(obj):
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            return obj

        state = {
            'epoch': epoch,
            'best_val_loss': float(self.best_val_loss),
            'train_loss_history': to_native(self.train_loss_history),
            'val_loss_history': to_native(self.val_loss_history),
            'lr_history': [float(lr) for lr in self.lr_history]
        }
        with open(self.checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(state, f, indent=4)

        if is_best:
            self.model.save_weights(
                str(self.checkpoint_dir / 'best_model.weights.h5')
            )

        if epoch % 10 == 0:
            self.model.save_weights(
                str(self.checkpoint_dir / f'model_epoch_{epoch}.weights.h5')
            )

    def resume_from_checkpoint(self):
        state_path = self.checkpoint_dir / 'training_state.json'
        best_path = self.checkpoint_dir / 'best_model.weights.h5'

        if not state_path.exists() or not best_path.exists():
            print("[INFO] 未找到检查点，从头开始")
            return 0

        with open(state_path, 'r') as f:
            state = json.load(f)

        self.current_epoch = state['epoch']
        self.best_val_loss = state['best_val_loss']
        self.train_loss_history = state['train_loss_history']
        self.val_loss_history = state['val_loss_history']
        self.lr_history = state.get('lr_history', [])

        if self.lr_history:
            self.current_lr = self.lr_history[-1]
            self.optimizer.learning_rate.assign(self.current_lr)

        for batch in self.train_dataset.get_dataset().take(1):
            X, M = batch
            _ = self.model(X, M, training=False)

        self.model.load_weights(str(best_path))
        print(f"[OK] 从 Epoch {self.current_epoch} 恢复 "
              f"(best_val={self.best_val_loss:.4f}, lr={self.current_lr:.6f})")
        return self.current_epoch


def main():
    parser = argparse.ArgumentParser(
        description='AE-CS V9 训练 (修复 delta 学习信号)'
    )

    parser.add_argument('--data_path', type=str,
                        default='hangmei_90_拼接好的.csv')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints_v9')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--early_stopping_patience', type=int, default=15)

    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--l2_reg', type=float, default=0.0005)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    parser.add_argument('--p_drop', type=float, default=0.2,
                        help='去噪掩盖比例 (V10: 0.5→0.2, 减少 train-test 分布偏移)')
    parser.add_argument('--n_corrupted', type=int, default=3)
    parser.add_argument('--p_consist', type=float, default=0.1)

    parser.add_argument('--lambda1', type=float, default=0.5)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--lambda3', type=float, default=0.5)

    parser.add_argument('--stride', type=int, default=12,
                        help='窗口滑动步长 (V9: 24→12, 增加训练样本)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("AE-CS V9 训练 (修复 delta 学习信号)")
    print("=" * 70)

    # 1. 数据加载
    print("\n[1] 加载数据...")
    loader = AECSDataLoader(
        batch_size=args.batch_size,
        shuffle_train=True,
        seed=args.seed
    )
    loader.preprocessor.stride = args.stride

    data_path = args.data_path
    if not os.path.exists(data_path):
        if os.path.exists(os.path.basename(data_path)):
            data_path = os.path.basename(data_path)

    loader.preprocessor.data_path = Path(data_path)

    datasets = loader.prepare(
        missing_rate=0.2,
        missing_type='MCAR',
        train_ratio=0.7,
        val_ratio=0.15
    )

    train_dataset = datasets['train']
    val_dataset = datasets['val']
    print(f"  训练集: {train_dataset.n_samples} 样本")
    print(f"  验证集: {val_dataset.n_samples} 样本")
    print(f"  特征数: {loader.n_features}")

    # 2. 创建模型
    print("\n[2] 创建模型...")
    model = AECS(
        n_features=loader.n_features,
        latent_dim=args.latent_dim,
        hidden_units=args.hidden_units,
        k_spatial=5,
        k_temporal=5,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg
    )
    print(f"  latent_dim={args.latent_dim}, dropout={args.dropout_rate}, "
          f"l2_reg={args.l2_reg}")

    # 3. 保存配置
    config = {
        'data_path': str(data_path),
        'n_features': loader.n_features,
        'window_size': loader.window_size,
        'batch_size': args.batch_size,
        'latent_dim': args.latent_dim,
        'hidden_units': args.hidden_units,
        'k_spatial': 5,
        'k_temporal': 5,
        'dropout_rate': args.dropout_rate,
        'l2_reg': args.l2_reg,
        'learning_rate': args.learning_rate,
        'p_drop': args.p_drop,
        'n_corrupted': args.n_corrupted,
        'p_consist': args.p_consist,
        'lambda1': args.lambda1,
        'lambda2': args.lambda2,
        'lambda3': args.lambda3,
        'stride': args.stride,
        'missing_rate': 0.2,
        'missing_type': 'MCAR',
        'seed': args.seed
    }

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # 4. 训练
    print("\n[3] 初始化训练器...")
    trainer = AECSTrainerV2(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        p_drop=args.p_drop,
        n_corrupted=args.n_corrupted,
        p_consist=args.p_consist,
        use_lr_scheduler=args.use_lr_scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr
    )

    trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
