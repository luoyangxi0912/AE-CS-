"""
循环训练脚本 - 在训练中使用迭代填充策略

核心改进：
1. 训练时每个batch进行多轮迭代填充
2. 损失基于最终迭代结果计算
3. 强制CPU训练

用法:
    python train_iterative.py --epochs 50 --max_iters 3 --checkpoint_dir checkpoints_iter
"""

# ===== 强制使用 CPU =====
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
print(f"[INFO] 强制使用 CPU 训练")

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

from models.ae_cs import AECS
from data import AECSDataLoader, AECSDataset
from models.losses import reconstruction_loss, consistency_loss
# neighborhood.py中的函数需要被导入
from models.neighborhood import NeighborhoodModule

# 计算自适应一致性损失的权重函数
def compute_consistency_weights(mask_orig, mask_corrupted_list, sigma_c=1.0):
    weights = []
    for mask_corrupted in mask_corrupted_list:
        diff = tf.abs(mask_corrupted - mask_orig)
        l1_distance = tf.reduce_sum(diff)
        weight = tf.exp(-l1_distance / (sigma_c ** 2))
        weights.append(float(weight.numpy()))
    return weights

class IterativeTrainer:
    """循环训练器"""

    def __init__(self, model, train_dataset, val_dataset,
                 checkpoint_dir='checkpoints_iter',
                 learning_rate=0.001,
                 lambda1=1.0, lambda2=0.1, lambda3=0.1,
                 train_iters=3, momentum=0.1,
                 n_features=None):
        """
        Args:
            model: AECS模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            checkpoint_dir: 检查点目录
            learning_rate: 学习率
            lambda1: 一致性损失权重
            lambda2: 空间损失权重
            lambda3: 时间损失权重
            train_iters: 训练时的迭代次数
            momentum: 迭代动量
            n_features: 特征数量
        """
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
        self.train_iters = train_iters
        self.momentum = momentum
        self.n_features = n_features

        # 训练历史
        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0

    def iterative_forward(self, X_observed, mask, training=True):
        """
        迭代前向传播

        Args:
            X_observed: 原始观测数据 [batch, time, features]
            mask: 掩码 [batch, time, features]
            training: 是否训练模式

        Returns:
            X_final: 最终迭代结果
            z_orig: 原始编码器输出（用于一致性损失）
            z_list: 所有编码器输出列表
        """
        # 初始化：缺失位置为0
        X_current = X_observed * mask
        X_prev = X_current

        z_orig_final = None
        
        # 记录三路编码的输出，用于Consistency Loss
        # 这里我们只取最后一次迭代的z_orig作为"主要"表征
        # 但为了计算一致性损失，我们需要多个视角的Z
        
        z_orig, z_space, z_time = None, None, None

        for i in range(self.train_iters):
            # 获取的三种输入准备方式已经在 models.ae_cs.AECS._prepare_inputs 中实现
            # 但这里我们需要"循环"更新，所以不能直接调那个方法，除非我们hack一下
            # 或者我们手动实现循环逻辑
            
            # 手动实现KNN更新逻辑（类似于iterative_imputation.py）
            x_space_init = self._compute_spatial_knn(X_current, X_observed, mask)
            x_time_init = self._compute_temporal_knn(X_current, X_observed, mask)

            # 三编码器前向传播
            x_zero = X_current * mask
            z_orig = self.model.encoder_orig(x_zero, mask, training=training)
            z_space = self.model.encoder_space(x_space_init, mask, training=training)
            z_time = self.model.encoder_time(x_time_init, mask, training=training)

            # 计算缺失率并融合
            missing_rate = 1.0 - tf.reduce_mean(mask, axis=[1, 2])
            alpha, z_fused = self.model.gating_network(
                z_orig, z_space, z_time, missing_rate, training=training
            )

            # 解码
            X_pred = self.model.decoder(z_fused, training=training)

            # 动量更新
            if self.momentum > 0 and i > 0:
                X_imputed = (1 - self.momentum) * X_pred + self.momentum * X_prev
            else:
                X_imputed = X_pred

            # 更新缺失位置，保持观测位置不变
            X_new = mask * X_observed + (1.0 - mask) * X_imputed

            X_prev = X_current
            X_current = X_new
            
            # 记录本轮结果
            z_orig_final = z_orig 
            # 记录三个视角的Z用于计算一致性
            # 这里我们把空间和时间视角的Z作为"增强版本"来计算一致性损失
            # 这是一个策略选择：强迫不同视角的表征一致
            z_space_final = z_space
            z_time_final = z_time

        # 将空间和时间编码作为"损坏版本"或"不同视角"来计算一致性
        z_list = [z_space_final, z_time_final]
        
        return X_current, z_orig_final, z_list

    def _compute_spatial_knn(self, X_filled, X_observed, mask, k=5):
        """基于已填充数据计算空间KNN"""
        batch_size = tf.shape(X_filled)[0].numpy()
        time_steps = tf.shape(X_filled)[1].numpy()
        n_features = tf.shape(X_filled)[2].numpy()

        X_filled_np = X_filled.numpy()
        X_obs_np = X_observed.numpy()
        M = mask.numpy()

        X_init = X_obs_np * M

        for b in range(batch_size):
            X_b = X_filled_np[b]
            M_b = M[b]

            for t in range(time_steps):
                missing_vars = np.where(M_b[t, :] == 0)[0]
                if len(missing_vars) == 0:
                    continue

                distances = np.full(time_steps, np.inf)
                for s in range(time_steps):
                    if s == t:
                        continue
                    diff = X_b[t, :] - X_b[s, :]
                    distances[s] = np.sqrt(np.mean(diff ** 2))

                k_actual = min(k, int(np.sum(distances < np.inf)))
                if k_actual == 0:
                    continue

                k_indices = np.argsort(distances)[:k_actual]
                k_distances = distances[k_indices]

                sigma = np.median(k_distances) + 1e-8
                weights = np.exp(-k_distances ** 2 / (sigma ** 2))
                weights = weights / (np.sum(weights) + 1e-8)

                for v in missing_vars:
                    X_init[b, t, v] = np.sum(weights * X_b[k_indices, v])

        return tf.constant(X_init, dtype=tf.float32)

    def _compute_temporal_knn(self, X_filled, X_observed, mask, k=5):
        """基于已填充数据计算时间KNN"""
        batch_size = tf.shape(X_filled)[0].numpy()
        time_steps = tf.shape(X_filled)[1].numpy()
        n_features = tf.shape(X_filled)[2].numpy()

        X_filled_np = X_filled.numpy()
        X_obs_np = X_observed.numpy()
        M = mask.numpy()

        X_init = X_obs_np * M

        for b in range(batch_size):
            X_b = X_filled_np[b]
            M_b = M[b]

            var_distances = np.zeros((n_features, n_features))
            for j in range(n_features):
                for m in range(n_features):
                    if j == m:
                        var_distances[j, m] = np.inf
                    else:
                        diff = X_b[:, j] - X_b[:, m]
                        var_distances[j, m] = np.sqrt(np.mean(diff ** 2))

            for v in range(n_features):
                k_actual = min(k, n_features - 1)
                neighbor_vars = np.argsort(var_distances[v, :])[:k_actual]
                neighbor_dists = var_distances[v, neighbor_vars]

                sigma = np.median(neighbor_dists) + 1e-8
                var_weights = np.exp(-neighbor_dists ** 2 / (sigma ** 2))
                var_weights = var_weights / (np.sum(var_weights) + 1e-8)

                missing_times = np.where(M_b[:, v] == 0)[0]
                for t in missing_times:
                    X_init[b, t, v] = np.sum(var_weights * X_b[t, neighbor_vars])

        return tf.constant(X_init, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def _compute_loss(self, X_true, X_pred, mask, z_orig, z_list):
        """计算总损失"""
        # 重建损失 (MSE)
        L_recon = reconstruction_loss(X_true, X_pred, mask)

        # 一致性损失 (Cross-View Consistency)
        # z_list包含了z_space和z_time，我们希望它们和z_orig一致
        # 这里权重视为1.0
        weights = [1.0] * len(z_list)
        L_consist = consistency_loss(z_orig, z_list, weights)

        # 邻域损失 (暂设为0，因计算开销大且在迭代中已隐式包含)
        L_space = tf.constant(0.0, dtype=tf.float32)
        L_time = tf.constant(0.0, dtype=tf.float32)

        # 总损失
        total = L_recon + self.lambda1 * L_consist + self.lambda2 * L_space + self.lambda3 * L_time

        return total, L_recon, L_consist, L_space, L_time

    def train_step(self, X_batch, mask_batch):
        """单步训练
        注意：X_batch即为原始数据(X_original)，mask_batch定义了观测位置
        """
        X_original = X_batch
        
        with tf.GradientTape() as tape:
            # 迭代前向传播 (X_pred是填充后的完整数据)
            X_pred, z_orig, z_list = self.iterative_forward(
                X_batch, mask_batch, training=True
            )

            # 计算损失
            total_loss, L_recon, L_consist, L_space, L_time = self._compute_loss(
                X_original, X_pred, mask_batch, z_orig, z_list
            )

        # 梯度更新
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return {
            'total': total_loss.numpy(),
            'recon': L_recon.numpy(),
            'consist': L_consist.numpy(),
            'space': L_space.numpy(),
            'time': L_time.numpy()
        }

    def validate(self):
        """验证"""
        val_losses = []

        for batch in self.val_dataset.get_dataset():
            X_batch, mask_batch = batch
            X_original = X_batch

            X_pred, z_orig, z_list = self.iterative_forward(
                X_batch, mask_batch, training=False
            )

            total_loss, L_recon, L_consist, L_space, L_time = self._compute_loss(
                X_original, X_pred, mask_batch, z_orig, z_list
            )

            val_losses.append({
                'total': total_loss.numpy(),
                'recon': L_recon.numpy(),
                'consist': L_consist.numpy(),
                'space': L_space.numpy(),
                'time': L_time.numpy()
            })

        # 计算平均
        avg_loss = {k: np.mean([l[k] for l in val_losses]) for k in val_losses[0].keys()}
        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        # 保存训练状态
        state = {
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'lr_history': self.lr_history
        }
        state_path = self.checkpoint_dir / 'training_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4)

        # 保存最佳模型
        if is_best:
            self.model.save_weights(str(self.checkpoint_dir / 'best_model.weights.h5'))
            print(f"  [保存] 最佳模型 (val_loss={self.best_val_loss:.4f})")

        # 每10轮保存一次
        if epoch % 10 == 0:
            self.model.save_weights(str(self.checkpoint_dir / f'model_epoch_{epoch}.weights.h5'))

    def resume_from_checkpoint(self):
        """从检查点恢复"""
        state_path = self.checkpoint_dir / 'training_state.json'
        best_model_path = self.checkpoint_dir / 'best_model.weights.h5'

        if not state_path.exists() or not best_model_path.exists():
            print("[INFO] 未找到检查点，从头开始训练")
            return 0

        with open(state_path, 'r') as f:
            state = json.load(f)

        self.current_epoch = state['epoch']
        self.best_val_loss = state['best_val_loss']
        self.train_loss_history = state['train_loss_history']
        self.val_loss_history = state['val_loss_history']
        self.lr_history = state.get('lr_history', [])

        # 恢复学习率
        if self.lr_history:
            self.current_lr = self.lr_history[-1]
            self.optimizer.learning_rate.assign(self.current_lr)

        # 先做一次前向传播创建模型变量
        for batch in self.train_dataset.get_dataset().take(1):
            X_batch, mask_batch = batch
            _ = self.model(X_batch, mask_batch, training=False)

        self.model.load_weights(str(best_model_path))
        print(f"[OK] 已从 Epoch {self.current_epoch} 恢复训练")
        return self.current_epoch

    def train(self, epochs=50, early_stopping_patience=15, resume=False):
        """训练主循环"""
        start_epoch = 0
        patience_counter = 0
        best_epoch = 0

        if resume:
            start_epoch = self.resume_from_checkpoint()
            best_epoch = start_epoch
            print(f"  从 Epoch {start_epoch + 1} 继续训练到 Epoch {epochs}")

        print(f"\n开始循环训练 (train_iters={self.train_iters}, momentum={self.momentum})")
        print("="*70)

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch + 1
            epoch_losses = []

            # 训练
            train_ds = self.train_dataset.get_dataset()
            for batch_idx, batch in enumerate(train_ds):
                # AECSDataset 返回 (X, Mask)
                X_batch, mask_batch = batch
                
                loss_dict = self.train_step(X_batch, mask_batch)
                epoch_losses.append(loss_dict)

                if (batch_idx + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1} Batch {batch_idx+1}: total={loss_dict['total']:.4f}, recon={loss_dict['recon']:.4f}")

            # 计算训练平均损失
            train_loss = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()}
            self.train_loss_history.append(train_loss)

            # 验证
            print("  正在验证...")
            val_loss = self.validate()
            self.val_loss_history.append(val_loss)
            self.lr_history.append(float(self.optimizer.learning_rate.numpy()))

            # 打印
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train: total={train_loss['total']:.4f}, recon={train_loss['recon']:.4f}")
            print(f"  Val:   total={val_loss['total']:.4f}, recon={val_loss['recon']:.4f}")

            # 检查最佳模型
            is_best = val_loss['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss['total']
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            # 保存检查点
            self.save_checkpoint(epoch + 1, is_best)

            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发！最佳模型在 Epoch {best_epoch}")
                break

        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.4f} (Epoch {best_epoch})")
        return self.best_val_loss


def main():
    parser = argparse.ArgumentParser(description='循环训练')
    parser.add_argument('--data_path', type=str, default='hangmei_90_拼接好的.csv')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_iter')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--train_iters', type=int, default=3, help='训练时迭代次数')
    parser.add_argument('--momentum', type=float, default=0.1, help='迭代动量')
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=0.1)
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')

    args = parser.parse_args()

    print("="*70)
    print("循环训练 (Iterative Training)")
    print("="*70)
    print(f"  迭代次数: {args.train_iters}")
    print(f"  动量: {args.momentum}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print("="*70)

    # 1. 初始化数据加载器
    print("[INFO] 正在初始化数据加载器...")
    loader = AECSDataLoader(
        batch_size=args.batch_size,
        shuffle_train=True,
        seed=42
    )
    
    # 处理数据路径
    data_path = args.data_path
    if not os.path.exists(data_path):
        if os.path.exists(os.path.basename(data_path)):
            data_path = os.path.basename(data_path)
            
    loader.preprocessor.data_path = Path(data_path)
    
    # 2. 准备数据
    print("[INFO] 正在准备数据...")
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

    # 3. 创建模型
    model = AECS(
        n_features=loader.n_features,
        latent_dim=args.latent_dim,
        hidden_units=args.hidden_units,
        k_spatial=5,
        k_temporal=5,
        dropout_rate=0.2,
        l2_reg=0.001
    )

    # 保存配置
    config = vars(args)
    config['n_features'] = loader.n_features
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 4. 创建训练器
    trainer = IterativeTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        train_iters=args.train_iters,
        momentum=args.momentum,
        n_features=loader.n_features
    )

    # 5. 开始训练
    trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
