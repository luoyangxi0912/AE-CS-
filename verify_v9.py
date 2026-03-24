"""
V9 最小验证 — 5个epoch快速检查所有修复是否生效
用法: python verify_v9.py
"""
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Python 3.8 兼容
import tensorflow as tf
import numpy as np
from pathlib import Path

from models.ae_cs import AECS
from data import AECSDataLoader
from models.losses import total_loss, generate_augmented_masks

tf.random.set_seed(42)
np.random.seed(42)

# === 1. 加载数据 (少量) ===
print("=" * 60)
print("V9 最小验证")
print("=" * 60)

loader = AECSDataLoader(batch_size=4, shuffle_train=True, seed=42)
loader.preprocessor.stride = 24  # 快速验证用大stride
loader.preprocessor.data_path = Path('hangmei_90_拼接好的.csv')
datasets = loader.prepare(missing_rate=0.2, missing_type='MCAR', train_ratio=0.7, val_ratio=0.15)
train_ds = datasets['train']
print(f"训练样本: {train_ds.n_samples}, 特征: {loader.n_features}")

# === 2. 创建模型 ===
model = AECS(n_features=loader.n_features, latent_dim=32, hidden_units=128,
             dropout_rate=0.3, l2_reg=0.0005)
optimizer = tf.keras.optimizers.Adam(0.001)

# === 3. 取一个batch做检查 ===
for X_batch, mask_batch in train_ds.get_dataset().take(1):
    pass

print(f"\nBatch shape: X={X_batch.shape}, M={mask_batch.shape}")

# === 检查1: stop_gradient ===
print("\n--- 检查1: stop_gradient ---")
from models.losses import consistency_loss
z_test = tf.Variable(tf.random.normal([2, 4, 32]))
z_corr = [tf.Variable(tf.random.normal([2, 4, 32]))]
with tf.GradientTape() as tape:
    loss = consistency_loss(z_test, z_corr, [1.0])
grad_z_orig = tape.gradient(loss, z_test)
if grad_z_orig is None:
    print("  ✅ stop_gradient 生效 — z_orig 无梯度")
else:
    print("  ❌ stop_gradient 未生效! z_orig 收到梯度!")

# === 检查2: Feature 11 delta 掩码 ===
print("\n--- 检查2: Feature 11 delta 掩码 ---")
outputs = model(X_batch, mask_batch, training=False, return_all=True)
# 检查 feat_delta_mask
mask_val = model.feat_delta_mask.numpy()
if mask_val[0, 0, 11] == 0.0 and np.sum(mask_val) == loader.n_features - 1:
    print("  ✅ Feature 11 掩码正确 (feat[11]=0, 其余=1)")
else:
    print(f"  ❌ Feature 11 掩码异常: {mask_val[0,0,:]}")

# === 检查3: recon_mask (V9核心) ===
print("\n--- 检查3: V9 recon_mask ---")
p_drop = 0.5
bernoulli = tf.cast(tf.random.uniform(tf.shape(mask_batch)) > p_drop, tf.float32)
corrupted_mask = mask_batch * bernoulli
recon_mask = mask_batch * (1.0 - corrupted_mask)

n_mask = float(tf.reduce_sum(mask_batch).numpy())
n_corrupted = float(tf.reduce_sum(corrupted_mask).numpy())
n_recon = float(tf.reduce_sum(recon_mask).numpy())
print(f"  原始mask位置: {n_mask:.0f}")
print(f"  corrupted后:  {n_corrupted:.0f}")
print(f"  recon_mask:   {n_recon:.0f} ({n_recon/n_mask*100:.0f}% 有用信号)")
if n_recon > 0 and n_recon < n_mask:
    print("  ✅ recon_mask 正确 — 仅额外掩盖位置")
else:
    print("  ❌ recon_mask 异常!")

# === 检查4: 输入噪声 ===
print("\n--- 检查4: 输入噪声强度 ---")
noise_std = model.encoder_orig.input_noise.stddev
print(f"  GaussianNoise std = {noise_std}")
if noise_std >= 0.05:
    print("  ✅ 噪声足够大 (≥0.05)")
else:
    print(f"  ⚠️ 噪声可能太小 ({noise_std})")

# === 检查5: delta clip 范围 ===
print("\n--- 检查5: Delta clip 范围 ---")
z_fused_big = tf.ones([1, 48, 32]) * 10.0  # 极端输入
delta_out = model.decoder(z_fused_big, training=False)
delta_max = float(tf.reduce_max(delta_out).numpy())
delta_min = float(tf.reduce_min(delta_out).numpy())
print(f"  Decoder输出范围: [{delta_min:.2f}, {delta_max:.2f}]")
if delta_max <= 2.01 and delta_min >= -2.01:
    print("  ✅ Delta clip [-2.0, 2.0] 生效")
else:
    print(f"  ❌ Delta clip 异常!")

# === 检查6: 5轮训练 — 监控 delta 和 loss ===
print("\n--- 检查6: 5轮迷你训练 ---")
print(f"{'Epoch':>5} {'L_total':>9} {'L_recon':>9} {'L_consist':>9} {'L_space':>9} {'delta_std':>10}")

for epoch in range(5):
    epoch_losses = []
    delta_stds = []

    for X_batch, mask_batch in train_ds.get_dataset():
        with tf.GradientTape() as tape:
            bernoulli = tf.cast(
                tf.random.uniform(tf.shape(mask_batch)) > p_drop, tf.float32
            )
            corrupted_mask = mask_batch * bernoulli

            outputs = model(X_batch, corrupted_mask, training=True, return_all=True)
            x_hat = outputs['x_hat']
            z_orig = outputs['z_orig']
            neighborhood_info = outputs['neighborhood_info']

            consistency_masks, consistency_weights = generate_augmented_masks(
                mask_batch, Q=3, p_drop=0.1
            )
            z_corrupted_list = []
            for cm in consistency_masks:
                z_corrupted_list.append(model.encode(X_batch, cm, training=True))

            recon_mask = mask_batch * (1.0 - corrupted_mask)

            loss, losses_dict = total_loss(
                x_true=X_batch, x_pred=x_hat, mask=mask_batch,
                z_orig=z_orig, z_corrupted_list=z_corrupted_list,
                neighborhood_info=neighborhood_info,
                lambda1=1.0, lambda2=1.0, lambda3=10.0,
                corruption_weights=consistency_weights,
                recon_mask=recon_mask,
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_losses.append({k: float(v.numpy()) for k, v in losses_dict.items()})

        # 计算 delta 的 std
        x_space_init = outputs['x_space_init']
        delta = x_hat - x_space_init
        delta_stds.append(float(tf.math.reduce_std(delta).numpy()))

    avg = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
    avg_delta_std = np.mean(delta_stds)

    print(f"{epoch+1:>5} {avg['total']:>9.4f} {avg['recon']:>9.4f} "
          f"{avg['consist']:>9.4f} {avg['space']:>9.4f} {avg_delta_std:>10.6f}")

# === 最终检查: L_consist 是否在下降但没有坍缩到 0 ===
print("\n--- 最终诊断 ---")
final_consist = avg['consist']
final_delta = avg_delta_std

checks_passed = 0
checks_total = 3

if final_consist > 1e-4:
    print(f"  ✅ L_consist={final_consist:.6f} > 1e-4 (没有坍缩到0)")
    checks_passed += 1
else:
    print(f"  ❌ L_consist={final_consist:.6f} 接近0 — 编码器可能在偷懒!")

if final_delta > 0.01:
    print(f"  ✅ delta_std={final_delta:.6f} > 0.01 (在学习非零修正)")
    checks_passed += 1
else:
    print(f"  ❌ delta_std={final_delta:.6f} ≈ 0 — delta 没有学到东西!")

if avg['recon'] < 2.0:
    print(f"  ✅ L_recon={avg['recon']:.4f} 在合理范围")
    checks_passed += 1
else:
    print(f"  ⚠️ L_recon={avg['recon']:.4f} 偏高")

print(f"\n{'='*60}")
print(f"验证结果: {checks_passed}/{checks_total} 通过")
if checks_passed == checks_total:
    print("🟢 所有检查通过，可以开始正式训练:")
    print("   python train_iterative.py --checkpoint_dir checkpoints_v9 --epochs 100")
else:
    print("🔴 存在问题，请检查上方标记为 ❌ 的项目")
print(f"{'='*60}")
