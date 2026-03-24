#!/bin/bash
# ============================================================
# AE-CS V9 云服务器一键部署 + 训练脚本
#
# 用法:
#   1. 将整个项目上传到云服务器 (推荐用 scp 或 git clone)
#   2. chmod +x run_cloud.sh
#   3. ./run_cloud.sh
#
# 推荐环境: Ubuntu 20.04+, CUDA 11.x, Python 3.10
# 推荐GPU: T4(16GB) / RTX 3090(24GB) / A5000(24GB)
# ============================================================

set -e

echo "============================================================"
echo "AE-CS V9 云服务器部署"
echo "============================================================"

# === 1. 安装依赖 ===
echo ""
echo "[1/4] 安装 Python 依赖..."
pip install tensorflow==2.10.0 numpy pandas scikit-learn -q
pip install faiss-cpu -q  # KNN 加速 (CPU版即可, KNN init 在 CPU 上运行)
echo "  ✅ 依赖安装完成"

# === 2. 检测 GPU ===
echo ""
echo "[2/4] 检测 GPU..."
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  ✅ 检测到 {len(gpus)} 个 GPU:')
    for gpu in gpus:
        print(f'     {gpu}')
    # 允许显存按需增长，防止OOM
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('  ⚠️ 未检测到 GPU，将使用 CPU 训练')
"

# === 3. 验证修复 ===
echo ""
echo "[3/4] 运行 V9 验证..."
python3 verify_v9.py
if [ $? -ne 0 ]; then
    echo "❌ 验证失败，请检查代码"
    exit 1
fi

# === 4. 正式训练 ===
echo ""
echo "[4/4] 开始正式训练..."
echo "  GPU 加速模式, batch_size=32, epochs=100"
echo ""

python3 train_cloud.py \
    --checkpoint_dir checkpoints_v9 \
    --epochs 100 \
    --batch_size 32 \
    --early_stopping_patience 20

echo ""
echo "============================================================"
echo "训练完成! 运行评估:"
echo "  python3 evaluate.py --checkpoint_dir checkpoints_v9"
echo "============================================================"
