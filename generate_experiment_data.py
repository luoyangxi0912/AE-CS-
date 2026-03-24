"""
实验数据生成脚本 - 确保所有方法共用同一数据切分和缺失 mask

论文公平性要求：
  所有方法（AE-CS + 全部基线）必须共用同一数据切分和同一缺失 mask。
  本脚本为每组实验配置生成并持久化 .npz 文件，供所有方法加载。

用法：
    python generate_experiment_data.py                     # 生成全部 5 组
    python generate_experiment_data.py --config C2         # 只生成 C2
    python generate_experiment_data.py --output_dir data/experiments

生成文件结构：
    data/experiments/
    ├── C1_MCAR_0.1.npz
    ├── C2_MCAR_0.2.npz
    ├── C3_MCAR_0.3.npz
    ├── C4_MCAR_0.4.npz
    └── C5_MAR_0.2.npz

每个 .npz 包含：
    train_X, train_mask       - 训练集 [n_train, window_size, n_features]
    val_X, val_mask           - 验证集
    test_X, test_mask         - 测试集
    split_info                - 切分索引信息 (dict)
    config                    - 实验配置 (dict)
"""

import os
import numpy as np
import argparse
import json
from pathlib import Path

from data.preprocessor import HangmeiPreprocessor


# 5 组实验配置
EXPERIMENT_CONFIGS = {
    'C1': {'missing_rate': 0.1, 'missing_type': 'MCAR'},
    'C2': {'missing_rate': 0.2, 'missing_type': 'MCAR'},
    'C3': {'missing_rate': 0.3, 'missing_type': 'MCAR'},
    'C4': {'missing_rate': 0.4, 'missing_type': 'MCAR'},
    'C5': {'missing_rate': 0.2, 'missing_type': 'MAR'},
}

# 固定全局参数
GLOBAL_SEED = 42
WINDOW_SIZE = 48
STRIDE = 12
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def generate_one_config(config_name, config, data_path, output_dir):
    """为单个实验配置生成数据并保存"""
    print(f"\n{'='*70}")
    print(f"生成配置: {config_name} "
          f"(missing_rate={config['missing_rate']}, "
          f"missing_type={config['missing_type']})")
    print(f"{'='*70}")

    preprocessor = HangmeiPreprocessor(
        data_path=data_path,
        scaler_type='standard',
        window_size=WINDOW_SIZE,
        stride=STRIDE
    )

    # 使用固定全局种子 + 配置名哈希，确保每组配置有不同但可复现的 mask
    # 同一配置的 mask 永远相同，不同配置的 mask 互不干扰
    config_seed = GLOBAL_SEED + hash(config_name) % 10000

    splits = preprocessor.prepare_data(
        missing_rate=config['missing_rate'],
        missing_type=config['missing_type'],
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=config_seed
    )

    train_X, train_mask = splits['train']
    val_X, val_mask = splits['val']
    test_X, test_mask = splits['test']

    # 计算切分索引信息（便于复现和调试）
    T = preprocessor.scaler.n_features_in_  # 不可靠，用 shape 代替
    df = preprocessor.load_data()
    T_total = len(df)
    gap = WINDOW_SIZE
    usable = T_total - 2 * gap
    train_len = int(usable * TRAIN_RATIO)
    val_len = int(usable * VAL_RATIO)

    split_info = {
        'T_total': T_total,
        'n_features': preprocessor.n_features,
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'gap': gap,
        'train_time_range': [0, train_len],
        'val_time_range': [train_len + gap, train_len + gap + val_len],
        'test_time_range': [train_len + gap + val_len + gap, T_total],
        'train_n_windows': len(train_X),
        'val_n_windows': len(val_X),
        'test_n_windows': len(test_X),
    }

    experiment_config = {
        'config_name': config_name,
        'missing_rate': config['missing_rate'],
        'missing_type': config['missing_type'],
        'global_seed': GLOBAL_SEED,
        'config_seed': config_seed,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
    }

    # 保存
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{config_name}_{config['missing_type']}_{config['missing_rate']}.npz"
    output_path = output_dir / filename

    np.savez_compressed(
        output_path,
        train_X=train_X,
        train_mask=train_mask,
        val_X=val_X,
        val_mask=val_mask,
        test_X=test_X,
        test_mask=test_mask,
        split_info=json.dumps(split_info),
        config=json.dumps(experiment_config),
    )

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n保存: {output_path} ({file_size:.1f} MB)")
    print(f"  train: X={train_X.shape}, mask={train_mask.shape}, "
          f"缺失率={1 - train_mask.mean():.2%}")
    print(f"  val:   X={val_X.shape}, mask={val_mask.shape}, "
          f"缺失率={1 - val_mask.mean():.2%}")
    print(f"  test:  X={test_X.shape}, mask={test_mask.shape}, "
          f"缺失率={1 - test_mask.mean():.2%}")

    return output_path


def verify_npz(path):
    """验证已保存的 .npz 文件完整性"""
    data = np.load(path, allow_pickle=True)
    config = json.loads(str(data['config']))
    split_info = json.loads(str(data['split_info']))

    print(f"\n验证: {path}")
    print(f"  配置: {config['config_name']} "
          f"({config['missing_type']}, rate={config['missing_rate']})")
    print(f"  seed: global={config['global_seed']}, "
          f"config={config['config_seed']}")

    for split in ['train', 'val', 'test']:
        X = data[f'{split}_X']
        M = data[f'{split}_mask']
        assert X.shape == M.shape, f"{split} shape mismatch: {X.shape} vs {M.shape}"
        print(f"  {split}: {X.shape}, 缺失率={1 - M.mean():.2%}")

    print("  验证通过")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='生成实验数据（固定切分 + 固定缺失 mask）'
    )
    parser.add_argument('--data_path', type=str,
                        default='hangmei_90_拼接好的.csv')
    parser.add_argument('--output_dir', type=str,
                        default='data/experiments')
    parser.add_argument('--config', type=str, default=None,
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help='只生成指定配置（默认全部）')
    parser.add_argument('--verify', action='store_true',
                        help='仅验证已有 .npz 文件')

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        if os.path.exists(os.path.basename(args.data_path)):
            args.data_path = os.path.basename(args.data_path)

    if args.config:
        configs = {args.config: EXPERIMENT_CONFIGS[args.config]}
    else:
        configs = EXPERIMENT_CONFIGS

    if args.verify:
        output_dir = Path(args.output_dir)
        for name, cfg in configs.items():
            filename = f"{name}_{cfg['missing_type']}_{cfg['missing_rate']}.npz"
            path = output_dir / filename
            if path.exists():
                verify_npz(path)
            else:
                print(f"[SKIP] {path} 不存在")
        return

    print("=" * 70)
    print("AE-CS 实验数据生成")
    print(f"  全局种子: {GLOBAL_SEED}")
    print(f"  窗口: {WINDOW_SIZE}, 步长: {STRIDE}")
    print(f"  切分: train={TRAIN_RATIO}, val={VAL_RATIO}, "
          f"test={1 - TRAIN_RATIO - VAL_RATIO:.2f}")
    print(f"  配置: {list(configs.keys())}")
    print("=" * 70)

    generated = []
    for name, cfg in configs.items():
        path = generate_one_config(name, cfg, args.data_path, args.output_dir)
        generated.append(path)

    print(f"\n{'='*70}")
    print(f"全部完成! 共生成 {len(generated)} 个 .npz 文件")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*70}")

    # 自动验证
    print("\n自动验证...")
    for path in generated:
        verify_npz(path)


if __name__ == '__main__':
    main()
