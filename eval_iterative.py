"""
迭代填充评估脚本 - 对比 Single-Pass vs Iterative Imputation

用法:
    python eval_iterative.py --checkpoint_dir checkpoints_v3 --max_iters 5
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 使用CPU

import tensorflow as tf
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 导入模型和数据
from models.ae_cs import AECS
from data import AECSDataLoader
from models.iterative_imputation import IterativeImputer, adaptive_iterative_impute


def load_model_and_data(checkpoint_dir: str):
    """加载模型和测试数据"""
    checkpoint_path = Path(checkpoint_dir)

    # 加载配置
    config_path = checkpoint_path / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"[INFO] 加载配置: {config_path}")

    # 创建数据加载器
    data_path = config.get('data_path', 'hangmei_90_拼接好的.csv')
    # 处理可能的路径问题
    if not os.path.exists(data_path):
        # 尝试在当前目录下找
        if os.path.exists(os.path.basename(data_path)):
            data_path = os.path.basename(data_path)
        else:
            print(f"[WARN] 数据文件 {data_path} 未找到，尝试使用默认为 hangmei_90_拼接好的.csv")
            data_path = 'hangmei_90_拼接好的.csv'
            
    loader = AECSDataLoader(
        batch_size=16,  # 评估时可以用较大batch
        shuffle_train=False,
        seed=42
    )
    loader.preprocessor.data_path = Path(data_path)
    if 'window_size' in config:
        loader.preprocessor.window_size = config['window_size']

    # 准备数据
    print("[INFO] 准备数据...")
    datasets = loader.prepare(
        missing_rate=config.get('missing_rate', 0.2),
        missing_type=config.get('missing_type', 'MCAR'),
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # 获取测试集 (AECSDataset对象)
    dataset = datasets['test']
    print(f"[INFO] 测试集样本数: {len(dataset)}")

    # 创建模型
    model = AECS(
        n_features=loader.n_features,
        latent_dim=config.get('latent_dim', 64),
        hidden_units=config.get('hidden_units', 128),
        k_spatial=config.get('k_spatial', 5),
        k_temporal=config.get('k_temporal', 5),
        dropout_rate=config.get('dropout_rate', 0.2),
        l2_reg=config.get('l2_reg', 0.001)
    )

    # 构建模型（需要先前向传播一次）
    test_ds = dataset.get_dataset()
    for X_batch, mask_batch in test_ds.take(1):
        _ = model(X_batch, mask_batch, training=False)
        break

    # 加载权重
    weights_path = checkpoint_path / 'best_model.weights.h5'
    if not weights_path.exists():
        # 尝试找其他权重文件
        weights_files = list(checkpoint_path.glob('*.weights.h5'))
        if weights_files:
            weights_path = weights_files[0]
            
    print(f"[INFO] 加载模型权重: {weights_path}")
    model.load_weights(str(weights_path))

    return model, dataset, config


def evaluate_single_pass(model, dataset):
    """Single-Pass 评估（当前方法）"""
    print("\n" + "="*60)
    print("Single-Pass 评估")
    print("="*60)

    all_y_true = []
    all_y_pred = []
    all_masks = []

    test_ds = dataset.get_dataset()

    for batch in test_ds:
        X_batch, mask_batch = batch
        X_original = X_batch # 原始数据

        # 单次前向传播
        X_pred = model(X_batch, mask_batch, training=False)

        all_y_true.append(X_original.numpy())
        all_y_pred.append(X_pred.numpy())
        all_masks.append(mask_batch.numpy())

    # 合并所有batch
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # 计算缺失位置的指标
    missing_mask = (masks == 0)
    
    # 避免空缺失
    if np.sum(missing_mask) == 0:
        print("[WARN] 测试集中没有缺失值 (mask全为1)")
        r2, rmse, mae = 1.0, 0.0, 0.0
    else:
        y_true_missing = y_true[missing_mask]
        y_pred_missing = y_pred[missing_mask]

        r2 = r2_score(y_true_missing, y_pred_missing)
        rmse = np.sqrt(mean_squared_error(y_true_missing, y_pred_missing))
        mae = mean_absolute_error(y_true_missing, y_pred_missing)

    print(f"  缺失位置 R²:   {r2:.4f}")
    print(f"  缺失位置 RMSE: {rmse:.4f}")
    print(f"  缺失位置 MAE:  {mae:.4f}")

    return {
        'r2': r2, 'rmse': rmse, 'mae': mae,
        'y_true': y_true, 'y_pred': y_pred, 'masks': masks
    }


def evaluate_iterative(model, dataset, max_iters=5, recompute_knn=True, momentum=0.1):
    """迭代填充评估"""
    print("\n" + "="*60)
    print(f"迭代填充评估 (max_iters={max_iters}, recompute_knn={recompute_knn})")
    print("="*60)

    imputer = IterativeImputer(
        model=model,
        max_iters=max_iters,
        tol=1e-4,
        momentum=momentum,
        recompute_knn=recompute_knn,
        verbose=False # 减少输出
    )

    all_y_true = []
    all_y_pred = []
    all_masks = []
    all_histories = []

    test_ds = dataset.get_dataset()
    batch_idx = 0
    total_batches = len(dataset) // 16 # Approx

    for batch in test_ds:
        X_batch, mask_batch = batch
        X_original = X_batch
        batch_idx += 1
        print(f"\r处理 Batch {batch_idx}...", end="")

        # 迭代填充
        X_imputed, history = imputer.impute(
            X=X_batch,  # 此处传X（包含缺失位置的真实值，但在impute通过mask过滤掉）
            mask=mask_batch,
            X_true=X_original  # 用于计算RMSE
        )

        all_y_true.append(X_original.numpy())
        all_y_pred.append(X_imputed.numpy())
        all_masks.append(mask_batch.numpy())
        all_histories.append(history)
    
    print("\n完成")

    # 合并所有batch
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # 计算缺失位置的指标
    missing_mask = (masks == 0)
    if np.sum(missing_mask) == 0:
        r2, rmse, mae = 1.0, 0.0, 0.0
    else:
        y_true_missing = y_true[missing_mask]
        y_pred_missing = y_pred[missing_mask]

        r2 = r2_score(y_true_missing, y_pred_missing)
        rmse = np.sqrt(mean_squared_error(y_true_missing, y_pred_missing))
        mae = mean_absolute_error(y_true_missing, y_pred_missing)

    print(f"\n[最终结果]")
    print(f"  缺失位置 R²:   {r2:.4f}")
    print(f"  缺失位置 RMSE: {rmse:.4f}")
    print(f"  缺失位置 MAE:  {mae:.4f}")

    # 统计迭代信息
    avg_iters = np.mean([h['n_iters'] for h in all_histories])
    converged_ratio = np.mean([h['converged'] for h in all_histories])
    print(f"  平均迭代次数: {avg_iters:.1f}")
    print(f"  收敛比例: {converged_ratio:.1%}")

    return {
        'r2': r2, 'rmse': rmse, 'mae': mae,
        'y_true': y_true, 'y_pred': y_pred, 'masks': masks,
        'histories': all_histories
    }


def compare_and_visualize(single_results, iter_results, save_dir='results/iterative_eval'):
    """对比两种方法并可视化"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"{'指标':<15} {'Single-Pass':<15} {'Iterative':<15} {'提升':<15}")
    print("-"*60)

    for metric in ['r2', 'rmse', 'mae']:
        s_val = single_results[metric]
        i_val = iter_results[metric]
        if metric == 'r2':
            improvement = i_val - s_val
            print(f"{metric.upper():<15} {s_val:<15.4f} {i_val:<15.4f} {improvement:+.4f}")
        else:
            improvement = (s_val - i_val) / s_val * 100
            print(f"{metric.upper():<15} {s_val:<15.4f} {i_val:<15.4f} {improvement:+.1f}%")

    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. R² 对比柱状图
    ax1 = axes[0]
    methods = ['Single-Pass', 'Iterative']
    r2_values = [single_results['r2'], iter_results['r2']]
    bars = ax1.bar(methods, r2_values, color=['#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('R²')
    ax1.set_title('R² Comparison')
    ax1.set_ylim(0, max(max(r2_values) * 1.2, 0.1))
    for bar, val in zip(bars, r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12)

    # 2. 预测值散点图对比
    ax2 = axes[1]
    masks = single_results['masks']
    missing_mask = (masks == 0)

    # 随机采样（避免点太多）
    n_missing = np.sum(missing_mask)
    if n_missing > 0:
        n_points = min(5000, n_missing)
        indices = np.random.choice(n_missing, n_points, replace=False)

        y_true = single_results['y_true'][missing_mask][indices]
        y_single = single_results['y_pred'][missing_mask][indices]
        y_iter = iter_results['y_pred'][missing_mask][indices]

        ax2.scatter(y_true, y_single, alpha=0.3, s=10, label='Single-Pass', c='orange')
        ax2.scatter(y_true, y_iter, alpha=0.3, s=10, label='Iterative', c='green')
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    
    ax2.set_xlabel('True Value')
    ax2.set_ylabel('Predicted Value')
    ax2.set_title('Prediction vs Truth (Missing Positions)')
    ax2.legend()

    # 3. 误差分布对比
    ax3 = axes[2]
    if n_missing > 0:
        error_single = np.abs(single_results['y_true'][missing_mask] - single_results['y_pred'][missing_mask])  
        error_iter = np.abs(iter_results['y_true'][missing_mask] - iter_results['y_pred'][missing_mask])        

        ax3.hist(error_single, bins=50, alpha=0.5, label=f'Single (MAE={np.mean(error_single):.4f})', color='orange')
        ax3.hist(error_iter, bins=50, alpha=0.5, label=f'Iterative (MAE={np.mean(error_iter):.4f})', color='green')
    
    ax3.set_xlabel('Absolute Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()

    plt.tight_layout()
    fig_path = save_path / 'comparison_single_vs_iterative.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n[OK] 对比图已保存: {fig_path}")

    # 保存结果JSON
    results = {
        'single_pass': {
            'r2': float(single_results['r2']),
            'rmse': float(single_results['rmse']),
            'mae': float(single_results['mae'])
        },
        'iterative': {
            'r2': float(iter_results['r2']),
            'rmse': float(iter_results['rmse']),
            'mae': float(iter_results['mae'])
        },
        'improvement': {
            'r2': float(iter_results['r2'] - single_results['r2']),
            'rmse_reduction': float((single_results['rmse'] - iter_results['rmse']) / (single_results['rmse'] + 1e-8) * 100),
            'mae_reduction': float((single_results['mae'] - iter_results['mae']) / (single_results['mae'] + 1e-8) * 100)
        }
    }

    results_path = save_path / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] 结果已保存: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='迭代填充评估')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v3',
                        help='模型检查点目录')
    parser.add_argument('--max_iters', type=int, default=5,
                        help='最大迭代次数')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='动量系数')
    parser.add_argument('--no_recompute_knn', action='store_true',
                        help='不重新计算KNN（简单迭代模式）')
    parser.add_argument('--save_dir', type=str, default='results/iterative_eval',
                        help='结果保存目录')

    args = parser.parse_args()

    print("="*60)
    print("迭代填充 vs Single-Pass 对比评估")
    print("="*60)

    # 加载模型和数据
    model, dataset, config = load_model_and_data(args.checkpoint_dir)

    # Single-Pass 评估
    single_results = evaluate_single_pass(model, dataset)

    # 迭代填充评估
    iter_results = evaluate_iterative(
        model, dataset,
        max_iters=args.max_iters,
        recompute_knn=not args.no_recompute_knn,
        momentum=args.momentum
    )

    # 对比可视化
    comparison = compare_and_visualize(single_results, iter_results, args.save_dir)

    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)

    # 给出建议
    r2_improvement = comparison['improvement']['r2']
    if r2_improvement > 0.05:
        print(f"\n[建议] 迭代填充显著提升 R² (+{r2_improvement:.4f})，建议:")
        print("  1. 增加迭代次数 (--max_iters 10)")
        print("  2. 调整动量 (--momentum 0.2)")
        print("  3. 考虑在训练中也使用迭代策略")
    elif r2_improvement > 0:
        print(f"\n[建议] 迭代填充有小幅提升 (+{r2_improvement:.4f})，可尝试:")
        print("  1. 增加迭代次数")
        print("  2. 检查是否收敛")
    else:
        print(f"\n[注意] 迭代填充未带来提升，可能原因:")
        print("  1. 模型本身预测能力不足")
        print("  2. KNN邻域质量已经较好")
        print("  3. 需要调整迭代参数")


if __name__ == '__main__':
    main()
