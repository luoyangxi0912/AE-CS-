#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探索性数据分析 (EDA) - Hangmei数据集
根据rightnowtesk.md任务清单详细执行
"""

import sys
import io

# 设置标准输出为UTF-8编码，避免Windows控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 配置设置
# ============================================================================

# 设置中文字体 (防止乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# 路径设置
DATA_PATH = Path(r"D:\数据补全\hangmei_90_拼接好的.csv")
RESULTS_DIR = Path(r"D:\数据补全\results\eda")
DIST_DIR = RESULTS_DIR / "distributions"

# 确保目录存在
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DIST_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Hangmei数据集探索性数据分析 (EDA)")
print("=" * 80)
print()

# ============================================================================
# 2. 读取数据
# ============================================================================

print("[步骤 1/7] 读取数据...")
try:
    df = pd.read_csv(DATA_PATH, encoding='gbk')
    print(f"✓ 数据读取成功！")
    print(f"  数据文件路径: {DATA_PATH}")
except Exception as e:
    print(f"✗ 数据读取失败: {e}")
    exit(1)

print()

# ============================================================================
# 3. 基础信息统计 (步骤 2)
# ============================================================================

print("[步骤 2/7] 基础信息统计...")
print("-" * 80)

# 数据集形状
n_rows, n_cols = df.shape
print(f"\n【数据集形状】")
print(f"  行数 (样本数): {n_rows}")
print(f"  列数 (特征数): {n_cols}")

# 列信息
print(f"\n【列信息】")
print(f"  总列数: {n_cols}")

# 识别列类型
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
datetime_cols = []

# 尝试识别时间列
for col in categorical_cols:
    try:
        pd.to_datetime(df[col].head(10))
        datetime_cols.append(col)
    except:
        pass

# 从类别列中移除时间列
categorical_cols = [col for col in categorical_cols if col not in datetime_cols]

print(f"  - 数值列: {len(numeric_cols)} 个")
print(f"  - 类别��: {len(categorical_cols)} 个")
print(f"  - 日期列: {len(datetime_cols)} 个")

# 打印前几列的名称
print(f"\n【前10列名称】")
for i, col in enumerate(df.columns[:10]):
    print(f"  {i+1}. {col}")
if n_cols > 10:
    print(f"  ... (共{n_cols}列)")

# 检查重复行
duplicates = df.duplicated().sum()
print(f"\n【重复行】")
print(f"  重复行数: {duplicates}")
if duplicates > 0:
    print(f"  重复率: {duplicates/n_rows*100:.2f}%")

# 基本统计量 (仅数值列)
if len(numeric_cols) > 0:
    print(f"\n【数值列基本统计】")
    summary_stats = df[numeric_cols].describe().T
    summary_stats['missing'] = df[numeric_cols].isnull().sum()
    summary_stats['missing_pct'] = (summary_stats['missing'] / n_rows * 100).round(2)

    # 保存完整统计信息
    summary_stats.to_csv(RESULTS_DIR / "basic_statistics.csv", encoding='utf-8-sig')
    print(f"  ✓ 完整统计信息已保存至: basic_statistics.csv")

    # 打印前5列的统计
    print(f"\n  前5个数值列统计示例:")
    print(summary_stats.head()[['mean', 'std', 'min', 'max', 'missing_pct']])

# 异常值检测 (IQR方法)
print(f"\n【异常值检测 (IQR方法)】")
outlier_info = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    n_outliers = len(outliers)

    if n_outliers > 0:
        outlier_info.append({
            'column': col,
            'n_outliers': n_outliers,
            'outlier_pct': round(n_outliers / n_rows * 100, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2)
        })

outlier_df = pd.DataFrame(outlier_info)
if len(outlier_df) > 0:
    outlier_df = outlier_df.sort_values('n_outliers', ascending=False)
    outlier_df.to_csv(RESULTS_DIR / "outlier_analysis.csv", index=False, encoding='utf-8-sig')
    print(f"  包含异常值的列: {len(outlier_df)} 个")
    print(f"  ✓ 异常值分析已保存至: outlier_analysis.csv")
    print(f"\n  Top 5 异常值最多的列:")
    print(outlier_df.head()[['column', 'n_outliers', 'outlier_pct']])
else:
    print(f"  未检测到异常值 (IQR方法)")

print()

# ============================================================================
# 4. 缺失值详细分析 (步骤 3)
# ============================================================================

print("[步骤 3/7] 缺失值详细分析...")
print("-" * 80)

# 计算缺失值
total_missing = df.isnull().sum().sum()
total_cells = n_rows * n_cols
missing_pct = (total_missing / total_cells) * 100

print(f"\n【缺失值概览】")
print(f"  总缺失值: {total_missing} 个")
print(f"  总单元格数: {total_cells} 个")
print(f"  总体缺失率: {missing_pct:.2f}%")

# 每列缺失值统计
missing_stats = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isnull().sum().values,
    'missing_pct': (df.isnull().sum().values / n_rows * 100).round(2)
})
missing_stats = missing_stats.sort_values('missing_count', ascending=False)

# 缺失值分类
complete_cols = missing_stats[missing_stats['missing_count'] == 0]
partial_missing_cols = missing_stats[(missing_stats['missing_count'] > 0) & (missing_stats['missing_count'] < n_rows)]
fully_missing_cols = missing_stats[missing_stats['missing_count'] == n_rows]

print(f"\n【缺失值分布】")
print(f"  完全无缺失的列: {len(complete_cols)} 个")
print(f"  部分缺失的列: {len(partial_missing_cols)} 个")
print(f"  完全缺失的列: {len(fully_missing_cols)} 个")

if len(partial_missing_cols) > 0:
    print(f"\n  缺失值最多的前10列:")
    top_missing = missing_stats.head(10)
    for idx, row in top_missing.iterrows():
        print(f"    {row['column']}: {row['missing_count']} 缺失 ({row['missing_pct']:.2f}%)")

# 保存缺失值统计
missing_stats.to_csv(RESULTS_DIR / "missing_value_analysis.csv", index=False, encoding='utf-8-sig')
print(f"\n  ✓ 缺失值分析已保存至: missing_value_analysis.csv")

# 生成缺失值可视化
print(f"\n【生成缺失值可视化】")

# 1. 缺失值矩阵
try:
    fig = plt.figure(figsize=(20, 10))
    msno.matrix(df, figsize=(20, 10), fontsize=10, sparkline=False)
    plt.title('Missing Value Matrix (缺失值矩阵)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "missing_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 缺失值矩阵图已生成: missing_matrix.png")
except Exception as e:
    print(f"  ✗ 缺失值矩阵图生成失败: {e}")

# 2. 缺失值热图
try:
    if len(partial_missing_cols) > 0:
        fig = plt.figure(figsize=(16, 12))
        msno.heatmap(df, figsize=(16, 12), fontsize=10)
        plt.title('Missing Value Heatmap (缺失值相关性热图)', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "missing_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 缺失值热图已生成: missing_heatmap.png")
except Exception as e:
    print(f"  ✗ 缺失值热图生成失败: {e}")

# 3. 缺失值树状图
try:
    if len(partial_missing_cols) > 0:
        fig = plt.figure(figsize=(16, 10))
        msno.dendrogram(df, fontsize=10)
        plt.title('Missing Value Dendrogram (缺失值聚类树状图)', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "missing_dendrogram.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 缺失值树状图已生成: missing_dendrogram.png")
except Exception as e:
    print(f"  ✗ 缺失值树状图生成失败: {e}")

# 4. 缺失值柱状图
try:
    if len(partial_missing_cols) > 0:
        top_20_missing = missing_stats[missing_stats['missing_count'] > 0].head(20)

        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.barh(range(len(top_20_missing)), top_20_missing['missing_pct'])
        ax.set_yticks(range(len(top_20_missing)))
        ax.set_yticklabels(top_20_missing['column'])
        ax.set_xlabel('Missing Percentage (%)', fontsize=12)
        ax.set_title('Top 20 Columns with Missing Values (缺失值最多的20列)', fontsize=14, pad=15)
        ax.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, (idx, row) in enumerate(top_20_missing.iterrows()):
            ax.text(row['missing_pct'], i, f"{row['missing_pct']:.1f}%",
                   va='center', ha='left', fontsize=9)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "missing_barplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 缺失值柱状图已生成: missing_barplot.png")
except Exception as e:
    print(f"  ✗ 缺失值柱状图生成失败: {e}")

print()

# ============================================================================
# 5. 数据分布分析 (步骤 4)
# ============================================================================

print("[步骤 4/7] 数据分布分析...")
print("-" * 80)

if len(numeric_cols) > 0:
    print(f"\n【数值列分布分析】")
    print(f"  共分析 {len(numeric_cols)} 个数值列")

    # 分布统计信息
    distribution_info = []

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)

            # Shapiro-Wilk正态性检验 (样本量限制)
            if len(data) <= 5000:
                try:
                    _, p_value = stats.shapiro(data)
                    is_normal = p_value > 0.05
                except:
                    p_value = np.nan
                    is_normal = False
            else:
                p_value = np.nan
                is_normal = False

            distribution_info.append({
                'column': col,
                'skewness': round(skewness, 3),
                'kurtosis': round(kurtosis, 3),
                'shapiro_p_value': round(p_value, 4) if not np.isnan(p_value) else 'N/A',
                'is_normal': is_normal
            })

    dist_df = pd.DataFrame(distribution_info)
    dist_df.to_csv(RESULTS_DIR / "distribution_summary.csv", index=False, encoding='utf-8-sig')
    print(f"  ✓ 分布统计已保存至: distribution_summary.csv")

    # 统计正态分布列
    normal_cols = dist_df[dist_df['is_normal'] == True]
    skewed_cols = dist_df[abs(dist_df['skewness']) > 1]

    print(f"\n  正态分布列 (Shapiro p>0.05): {len(normal_cols)} 个")
    print(f"  显著偏态列 (|skewness|>1): {len(skewed_cols)} 个")

    # 生成分布可视化
    print(f"\n【生成分布可视化】")

    # 1. 所有数值列的直方图
    try:
        n_numeric = len(numeric_cols)
        n_cols_plot = 5
        n_rows_plot = int(np.ceil(n_numeric / n_cols_plot))

        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(25, 5*n_rows_plot))
        axes = axes.flatten() if n_numeric > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            data = df[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
                ax.set_title(f'{col}\n(n={len(data)}, μ={data.mean():.2f})', fontsize=10)
                ax.set_xlabel('Value', fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                ax.grid(axis='y', alpha=0.3)

        # 隐藏多余的子图
        for i in range(n_numeric, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Distribution Histograms - All Numeric Columns (所有数值列分布直方图)',
                     fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(DIST_DIR / "all_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 所有分布直方图已生成: distributions/all_distributions.png")
    except Exception as e:
        print(f"  ✗ 分布直方图生成失败: {e}")

    # 2. 箱线图
    try:
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(25, 5*n_rows_plot))
        axes = axes.flatten() if n_numeric > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            data = df[col].dropna()
            if len(data) > 0:
                bp = ax.boxplot(data, vert=True, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                ax.set_title(f'{col}', fontsize=10)
                ax.set_ylabel('Value', fontsize=8)
                ax.grid(axis='y', alpha=0.3)

        for i in range(n_numeric, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Boxplots - All Numeric Columns (所有数值列箱线图)',
                     fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(DIST_DIR / "all_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 所有箱线图已生成: distributions/all_boxplots.png")
    except Exception as e:
        print(f"  ✗ 箱线图生成失败: {e}")

    # 3. Q-Q图 (选择前20列)
    try:
        n_qq = min(20, len(numeric_cols))
        qq_cols = numeric_cols[:n_qq]
        n_cols_qq = 5
        n_rows_qq = int(np.ceil(n_qq / n_cols_qq))

        fig, axes = plt.subplots(n_rows_qq, n_cols_qq, figsize=(25, 5*n_rows_qq))
        axes = axes.flatten() if n_qq > 1 else [axes]

        for i, col in enumerate(qq_cols):
            ax = axes[i]
            data = df[col].dropna()
            if len(data) > 0:
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f'{col}', fontsize=10)
                ax.grid(alpha=0.3)

        for i in range(n_qq, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Q-Q Plots - First 20 Numeric Columns (前20个数值列Q-Q图)',
                     fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(DIST_DIR / "qq_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Q-Q图已生成: distributions/qq_plots.png")
    except Exception as e:
        print(f"  ✗ Q-Q图生成失败: {e}")

    # 4. 相关性矩阵热图
    try:
        # 选择缺失值较少的数值列 (缺失<50%)
        valid_cols = [col for col in numeric_cols if df[col].isnull().sum() / len(df) < 0.5]

        if len(valid_cols) > 2:
            corr_matrix = df[valid_cols].corr()

            fig, ax = plt.subplots(figsize=(20, 18))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                       vmin=-1, vmax=1, ax=ax)
            ax.set_title('Correlation Matrix Heatmap (相关性矩阵热图)', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(DIST_DIR / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 相关性热图已生成: distributions/correlation_heatmap.png")

            # 找出高度相关的特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': round(corr_val, 3)
                        })

            if len(high_corr_pairs) > 0:
                high_corr_df = pd.DataFrame(high_corr_pairs)
                high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
                high_corr_df.to_csv(RESULTS_DIR / "high_correlation_pairs.csv",
                                   index=False, encoding='utf-8-sig')
                print(f"  ✓ 高相关特征对已保存: high_correlation_pairs.csv")
                print(f"    找到 {len(high_corr_pairs)} 对高度相关特征 (|r|>0.8)")
    except Exception as e:
        print(f"  ✗ 相关性分析失败: {e}")

print()

# ============================================================================
# 6. 时间序列特性分析 (步骤 5)
# ============================================================================

print("[步骤 5/7] 时间序列特性分析...")
print("-" * 80)

if len(datetime_cols) > 0:
    print(f"\n【时间列分析】")
    print(f"  识别到 {len(datetime_cols)} 个时间列:")

    for time_col in datetime_cols:
        print(f"\n  分析列: {time_col}")

        try:
            # 转换为datetime类型
            df[time_col] = pd.to_datetime(df[time_col])

            # 时间范围
            time_range = df[time_col].max() - df[time_col].min()
            print(f"    时间范围: {df[time_col].min()} 到 {df[time_col].max()}")
            print(f"    时间跨度: {time_range}")

            # 检查连续性
            df_sorted = df.sort_values(time_col)
            time_diffs = df_sorted[time_col].diff().dropna()

            if len(time_diffs) > 0:
                most_common_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                print(f"    最常见时间间隔: {most_common_diff}")

                # 检查缺失的时间点
                expected_points = int(time_range / most_common_diff) + 1
                actual_points = len(df)
                missing_points = expected_points - actual_points

                if missing_points > 0:
                    print(f"    ⚠ 可能缺失 {missing_points} 个时间点")

            # 绘制时间序列趋势图 (选择几个代表性的数值列)
            if len(numeric_cols) > 0:
                selected_features = numeric_cols[:6]  # 选择前6个特征

                fig, axes = plt.subplots(len(selected_features), 1,
                                        figsize=(16, 3*len(selected_features)))
                if len(selected_features) == 1:
                    axes = [axes]

                for i, feat in enumerate(selected_features):
                    ax = axes[i]
                    df_sorted.plot(x=time_col, y=feat, ax=ax, marker='.',
                                  markersize=2, linewidth=0.5)
                    ax.set_title(f'Time Series: {feat}', fontsize=12)
                    ax.set_xlabel('Time', fontsize=10)
                    ax.set_ylabel(feat, fontsize=10)
                    ax.grid(alpha=0.3)

                plt.suptitle(f'Time Series Trends (时间序列趋势)', fontsize=14, y=0.995)
                plt.tight_layout()
                plt.savefig(DIST_DIR / f"time_series_trends_{time_col}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    ✓ 时间序列趋势图已生成: distributions/time_series_trends_{time_col}.png")

        except Exception as e:
            print(f"    ✗ 时间列分析失败: {e}")
else:
    print(f"\n  未识别到明确的时间列")
    print(f"  如果数据包含时间信息，可能需要手动指定列名")

print()

# ============================================================================
# 7. 生成EDA报告 (步骤 6)
# ============================================================================

print("[步骤 6/7] 生成EDA报告...")
print("-" * 80)

report_content = f"""# 探索性数据分析报告 (EDA Report)

**数据集**: hangmei_90_拼接好的.csv
**分析日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析工具**: Python (pandas, numpy, matplotlib, seaborn, missingno)

---

## 1. 数据集概述

### 1.1 基本信息
- **样本数**: {n_rows}
- **特征数**: {n_cols}
- **数值列**: {len(numeric_cols)} 个
- **类别列**: {len(categorical_cols)} 个
- **日期列**: {len(datetime_cols)} 个
- **重复行**: {duplicates} 行 ({duplicates/n_rows*100:.2f}%)

### 1.2 数据类型分布
```
数值列 ({len(numeric_cols)}个): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
类别列 ({len(categorical_cols)}个): {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}
日期列 ({len(datetime_cols)}个): {', '.join(datetime_cols)}
```

---

## 2. 缺失值情况

### 2.1 总体缺失率
- **总缺失值**: {total_missing} 个
- **总单元格数**: {total_cells} 个
- **总体缺失率**: {missing_pct:.2f}%

### 2.2 缺失值分布
- **完全无缺失的列**: {len(complete_cols)} 个
- **部分缺失的列**: {len(partial_missing_cols)} 个
- **完全缺失的列**: {len(fully_missing_cols)} 个

### 2.3 缺失值最多的列 (Top 10)
"""

if len(partial_missing_cols) > 0:
    top_10_missing = missing_stats.head(10)
    report_content += "\n| 列名 | 缺失数量 | 缺失百分比 |\n|------|---------|------------|\n"
    for idx, row in top_10_missing.iterrows():
        report_content += f"| {row['column']} | {row['missing_count']} | {row['missing_pct']:.2f}% |\n"
else:
    report_content += "\n*数据集无缺失值*\n"

report_content += f"""

### 2.4 缺失值模式分析
根据缺失值可视化分析:
- 详见 `missing_matrix.png` - 缺失值矩阵图
- 详见 `missing_heatmap.png` - 缺失值相关性热图
- 详见 `missing_dendrogram.png` - 缺失值聚类树状图

**��步判断**:
- 缺失机制: [需根据可视化结果判断 MCAR/MAR/MNAR]
- 需要填补的列: {len(partial_missing_cols)} 个

---

## 3. 数据分布特征

### 3.1 分布类型统计
"""

if len(numeric_cols) > 0:
    normal_count = len(dist_df[dist_df['is_normal'] == True])
    skewed_count = len(dist_df[abs(dist_df['skewness']) > 1])

    report_content += f"""
- **正态分布列**: {normal_count} 个
- **显著偏态列 (|skewness|>1)**: {skewed_count} 个
- **需要变换的列**: {skewed_count} 个 (建议对偏态列进行log/sqrt变换)

### 3.2 偏度和峰度分析
详见 `distribution_summary.csv` 完整统计信息

### 3.3 偏态最严重的列 (Top 5)
"""

    top_skewed = dist_df.reindex(dist_df['skewness'].abs().sort_values(ascending=False).index).head(5)
    report_content += "\n| 列名 | 偏度 | 峰度 |\n|------|------|------|\n"
    for idx, row in top_skewed.iterrows():
        report_content += f"| {row['column']} | {row['skewness']} | {row['kurtosis']} |\n"

report_content += f"""

---

## 4. 异常值情况

### 4.1 异常值检测 (IQR方法)
- **包含异常值的列**: {len(outlier_df) if len(outlier_df) > 0 else 0} 个
"""

if len(outlier_df) > 0:
    report_content += f"""
### 4.2 异常值最多的列 (Top 10)

| 列名 | 异常值数量 | 异常值百分比 |
|------|-----------|--------------|
"""
    top_10_outliers = outlier_df.head(10)
    for idx, row in top_10_outliers.iterrows():
        report_content += f"| {row['column']} | {row['n_outliers']} | {row['outlier_pct']:.2f}% |\n"

    report_content += """
**处理建议**:
- 异常值百分比<1%: 可考虑删除
- 异常值百分比 1-5%: 建议使用Winsorization或Cap方法
- 异常值百分比>5%: 需仔细检查，可能是真实数据特性
"""
else:
    report_content += "\n*未检测到异常值 (IQR方法)*\n"

report_content += f"""

---

## 5. 特征相关性

### 5.1 相关性分析
详见 `distributions/correlation_heatmap.png`
"""

try:
    if len(high_corr_pairs) > 0:
        report_content += f"""
### 5.2 高度相关特征对 (|r|>0.8)
找到 **{len(high_corr_pairs)}** 对高度相关特征

**Top 10 相关对**:

| 特征1 | 特征2 | 相关系数 |
|-------|-------|----------|
"""
        top_corr = high_corr_df.head(10)
        for idx, row in top_corr.iterrows():
            report_content += f"| {row['feature_1']} | {row['feature_2']} | {row['correlation']} |\n"

        report_content += """
**建议**: 考虑删除冗余特征以减少多重共线性
"""
    else:
        report_content += "\n*未发现高度相关的特征对 (|r|>0.8)*\n"
except:
    report_content += "\n*相关性分析未执行*\n"

report_content += f"""

---

## 6. 时间序列特性

"""

if len(datetime_cols) > 0:
    for time_col in datetime_cols:
        try:
            time_range = df[time_col].max() - df[time_col].min()
            report_content += f"""
### 时间列: {time_col}
- **时间范围**: {df[time_col].min()} 到 {df[time_col].max()}
- **时间跨度**: {time_range}
- **样本数**: {len(df)}

详见时间序列趋势图: `distributions/time_series_trends_{time_col}.png`
"""
        except:
            pass
else:
    report_content += "*未识别到明确的时间列*\n"

report_content += f"""

---

## 7. 数据质量评估

### 7.1 数据完整性
- **缺失率**: {missing_pct:.2f}% - {'优秀' if missing_pct < 5 else '良好' if missing_pct < 10 else '中等' if missing_pct < 20 else '较差'}
- **重复率**: {duplicates/n_rows*100:.2f}% - {'优秀' if duplicates == 0 else '需处理'}

### 7.2 数据分布
- **正态分布比例**: {normal_count/len(numeric_cols)*100:.1f}% ({normal_count}/{len(numeric_cols)})
- **偏态分布比例**: {skewed_count/len(numeric_cols)*100:.1f}% ({skewed_count}/{len(numeric_cols)})

### 7.3 数据异常
- **异常值列数**: {len(outlier_df) if len(outlier_df) > 0 else 0} / {len(numeric_cols)}

### 7.4 总体评价
"""

# 数据质量评分
quality_score = 100
quality_score -= min(missing_pct, 30)  # 缺失率扣分
quality_score -= min(duplicates/n_rows*100*2, 10)  # 重复率扣分
quality_score -= min(len(outlier_df)/len(numeric_cols)*20, 20) if len(outlier_df) > 0 else 0  # 异常值扣分

if quality_score >= 90:
    quality_level = "优秀"
elif quality_score >= 75:
    quality_level = "良好"
elif quality_score >= 60:
    quality_level = "中等"
else:
    quality_level = "较差"

report_content += f"""
**数据质量得分**: {quality_score:.1f}/100 - **{quality_level}**

---

## 8. 预处理建议

### 8.1 缺失值处理
"""

if missing_pct < 5:
    report_content += "- 缺失率较低，建议使用简单填补方法（均值/中位数/众数）\n"
elif missing_pct < 20:
    report_content += "- 缺失率中等，建议使用高级填补方法（KNN/MICE/深度学习）\n"
else:
    report_content += "- 缺失率较高，建议使用深度学习方法（如本项目的AE-CS模型）\n"

report_content += f"""
- 完全缺失的列 ({len(fully_missing_cols)}个): 建议删除
- 部分缺失的列 ({len(partial_missing_cols)}个): 需要填补

### 8.2 数据归一化
"""

if skewed_count > len(numeric_cols) * 0.3:
    report_content += f"- **{skewed_count}** 个列存在显著偏态，建议:\n"
    report_content += "  - 对正偏态列进行log变换或sqrt变换\n"
    report_content += "  - 对负偏态列进行平方变换或指数变换\n"
    report_content += "  - 或使用RobustScaler进行标准化\n"
else:
    report_content += "- 大部分列分布较为正常，建议使用StandardScaler或MinMaxScaler\n"

report_content += f"""
### 8.3 异常值处理
"""

if len(outlier_df) > 0:
    report_content += f"- **{len(outlier_df)}** 个列包含异常值，建议:\n"
    report_content += "  - 百分比<1%的列: 直接删除异常值\n"
    report_content += "  - 百分比1-5%的列: 使用Winsorization (限幅处理)\n"
    report_content += "  - 百分比>5%的列: 仔细检查，可能是真实特性，保留\n"
else:
    report_content += "- 未检测到明显异常值，无需特殊处理\n"

report_content += f"""
### 8.4 特征工程
"""

try:
    if len(high_corr_pairs) > 0:
        report_content += f"- **{len(high_corr_pairs)}** 对高度相关特征，建议:\n"
        report_content += "  - 使用VIF (方差膨胀因子) 进一步检测多重共线性\n"
        report_content += "  - 删除冗余特征或进行PCA降维\n"
except:
    pass

if len(datetime_cols) > 0:
    report_content += "- 提取时间特征: 年、月、日、小时、星期等\n"
    report_content += "- 考虑创建时间序列滞后特征\n"

report_content += f"""
### 8.5 数据划分建议
- **训练集**: 70% ({int(n_rows*0.7)} 样本)
- **验证集**: 15% ({int(n_rows*0.15)} 样本)
- **测试集**: 15% ({int(n_rows*0.15)} 样本)

**注意**: 如果是时间序列数据，使用时间顺序划分而非随机划分

---

## 9. 关键发现总结

### 9.1 数据规模
- 样本数: **{n_rows}**
- 特征数: **{n_cols}**
- 时间跨度: {'已识别' if len(datetime_cols) > 0 else '未明确'}

### 9.2 缺失值情况
- 总体缺失率: **{missing_pct:.2f}%**
- 缺失值模式: [需根据可视化结果判断]
- 需要填补的列: **{len(partial_missing_cols)}** 个

### 9.3 数据分布
- 正态分布列: **{normal_count}** 个 ({normal_count/len(numeric_cols)*100:.1f}%)
- 偏态分布列: **{skewed_count}** 个 ({skewed_count/len(numeric_cols)*100:.1f}%)
- 需要变换的列: **{skewed_count}** 个

### 9.4 异常值
- 包含异常值的列: **{len(outlier_df) if len(outlier_df) > 0 else 0}** 个
- 异常值处理建议: 详见第8.3节

### 9.5 特征相关性
"""

try:
    if len(high_corr_pairs) > 0:
        report_content += f"- 高度相关特征对 (|r|>0.8): **{len(high_corr_pairs)}** 对\n"
        report_content += "- 建议删除的冗余特征: 详见 `high_correlation_pairs.csv`\n"
    else:
        report_content += "- 未发现显著的特征冗余\n"
except:
    report_content += "- 相关性分析未完成\n"

report_content += f"""

---

## 10. 下一步工作

### 10.1 数据预处理
1. ���现数据加载器 (`data/dataset.py`)
2. 实现数据预处理器 (`data/preprocessor.py`)
   - 缺失值掩码生成
   - 数据归一化/标准化
   - 异常值处理
   - 时间窗口切分

### 10.2 模型开发
1. 完善AE-CS模型中的邻域搜索模块
2. 集成FAISS加速k-NN搜索
3. 实现训练脚本 (`train.py`)
4. 实现推理脚本 (`inference.py`)

### 10.3 实验验证
1. 设计缺失值填补实验
2. 与baseline方法对比 (KNN, MICE, BRITS等)
3. 评估填补质量 (MAE, RMSE, MRE)

---

## 附录: 生成的文件清单

### 统计文件
- `basic_statistics.csv` - 基础统计信息
- `missing_value_analysis.csv` - 缺失值分析
- `distribution_summary.csv` - 分布统计
- `outlier_analysis.csv` - 异常值分析
- `high_correlation_pairs.csv` - 高相关特征对

### 可视化文件
- `missing_matrix.png` - 缺失值矩阵图
- `missing_heatmap.png` - 缺失值相关性热图
- `missing_dendrogram.png` - 缺失值聚类树状图
- `missing_barplot.png` - 缺失值柱状图
- `distributions/all_distributions.png` - 所有分布直方图
- `distributions/all_boxplots.png` - 所有箱线图
- `distributions/qq_plots.png` - Q-Q图
- `distributions/correlation_heatmap.png` - 相关性热图
- `distributions/time_series_trends_*.png` - 时间序列趋势图

---

**报告生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# 保存报告
report_path = RESULTS_DIR / "EDA_REPORT.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✓ EDA报告已生成: {report_path}")
print()

# ============================================================================
# 8. 验证和检查 (步骤 7)
# ============================================================================

print("[步骤 7/7] 验证和检查...")
print("-" * 80)

# 检查所有输出文件
required_files = {
    'basic_statistics.csv': RESULTS_DIR / 'basic_statistics.csv',
    'missing_value_analysis.csv': RESULTS_DIR / 'missing_value_analysis.csv',
    'distribution_summary.csv': RESULTS_DIR / 'distribution_summary.csv',
    'EDA_REPORT.md': RESULTS_DIR / 'EDA_REPORT.md',
    'missing_matrix.png': RESULTS_DIR / 'missing_matrix.png',
    'all_distributions.png': DIST_DIR / 'all_distributions.png',
    'all_boxplots.png': DIST_DIR / 'all_boxplots.png',
    'qq_plots.png': DIST_DIR / 'qq_plots.png',
    'correlation_heatmap.png': DIST_DIR / 'correlation_heatmap.png',
}

print(f"\n【文件验证】")
all_ok = True
for name, path in required_files.items():
    if path.exists():
        file_size = path.stat().st_size / 1024  # KB
        print(f"  ✓ {name} ({file_size:.1f} KB)")
    else:
        print(f"  ✗ {name} - 文件未找到")
        all_ok = False

# 数据合理性检查
print(f"\n【数据合理性检查】")
checks_passed = 0
total_checks = 5

# 检查1: 样本数>0
if n_rows > 0:
    print(f"  ✓ 样本数检查通过 (n={n_rows})")
    checks_passed += 1
else:
    print(f"  ✗ 样本数异常")

# 检查2: 特征数>0
if n_cols > 0:
    print(f"  ✓ 特征数检查通过 (n={n_cols})")
    checks_passed += 1
else:
    print(f"  ✗ 特征数异常")

# 检查3: 缺失率在合理范围
if 0 <= missing_pct <= 100:
    print(f"  ✓ 缺失率检查通过 ({missing_pct:.2f}%)")
    checks_passed += 1
else:
    print(f"  ✗ 缺失率异常")

# 检查4: 统计值无NaN或Inf
if len(numeric_cols) > 0:
    has_nan = df[numeric_cols].describe().isnull().any().any()
    has_inf = np.isinf(df[numeric_cols].describe()).any().any()
    if not has_nan and not has_inf:
        print(f"  ✓ 统计值检查通过 (无NaN或Inf)")
        checks_passed += 1
    else:
        print(f"  ✗ 统计值存在NaN或Inf")

# 检查5: 图片分辨率
try:
    from PIL import Image
    test_img = Image.open(RESULTS_DIR / 'missing_matrix.png')
    dpi = test_img.info.get('dpi', (72, 72))
    if dpi[0] >= 300:
        print(f"  ✓ 图片分辨率检查通过 (DPI={dpi[0]})")
        checks_passed += 1
    else:
        print(f"  ⚠ 图片分辨率偏低 (DPI={dpi[0]})")
except:
    print(f"  ⚠ 无法检查图片分辨率")

print(f"\n【总体检查结果】")
print(f"  通过检查: {checks_passed}/{total_checks}")
if all_ok and checks_passed >= 4:
    print(f"  ✓ EDA任务成功完成！")
else:
    print(f"  ⚠ 部分检查未通过，请review")

# ============================================================================
# 完成
# ============================================================================

print()
print("=" * 80)
print("EDA分析完成!")
print("=" * 80)
print(f"\n所有结果已保存至: {RESULTS_DIR}")
print(f"\n下一步建议:")
print(f"  1. 查看 EDA_REPORT.md 了解详细分析结果")
print(f"  2. 查看 results/eda/ 目录下的所有可视化图表")
print(f"  3. 根据分析结果设计数据预处理策略")
print(f"  4. 开始实现 data/preprocessor.py 和 data/dataset.py")
print()
