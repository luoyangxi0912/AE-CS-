"""
AE-CS Models Package

核心组件（符合专利步骤S2-S6）：
- Encoder: 编码器网络
- Decoder: 解码器网络
- GatingNetwork: 自适应融合门控网络
- AECS: 完整的三编码器-单解码器模型（专利架构）
- AECS_Legacy: 旧版单编码器模型（向后兼容）

输入初始化函数：
- compute_spatial_knn_init: 空间K近邻均值初始化
- compute_temporal_knn_init: 时间K近邻均值初始化

损失函数：
- reconstruction_loss: 重建损失（仅在观测位置计算）
- consistency_loss: 一致性损失（加权约束）
- spatial_preservation_loss: 空间流形保持损失
- temporal_preservation_loss: 时间流形保持损失
- total_loss: 联合损失函数

一致性增强函数：
- bernoulli_corruption: Bernoulli掩码损坏
- bernoulli_corruption_with_weight: 带权重的掩码损坏
- generate_augmented_masks: 生成Q个增强掩码及权重
- compute_consistency_weights: 计算一致性权重

循环插补（可选扩展）：
- IterativeImputer: 循环插补器
- adaptive_iterative_impute: 自适应循环插补
"""

from .ae_cs import (
    Encoder,
    Decoder,
    GatingNetwork,
    AECS,
    AECS_Legacy,
    gaussian_activation,
    compute_spatial_knn_init,
    compute_temporal_knn_init
)
from .losses import (
    reconstruction_loss,
    missing_position_loss,
    consistency_loss,
    spatial_preservation_loss,
    temporal_preservation_loss,
    total_loss,
    bernoulli_corruption,
    bernoulli_corruption_with_weight,
    generate_augmented_masks,
    compute_consistency_weights
)
from .neighborhood import (
    FAISSKNNSearcher,
    NeighborhoodModule,
    weighted_aggregation
)
from .iterative_imputation import (
    IterativeImputer,
    adaptive_iterative_impute,
    compute_spatial_knn_init_iterative,
    compute_temporal_knn_init_iterative
)

__all__ = [
    # 模型组件
    'Encoder',
    'Decoder',
    'GatingNetwork',
    'AECS',
    'AECS_Legacy',
    'gaussian_activation',
    # 输入初始化
    'compute_spatial_knn_init',
    'compute_temporal_knn_init',
    # 损失函数
    'reconstruction_loss',
    'missing_position_loss',
    'consistency_loss',
    'spatial_preservation_loss',
    'temporal_preservation_loss',
    'total_loss',
    # 一致性增强
    'bernoulli_corruption',
    'bernoulli_corruption_with_weight',
    'generate_augmented_masks',
    'compute_consistency_weights',
    # 邻域模块
    'FAISSKNNSearcher',
    'NeighborhoodModule',
    'weighted_aggregation',
    # 循环插补
    'IterativeImputer',
    'adaptive_iterative_impute',
    'compute_spatial_knn_init_iterative',
    'compute_temporal_knn_init_iterative'
]
