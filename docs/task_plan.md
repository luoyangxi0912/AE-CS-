# AE-CS 实验任务计划

> 目标：让 AE-CS 模型性能超过 KNN 基线（R²=0.93），达到可发表水平
> 创建日期：2026-03-24
> 当前状态：V11 代码修复完成，待重新训练验证

---

## 当前进展

- 已完成 V9 架构重写（BN→LN、Encoder pre_filled、consistency sigma_c 修复）
- 已在 AutoDL GPU 服务器完成首次训练（train_cloud.py, Epoch 40 早停, 60 epochs 总计）
- 已完成首次评估：缺失位 R²=0.9275, 整体 R²=0.9856
- 已完成 V10 训练（`checkpoints_v10`，早停 Epoch 34，最佳验证 recon=0.1385）
- 已完成 V10 评估（`results_v10`）：缺失位 R²=0.9271，RMSE=0.1637，MAE=0.0905
- **V10 结论：相比 V9 未提升主指标（R² 0.9275 → 0.9271），仍低于 KNN 基线 0.93**
- **V11 代码修复完成（2026-03-25）：修复 5 个 bug，详见下方任务 2**

### V9 vs V10 对比（缺失位）

| 版本 | R² | RMSE | MAE | 结论 |
|------|----|------|-----|------|
| V9 | 0.9275 | 0.1633 | 0.0892 | - |
| V10 | 0.9271 | 0.1637 | 0.0905 | 略退化 |
| V11 | 待训练 | - | - | 邻域损失 + 幅度信息修复 |

---

## 任务 1：修正训练超参数默认值 — **已完成**

### 解决的问题
当前超参数导致训练-测试分布不匹配、损失函数失衡，模型学到的 delta 在测试时过度修正。

### 已作决策

**决策 1.1：p_drop 从 0.5 → 0.2**
- 理由：p_drop=0.5 时训练输入约 60% 缺失，测试仅 20%。KNN 在训练时用残缺数据填充（质量差），模型学到"大力修正"的 delta。测试时 KNN 用完整数据填充（质量好），同样的大 delta 反而过度修正 → 劣化基线。
- 排除 p_drop=0.3：仍有 44% vs 20% 的偏移，不彻底
- 排除保持 0.5：已实验证明有害（R² < 基线）

**决策 1.2：lambda1 从 1.0 → 0.5**
- 理由：训练日志显示 L_consist=0.756 vs L_recon=0.131，一致性梯度是重建梯度的 5.7 倍。模型优先学"对 mask 扰动不敏感"，而非"填补准确"。
- 排除 lambda1=0.1：过度削弱一致性，表示可能坍塌（BYOL/SimSiam 机制需要足够的一致性约束）
- 排除保持 1.0：已证明导致梯度失衡

**决策 1.3：dropout 从 0.3 → 0.1**
- 理由：在 recon 梯度已被 consist 挤压的情况下，0.3 的 dropout 进一步限制模型容量。
- 排除 0.0：三编码器容易过拟合，需要最低限度正则化
- 排除 0.2：0.1 已在 BN→LN 修复后确认有效

**决策 1.4：lambda2/lambda3 初始值 0.5/0.5**
- 理由：V11 修复邻域损失后，L_space/L_time 将有实际梯度贡献，0.5 作为中性起点，根据训练日志微调。

### 状态
- [x] 修改 train_iterative.py 默认参数（V10 已完成）
- [ ] 修改 train_cloud.py 默认参数
- [ ] 推送到 GitHub

---

## 任务 2：排查并修复邻域损失为零 — **已完成**

### 解决的问题
L_space≈0.0006、L_time=0.0000，邻域保持损失几乎不贡献梯度。论文架构中 L_space 和 L_time 是核心组件，用于保持潜在空间的时空流形结构。若它们无效，模型退化为普通 autoencoder + 一致性正则。

### 根因分析与修复（2026-03-25 代码审查发现）

发现 5 个 bug，已全部修复并通过编译 + 梯度冒烟测试验证：

**Bug 1（P0）：L_time 梯度完全断裂**
- 根因：`compute_temporal_neighborhood_with_mapping` 中 `z_b = z[b].numpy()` 将 TF tensor 转为 numpy，随后 `z_var = tf.constant(np.stack(...))` 创建无梯度的常量。L_time 虽能计算出数值，但 `tape.gradient()` 对所有模型参数返回零梯度。
- 修复：用 `tf.gather` + `tf.reduce_mean` + `tf.stack` 重写 z_var 计算，保持 z → z_var → z_var_neighbors 的完整梯度链。
- 文件：`models/neighborhood.py`
- 验证：`grad is None = False`，`sum(z_var)` 对 z 的梯度范数非零。

**Bug 2（P1）：L_space/L_time 只正则化 encoder_orig，不影响 encoder_space/encoder_time**
- 根因：`AECS.call()` 中 `compute_neighborhood_embeddings` 只传入 `z_orig`，导致 `z_neighbors_space` 和 `z_var` 都从 encoder_orig 的输出收集。空间/时间编码器完全没有邻域保持约束。
- 修复：L_space 改为从 z_space 收集邻居 + 用 z_space 做锚点；L_time 从 z_time 计算 z_var。
- 文件：`models/ae_cs.py`（neighborhood_info 构造）、`models/losses.py`（`z_space_anchor`）

**Bug 3（P2）：Decoder 收到 L2 归一化后的 z_fused，丢失幅度信息**
- 根因：z_fused 是 L2-normalized 向量的加权组合，编码器输出的幅度信息全部丢失，decoder 只能从方向信息重建 delta。
- 修复：gating 网络仍在归一化空间计算 alpha（保证稳定性），但 z_fused 改为未归一化 z 的加权组合。
- 文件：`models/ae_cs.py`（`AECS.call()` 和 `fuse_representations()`）

**Bug 4（P3）：evaluate.py 未从 config 恢复 stride**
- 根因：评估默认 stride=1（训练 stride=12），产生大量重叠窗口浪费计算。
- 修复：`data_loader.preprocessor.stride = config.get('stride', 12)`
- 文件：`evaluate.py`

**Bug 5（P3）：reconstruction_loss 文档字符串描述不存在的 corrupted_mask 参数**
- 修复：更新文档，准确描述 mask 参数含义。
- 文件：`models/losses.py`

**性能优化：消除重复邻域计算**
- 旧代码：调用两次 `compute_neighborhood_embeddings`（一次 z_space，一次 z_time），空间 KNN 搜索（基于 X）重复计算。
- 优化：只调用一次（传 z_time），复用 `indices_space` 对 z_space 做 `tf.gather`。

### 状态
- [x] 根因确认：L_time=0 是梯度断裂（非表示坍塌）
- [x] L_time 梯度修复 + 冒烟测试通过
- [x] L_space/L_time 编码器正则化目标修复
- [x] Decoder 幅度信息修复
- [x] evaluate.py stride 修复
- [x] 语法编译通过（python -m compileall）

---

## 任务 3：在 AutoDL 上重新训练（V11） — **待执行**

### 解决的问题
用修正后的代码重新训练，验证邻域损失修复是否使模型超过 KNN 基线。

### 已作决策

**决策 3.1：早停指标从 val_total → val_recon**（V10 已实现）

**决策 3.2：V11 训练要点**
- 代码基础：包含任务 2 的全部 bug 修复
- 关键变化：L_space 和 L_time 首次有实际梯度贡献
- 需关注：训练日志中 L_space 和 L_time 的量级，据此调整 lambda2/lambda3

**决策 3.3：训练命令**
```bash
python train_cloud.py \
  --p_drop 0.2 \
  --dropout_rate 0.1 \
  --lambda1 0.5 \
  --lambda2 0.5 \
  --lambda3 0.5 \
  --epochs 100 \
  --early_stopping_patience 20 \
  --checkpoint_dir checkpoints_v11
```

### 待解决
- [x] 已执行 V10 训练与评估（AutoDL）
- [x] 已对比 V9 与 V10 的 per-feature 性能
- [ ] **将 V11 代码修复推送到 GitHub**
- [ ] 在 AutoDL 上执行 V11 训练
- [ ] 监控训练日志：确认 L_space > 0、L_time > 0、L_recon 梯度占比提升
- [ ] 若 L_space/L_time 量级过大或过小，微调 lambda2/lambda3
- [ ] V11 评估并与 V9/V10 对比，验证是否超过 R²=0.93

---

## 任务 4：跑完 5 组实验配置

### 解决的问题
论文需要完整的实验矩阵，不同缺失率和缺失模式下的性能对比。

### 已作决策

**决策 4.1：5 组配置**
| 配置 | 缺失率 | 缺失类型 |
|------|--------|----------|
| C1   | 0.1    | MCAR     |
| C2   | 0.2    | MCAR     |
| C3   | 0.3    | MCAR     |
| C4   | 0.4    | MCAR     |
| C5   | 0.2    | MAR      |

- 排除 MNAR（Missing Not at Random）：数据集无法确定 MNAR 机制，强行模拟不可信
- 排除更多缺失率档位：5 组已覆盖主要范围，更多增加训练时间但信息增量小

### 待解决
- [ ] **前置：实现 mask 持久化脚本**（生成固定切分 + 固定缺失 mask 的 .npz，所有方法共用）
- [ ] 任务 3 单配置验证通过后再启动
- [ ] 每组需独立完整训练（不能共享 checkpoint）
- [ ] 预估总耗时（GPU 上每组约 1-2 小时）

---

## 任务 5：基线对比与论文数据整理

### 解决的问题
论文需要与多种基线方法的定量对比表格，证明 AE-CS 的优势。

### 已作决策

**决策 5.1：基线方法选择**
| 方法 | 类型 | 理由 |
|------|------|------|
| Mean Imputation | 统计 | 最简单基线，下界参考 |
| KNN Imputation   | 统计 | 当前基线 R²=0.93 |
| MICE             | 统计 | 经典多重填补，审稿人必问 |
| MissForest       | ML   | 随机森林填补，近年常用 |
| BRITS            | DL   | 双向 RNN 填补，时序数据 SOTA 级 |

- 排除 GAIN（GAN-based）：实现复杂且在工业时序数据上不稳定
- 排除 Transformer-based（SAITS 等）：需要大量数据，2793 时间点不够

**决策 5.2：评估指标**
- 主指标：R²、RMSE、MAE（缺失位置）
- 辅助指标：per-feature R² 分布图、不同缺失率下的性能曲线

### 待解决
- [ ] **前置：加载任务 4 生成的 .npz mask 文件**，确保基线方法使用完全相同的切分和缺失模式
- [ ] BRITS 需要单独实现或找开源代码
- [ ] 确认 MICE/MissForest 在 scikit-learn/fancyimpute 中的可用性

---

## 全局约束

1. **实验公平性（最高优先级）**：所有方法（AE-CS + 全部基线）必须共用同一数据切分（train/val/test 时间索引）和同一缺失 mask（per-config 固定随机种子生成）。这是论文对比实验的硬性前提，不满足则审稿人可直接拒稿。需在任务 4 之前实现 mask 持久化脚本，将切分索引和缺失 mask 保存为 .npz 文件供所有方法加载。
2. **CPU 限制**：本地 GPU（GTX 960M, 960MB）不可用，大规模训练必须在 AutoDL 上进行
3. **数据量小**：仅 2793 × 44，不支持过深的网络或需要大量数据的方法
4. **Feature 11 特殊处理**：方差极小，delta 必须通过 feat_delta_mask 强制为 0
5. **不可截断数据/减少 epochs**：CLAUDE.md 明确禁止为加速而牺牲完整性
6. **分布偏移已确认**：Train(0.006, 0.75) vs Val(-0.43, 1.88) vs Test(0.34, 0.60)，gap-based split 导致的真实偏移，不是 bug
7. **AutoDL 服务器按时计费**：需要高效利用，避免无效训练消耗资源

---

## 执行顺序

```
任务1 (修参数) ──── 已完成
                          ↘
任务2 (修邻域损失) ── 已完成 → 任务3 (V11 重新训练) ← 当前步骤
                                        ↓
                                   任务4 (5组实验) → 任务5 (基线对比)
```
