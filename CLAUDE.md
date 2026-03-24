# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AE-CS: AutoEncoder with Coherent denoising and Spatio-temporal neighborhood-preserving embedding for missing data imputation in multivariate time series (industrial process data).

## Environment

- **Python 3.10** at `C:/Python310/python.exe` (TensorFlow 2.10 installed here)
- **CPU-only** — GPU is GTX 960M (960MB, unusable). TensorFlow forces CPU via `CUDA_VISIBLE_DEVICES=-1`.
- Training takes ~30-40 min/epoch. Full 5-config evaluation takes days. Never truncate data or reduce epochs for speed.

## Commands

```bash
# Train (single config)
python train.py --missing_rate 0.2 --missing_type MCAR --epochs 100 --checkpoint_dir checkpoints_v9

# Resume interrupted training (ALWAYS use --resume for continuation)
python train.py --checkpoint_dir checkpoints_v9 --resume

# Train with iterative denoising (paper-correct architecture)
python train_iterative.py --checkpoint_dir checkpoints_v9 --epochs 100

# Evaluate
python evaluate.py --checkpoint_dir checkpoints_v9 --data_path hangmei_90_拼接好的.csv

# Run all 5 experiment configs
python run_experiments.py
```

## Architecture

```
Input: X ∈ R^(T×N), M ∈ {0,1}^(T×N)  (T=timesteps, N=44 features)

[X⊙M ‖ M]───→ Encoder_orig ──→ Z_orig ─┐
                                          │
[X_space ‖ M]→ Encoder_space → Z_space ─┤→ Gating(ρ) → Z_fused → Decoder → Δx
                                          │   Network          ↓
[X_time ‖ M]─→ Encoder_time ─→ Z_time ─┘          X̂ = X_space_init + Δx
```

- **3 encoders** (same structure, independent weights): 2×LSTM(128) → LayerNorm → Dense(32)
- **Decoder**: 2×LSTM(128) → LayerNorm → Dense(44) → clip[-2.0, 2.0]
- **Gating**: GAP(Z) + ρ → Dense layers with gaussian activation → softmax → α₁,α₂,α₃
- **Residual**: `x_hat = x_space_init + x_delta` (KNN baseline + learned correction)

### Key modules

| File | Purpose |
|------|---------|
| `models/ae_cs.py` | AECS model, Encoder, Decoder, GatingNetwork, KNN init functions |
| `models/losses.py` | L_recon, L_consist, L_space, L_time, total_loss |
| `models/neighborhood.py` | Spatial/temporal KNN with partial distance (FAISS) |
| `data/preprocessor.py` | Z-score normalization, windowing, gap-based train/val/test split |
| `data/dataset.py` | AECSDataset, AECSDataLoader (TF dataset wrappers) |
| `train_iterative.py` | AECSTrainerV2: paper-correct training with denoising + consistency |
| `evaluate.py` | Test set evaluation with R², RMSE, MAE metrics |

### Loss function

```
L_total = L_recon + λ₁·L_consist + λ₂·L_space + λ₃·L_time
```

- **L_recon**: MSE on extra-masked positions only (not all observed — see V9 fix below)
- **L_consist**: Weighted L2 between Z_orig and K corrupted-mask encodings, with stop_gradient anchor
- **L_space/L_time**: Weighted distance on L2-normalized latent vectors (prevents representation collapse)

### Training strategy

Self-supervised mask reconstruction: extra-mask p_drop of observed values during forward pass, but compute loss on original mask. This creates supervised signal for learning corrections at "pseudo-missing" positions.

## Critical design decisions

- **LayerNorm not BatchNorm**: Three encoders receive inputs with different density (20% zeros vs KNN-filled). BN learns different running stats → 18x Z norm mismatch. LN normalizes per-sample.
- **Feature 11 delta mask**: Near-zero variance feature. Any non-zero delta causes R²=-838M. Delta forced to 0 via `feat_delta_mask`.
- **Partial distance for KNN**: Only uses co-observed variables for distance, avoiding false similarity from shared zeros.
- **Gap-based time split**: buffer_zone=window_size between train/val/test to prevent leakage.

## V9 critical fix (latest)

**Problem**: With residual connection `x_hat = x_knn + delta`, 80% of L_recon positions have x_knn=x_true (observed positions kept by KNN init), so optimal delta=0. Only 20% extra-masked positions need non-zero delta. The zero-signal dominates → delta never learns.

**Fix**: `reconstruction_loss` now accepts `corrupted_mask` parameter. When provided, computes loss only on `mask * (1 - corrupted_mask)` positions (where delta targets are non-zero). Combined with p_drop=0.5 (was 0.2) for stronger signal.

## Operational rules

- **Never kill background training** without explicit user instruction.
- **Always use `--resume`** when continuing training.
- Checkpoint format: `training_state.json` + `best_model.weights.h5` in checkpoint dir.
- Evaluation and plots only after all 5 configs complete their full epochs.
- Data file: `hangmei_90_拼接好的.csv` (2793 timepoints × 44 features).

## Hyperparameters (current defaults in train_iterative.py)

latent_dim=32, hidden_units=128, dropout=0.3, l2_reg=0.0005, p_drop=0.5, p_consist=0.1, n_corrupted=3, lambda1=1.0, lambda2=1.0, lambda3=10.0, lr=0.001, batch_size=8, window_size=48, stride=12, k_spatial=5, k_temporal=5
