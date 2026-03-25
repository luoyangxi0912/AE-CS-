#!/bin/bash
# ============================================================
# AE-CS V11 Cloud One-Click Script
# - Install deps
# - Train with full logs
# - Auto evaluate
# - Package artifacts for download
# ============================================================

set -eu

CHECKPOINT_DIR="checkpoints_v11"
RESULTS_DIR="results/eval_v11"
ARTIFACT_ROOT="artifacts"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ARTIFACT_ROOT}/run_${RUN_TS}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${CHECKPOINT_DIR}" "${RUN_DIR}"

TRAIN_LOG="${LOG_DIR}/train.log"
EVAL_LOG="${LOG_DIR}/eval.log"
ENV_LOG="${LOG_DIR}/env.txt"
PIP_FREEZE_LOG="${LOG_DIR}/pip_freeze.txt"
PACKAGE_PATH="${RUN_DIR}/ae_cs_v11_${RUN_TS}.tar.gz"

echo "============================================================"
echo "AE-CS V11 Cloud Pipeline"
echo "Run dir: ${RUN_DIR}"
echo "============================================================"

echo ""
echo "[1/5] Install Python dependencies..."
pip install tensorflow==2.10.0 numpy pandas scikit-learn -q
pip install faiss-cpu -q
echo "  Done."

echo ""
echo "[2/5] Collect environment info..."
{
  echo "Timestamp: ${RUN_TS}"
  echo "Working dir: $(pwd)"
  echo "Python: $(python3 --version 2>&1)"
  echo "Pip: $(pip --version 2>&1)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
} | tee "${ENV_LOG}"
pip freeze > "${PIP_FREEZE_LOG}" || true

echo ""
echo "[3/5] Check GPU..."
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  Detected {len(gpus)} GPU(s):')
    for gpu in gpus:
        print(f'    - {gpu}')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('  No GPU detected, training will run on CPU.')
" | tee -a "${ENV_LOG}"

echo ""
echo "[4/5] Train model (log: ${TRAIN_LOG})..."
python3 train_cloud.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --epochs 100 \
    --batch_size 32 \
    --early_stopping_patience 20 \
    2>&1 | tee "${TRAIN_LOG}"

echo ""
echo "[5/5] Evaluate model (log: ${EVAL_LOG})..."
python3 evaluate.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee "${EVAL_LOG}"

echo ""
echo "Packaging artifacts..."
tar -czf "${PACKAGE_PATH}" \
  "${CHECKPOINT_DIR}" \
  "${RESULTS_DIR}" \
  "${LOG_DIR}" || true

echo ""
echo "============================================================"
echo "Pipeline finished."
echo "Logs:"
echo "  - ${TRAIN_LOG}"
echo "  - ${EVAL_LOG}"
echo "Results:"
echo "  - ${RESULTS_DIR}/metrics.json"
echo "  - ${RESULTS_DIR}/feature_performance.csv"
echo "Artifact package:"
echo "  - ${PACKAGE_PATH}"
echo "============================================================"
