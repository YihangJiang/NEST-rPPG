#!/usr/bin/env bash
set -euo pipefail

# Optuna fine-tuning for train_regions.py (tau_info, weight_info; loss_type=One from config).
# Requires: pip install optuna

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpipe

# Stop Jupyter kernels / other GPU jobs before running (OOM if GPU is ~full).
# Example: malar region pair (edit src/tgt/regions/n-trials as needed)
python optuna_tune_regions.py \
  --src 'BUAA_my_in' \
  -t 'PURE_my_in' \
  --regions all \
  --n-trials 20 \
  --seed 0 \
  --fresh

# Other malar pairs:
# python optuna_tune_regions.py --src 'PURE_my_in'  -t 'BUAA_my_in' --regions all --n-trials 20
# python optuna_tune_regions.py --src 'UBFC_my_in'  -t 'PURE_my_in' --regions all --n-trials 20
# python optuna_tune_regions.py --src 'UBFC_my_in'  -t 'BUAA_my_in' --regions all --n-trials 20
# python optuna_tune_regions.py --src 'BUAA_my_in'  -t 'PURE_my_in' --regions all --n-trials 20
# python optuna_tune_regions.py --src 'BUAA_my_in'  -t 'UBFC_my_in' --regions all --n-trials 20

# Eye region pairs use --regions neg and *_my_eye domains.
