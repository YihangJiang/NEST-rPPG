#!/usr/bin/env bash
set -euo pipefail

# Run region training then evaluation in terminal Python (mpipe conda env, GPU).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpipe

# ============================================
# Compare tau=0.03: weight_info=0.01 vs weight_info=0
# 6 cross-dataset pairs (PURE, UBFC, BUAA — all directed pairs)
# ============================================
# python train_regions.py --src 'PURE_my_in' -t 'UBFC_my_in' --regions all --tau-info 0.05 --weight_info 0.01
# python eval_from_bvp.py
# python train_regions.py --src 'PURE_my_in' -t 'UBFC_my_in' --regions all --tau-info 0.05 --weight_info 0
# python eval_from_bvp.py

# python train_regions.py --src 'PURE_my_in' -t 'BUAA_my_in' --regions all --tau-info 0.05 --weight_info 0.01
# python eval_from_bvp.py
# python train_regions.py --src 'PURE_my_in' -t 'BUAA_my_in' --regions all --tau-info 0.05 --weight_info 0
# python eval_from_bvp.py

# python train_regions.py --src 'UBFC_my_in' -t 'PURE_my_in' --regions all --tau-info 0.05 --weight_info 0.01
# python eval_from_bvp.py
# python train_regions.py --src 'UBFC_my_in' -t 'PURE_my_in' --regions all --tau-info 0.05 --weight_info 0
# python eval_from_bvp.py

# python train_regions.py --src 'UBFC_my_in' -t 'BUAA_my_in' --regions all --tau-info 0.05 --weight_info 0.01
# python eval_from_bvp.py
# python train_regions.py --src 'UBFC_my_in' -t 'BUAA_my_in' --regions all --tau-info 0.05 --weight_info 0
# python eval_from_bvp.py

python train_regions.py --src 'BUAA_my_in' -t 'PURE_my_in' --regions all --tau-info 0.05 --weight_info 0.01
python eval_from_bvp.py
# python train_regions.py --src 'BUAA_my_in' -t 'PURE_my_in' --regions all --tau-info 0.05 --weight_info 0
# python eval_from_bvp.py

# python train_regions.py --src 'BUAA_my_in' -t 'UBFC_my_in' --regions all --tau-info 0.05 --weight_info 0.01
# python eval_from_bvp.py
# python train_regions.py --src 'BUAA_my_in' -t 'UBFC_my_in' --regions all --tau-info 0.05 --weight_info 0
# python eval_from_bvp.py


