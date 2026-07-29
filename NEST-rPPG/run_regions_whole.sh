#!/usr/bin/env bash
set -euo pipefail

# Train on whole-face STMaps (*_my), infer on infraorbital region (*_my_in).
# Alignment controlled by --weight_info only (0 = off). Uses training conda env for training and eval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH."
  exit 1
fi

# Ensure `conda activate` works in non-interactive shells.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpipe

# # ============================================
# # Training: PURE_my -> Testing: UBFC_my_in
# # ============================================
# conda activate training; python train_whole.py --src 'PURE_my' -t 'UBFC_my_in' --weight_info 0.01
# conda activate training; python eval_from_bvp.py

# # ============================================
# # Training: UBFC_my -> Testing: PURE_my_in
# # ============================================
# conda activate training; python train_whole.py --src 'UBFC_my' -t 'PURE_my_in' --weight_info 0.01
# conda activate training; python eval_from_bvp.py

# # ============================================
# # Training: BUAA_my -> Testing: UBFC_my_in
# # ============================================
# conda activate training; python train_whole.py --src 'BUAA_my' -t 'UBFC_my_in' --weight_info 0.01
# conda activate training; python eval_from_bvp.py

# ============================================
# Training: UBFC_my -> Testing: BUAA_my_in
# ============================================
# conda activate training; python train_whole.py --src 'UBFC_my' -t 'BUAA_my_in' --weight_info 0.01
# conda activate training; python eval_from_bvp.py

# # ============================================
# # Training: BUAA_my -> Testing: PURE_my_in
# # ============================================
# conda activate training; python train_whole.py --src 'BUAA_my' -t 'PURE_my_in' --weight_info 0.01
# conda activate training; python eval_from_bvp.py

# # ============================================
# # Training: PURE_my -> Testing: BUAA_my_in
# # ============================================
# conda activate training; python train_whole.py --src 'PURE_my' -t 'BUAA_my_in' --weight_info 0.01
# conda activate training; python eval_from_bvp.py

# ============================================
# weight_info=0 / 0.01 variants
# ============================================

python train_whole.py --src 'UBFC_my' -t 'PURE_my_in' --weight_info 0
python eval_from_bvp.py


python train_whole.py --src 'BUAA_my' -t 'UBFC_my_in' --weight_info 0
python eval_from_bvp.py


python train_whole.py --src 'UBFC_my' -t 'BUAA_my_in' --weight_info 0
python eval_from_bvp.py


python train_whole.py --src 'BUAA_my' -t 'PURE_my_in' --weight_info 0
python eval_from_bvp.py

python train_whole.py --src 'PURE_my' -t 'BUAA_my_in' --weight_info 0
python eval_from_bvp.py

python train_whole.py --src 'UBFC_my' -t 'BUAA_my_in' --weight_info 0
python eval_from_bvp.py



