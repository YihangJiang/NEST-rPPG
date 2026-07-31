#!/usr/bin/env bash
set -euo pipefail

# Inter-Dataset (Cross-Domain) Region Training with Baseline & HR Temporal Resampling Data Augmentation.
# Evaluates cross-dataset generalization (e.g. PURE -> UBFC, PURE -> BUAA) with and without HR augmentation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpipe

WEIGHT_INFO=${1:-0.01}
TAU_INFO=${2:-0.05}
MAX_ITER=${3:-1000}

echo "=========================================================="
echo "  Inter-Dataset Cross-Domain Training Pipeline"
echo "  weight_info: $WEIGHT_INFO | tau_info: $TAU_INFO | max_iter: $MAX_ITER"
echo "=========================================================="

# STEP 0: Baseline (no HR temporal resampling augmentation)
echo ""
echo "--- STEP 0: Baseline (no HR augmentation) ---"

python train_regions.py \
  --src 'PURE_my_in' \
  -t 'UBFC_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 1.0 \
  --run_tag baseline

python train_regions.py \
  --src 'PURE_my_in' \
  -t 'BUAA_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 1.0 \
  --run_tag baseline

python train_regions.py \
  --src 'UBFC_my_in' \
  -t 'PURE_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 1.0 \
  --run_tag baseline

# STEP 1: High-end HR Augmentation (compress STMap -> simulate faster HR up to 2.0x)
echo ""
echo "--- STEP 1: High-end HR Augmentation (hr_aug_max 2.0) ---"

python train_regions.py \
  --src 'PURE_my_in' \
  -t 'UBFC_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 2.0 \
  --run_tag aug

python train_regions.py \
  --src 'PURE_my_in' \
  -t 'BUAA_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 2.0 \
  --run_tag aug

python train_regions.py \
  --src 'UBFC_my_in' \
  -t 'PURE_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 2.0 \
  --run_tag aug

# STEP 2: Both-ends HR Augmentation (fast HR up to 2.0x + slow HR down to 0.5x)
echo ""
echo "--- STEP 2: Both-ends HR Augmentation (hr_aug_max 2.0, hr_aug_min 0.5) ---"

python train_regions.py \
  --src 'PURE_my_in' \
  -t 'UBFC_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 2.0 \
  --hr_aug_min 0.5 \
  --run_tag aug_both

python train_regions.py \
  --src 'PURE_my_in' \
  -t 'BUAA_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 2.0 \
  --hr_aug_min 0.5 \
  --run_tag aug_both

echo ""
echo "Inter-dataset HR augmentation training pipeline completed."
