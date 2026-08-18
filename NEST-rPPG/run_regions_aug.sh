#!/usr/bin/env bash
# ==============================================================================
# Inter-dataset Cross-Domain Baseline & HR Augmentation Pipeline
# Baseline: No augmentation (hr_aug_max 1.0)
# Step 1: High-end HR Augmentation (hr_aug_max 2.0)
# Step 2: Both-ends HR Augmentation (hr_aug_max 2.0, hr_aug_min 0.5)
# ==============================================================================

set -e

WEIGHT_INFO=0.01
TAU_INFO=0.05
MAX_ITER=1000

echo "Starting Inter-Dataset HR Augmentation Pipeline..."
echo "  weight_info: $WEIGHT_INFO"
echo "  tau_info:    $TAU_INFO"
echo "  max_iter:    $MAX_ITER"

# ------------------------------------------------------------------------------
# STEP 0: Baseline (No HR Augmentation)
# ------------------------------------------------------------------------------
echo ""
echo "--- STEP 0: Baseline Without HR Augmentation (hr_aug_max 1.0) ---"

python train_aug.py \
  --src 'PURE_my_in' \
  -t 'UBFC_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 1.0 \
  --run_tag baseline

python train_aug.py \
  --src 'PURE_my_in' \
  -t 'BUAA_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 1.0 \
  --run_tag baseline

python train_aug.py \
  --src 'UBFC_my_in' \
  -t 'PURE_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 1.0 \
  --run_tag baseline

# ------------------------------------------------------------------------------
# STEP 1: High-end HR Augmentation (compress STMap -> simulate faster HR up to 2.0x)
# ------------------------------------------------------------------------------
echo ""
echo "--- STEP 1: High-end HR Augmentation (hr_aug_max 2.0) ---"

python train_aug.py \
  --src 'PURE_my_in' \
  -t 'UBFC_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 2.0 \
  --run_tag aug

python train_aug.py \
  --src 'PURE_my_in' \
  -t 'BUAA_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 2.0 \
  --run_tag aug

python train_aug.py \
  --src 'UBFC_my_in' \
  -t 'PURE_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 2.0 \
  --run_tag aug

# ------------------------------------------------------------------------------
# STEP 2: Both-ends HR Augmentation (fast HR up to 2.0x + slow HR down to 0.5x)
# ------------------------------------------------------------------------------
echo ""
echo "--- STEP 2: Both-ends HR Augmentation (hr_aug_max 2.0, hr_aug_min 0.5) ---"

python train_aug.py \
  --src 'PURE_my_in' \
  -t 'UBFC_my_in' \
  --regions all \
  --weight_info "$WEIGHT_INFO" \
  --tau_info "$TAU_INFO" \
  --max_iter "$MAX_ITER" \
  --hr_aug_max 2.0 \
  --hr_aug_min 0.5 \
  --run_tag aug_both

python train_aug.py \
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
