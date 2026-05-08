#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Intratest 5-fold CV with temporal HR augmentation on all datasets.
# hr_aug_max=2.0  : compress up to 2× more source frames → simulate up to 2× HR
# hr_aug_prob=0.5 : apply augmentation to 50% of training samples per epoch
#
# PURE: participant 07 (tachycardia, ~130 BPM) is excluded from augmentation
# since further compression would push their labels out of valid range.
# UBFC/BUAA: no exclusions needed (no known HR outliers).

python train_intratest_cv_aug.py --dataset PURE_my_in --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_prob 0.5 --hr_aug_exclude 07
python train_intratest_cv_aug.py --dataset UBFC_my_in --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_prob 0.5
python train_intratest_cv_aug.py --dataset BUAA_my_in --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_prob 0.5
