#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Intratest 5-fold cross-validation using BaseNet (within-dataset train/test)
# Running without contrastive losses (weight_cl=0) to check stability first.

python train_intratest_cv.py --dataset PURE_my_in --folds 5 --max_iter 1000 --weight_cl 0
python train_intratest_cv.py --dataset UBFC_my_in --folds 5 --max_iter 1000 --weight_cl 0
python train_intratest_cv.py --dataset BUAA_my_in --folds 5 --max_iter 1000 --weight_cl 0
