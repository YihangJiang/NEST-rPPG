#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="Training_Log"

# Run a single dataset if its output CSV doesn't already exist.
# Usage: run_if_missing <csv_path> <python_script> [args...]
run_if_missing() {
    local csv="$1"; shift
    if [[ -f "$csv" ]]; then
        echo "  [SKIP] Already done: $csv"
    else
        echo "  [RUN ] $*"
        python "$@"
    fi
}

echo "========================================================"
echo "  STEP 1: Baseline (no augmentation)"
echo "========================================================"
run_if_missing "$LOG_DIR/intratest_cv_PURE_my_in_wl0.0_results.csv" \
    train_intratest_cv.py --dataset PURE_my_in --folds 5 --max_iter 1000 --weight_cl 0

run_if_missing "$LOG_DIR/intratest_cv_UBFC_my_in_wl0.0_results.csv" \
    train_intratest_cv.py --dataset UBFC_my_in --folds 5 --max_iter 1000 --weight_cl 0

run_if_missing "$LOG_DIR/intratest_cv_BUAA_my_in_wl0.0_results.csv" \
    train_intratest_cv.py --dataset BUAA_my_in --folds 5 --max_iter 1000 --weight_cl 0

echo ""
echo "========================================================"
echo "  STEP 2: High-end HR augmentation (fast HR only, r up to 2.0)"
echo "========================================================"
run_if_missing "$LOG_DIR/intratest_cv_PURE_my_in_wl0.0_aug_results.csv" \
    train_intratest_cv_aug.py --dataset PURE_my_in --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_prob 0.5 --run_tag aug

run_if_missing "$LOG_DIR/intratest_cv_UBFC_my_in_wl0.0_aug_results.csv" \
    train_intratest_cv_aug.py --dataset UBFC_my_in --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_prob 0.5 --run_tag aug

run_if_missing "$LOG_DIR/intratest_cv_BUAA_my_in_wl0.0_aug_results.csv" \
    train_intratest_cv_aug.py --dataset BUAA_my_in --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_prob 0.5 --run_tag aug

echo ""
echo "========================================================"
echo "  STEP 3: Both-ends HR augmentation (fast r up to 2.0 + slow r down to 0.5)"
echo "========================================================"
run_if_missing "$LOG_DIR/intratest_cv_PURE_my_in_wl0.0_aug_both_results.csv" \
    train_intratest_cv_aug.py --dataset PURE_my_in --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_min 0.5 --hr_aug_prob 0.5 --run_tag aug_both

run_if_missing "$LOG_DIR/intratest_cv_UBFC_my_in_wl0.0_aug_both_results.csv" \
    train_intratest_cv_aug.py --dataset UBFC_my_in --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_min 0.5 --hr_aug_prob 0.5 --run_tag aug_both

run_if_missing "$LOG_DIR/intratest_cv_BUAA_my_in_wl0.0_aug_both_results.csv" \
    train_intratest_cv_aug.py --dataset BUAA_my_in --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_min 0.5 --hr_aug_prob 0.5 --run_tag aug_both

echo ""
echo "========================================================"
echo "  STEP 4: BUAA 100 lux only (baseline + augmentation)"
echo "========================================================"
run_if_missing "$LOG_DIR/intratest_cv_BUAA_my_in_100lux_wl0.0_results.csv" \
    train_intratest_cv.py --dataset BUAA_my_in_100lux --folds 5 --max_iter 1000 --weight_cl 0 \
    --session_filter "100.0"

run_if_missing "$LOG_DIR/intratest_cv_BUAA_my_in_100lux_wl0.0_aug_results.csv" \
    train_intratest_cv_aug.py --dataset BUAA_my_in_100lux --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_prob 0.5 --run_tag aug --session_filter "100.0"

run_if_missing "$LOG_DIR/intratest_cv_BUAA_my_in_100lux_wl0.0_aug_both_results.csv" \
    train_intratest_cv_aug.py --dataset BUAA_my_in_100lux --folds 5 --max_iter 1000 --weight_cl 0 \
    --hr_aug_max 2.0 --hr_aug_min 0.5 --hr_aug_prob 0.5 --run_tag aug_both --session_filter "100.0"

echo ""
echo "========================================================"
echo "  STEP 5: Comparison"
echo "========================================================"
echo ""
echo "--- Baseline vs High-end aug (fast HR only) ---"
python compare_aug_results.py --tag aug

echo ""
echo "--- Baseline vs Both-ends aug (fast + slow HR) ---"
python compare_aug_results.py --tag aug_both

echo ""
echo "--- BUAA 100lux: Baseline vs High-end aug ---"
python compare_aug_results.py --tag aug --datasets BUAA_my_in_100lux

echo ""
echo "--- BUAA 100lux: Baseline vs Both-ends aug ---"
python compare_aug_results.py --tag aug_both --datasets BUAA_my_in_100lux

echo ""
echo "All done."
