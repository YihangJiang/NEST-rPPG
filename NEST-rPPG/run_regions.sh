#!/usr/bin/env bash
set -euo pipefail

# Run region training (rppg env) then evaluation (mprppg env) in terminal Python.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH."
  exit 1
fi

# Ensure `conda activate` works in non-interactive shells.
source "$(conda info --base)/etc/profile.d/conda.sh"

# conda activate rppg; python train_regions.py --src 'PURE_my_in' -t 'UBFC_my_in' -ui --weight_info 0
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'PURE_my_in' -t 'UBFC_my_in' -ui --weight_info 0.01
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'PURE_my_rm' -t 'UBFC_my_in'
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'PURE_my_rm' -t 'UBFC_my_rm'
# conda activate mprppg; python eval_from_bvp.py

conda activate rppg; python train_regions.py --src 'UBFC_my_in' -t 'PURE_my_in' -ui --weight_info 0
conda activate mprppg; python eval_from_bvp.py
conda activate rppg; python train_regions.py --src 'UBFC_my_in' -t 'PURE_my_in' -ui --weight_info 0.01
conda activate mprppg; python eval_from_bvp.py
conda activate rppg; python train_regions.py --src 'UBFC_my_rm' -t 'PURE_my_in'
conda activate mprppg; python eval_from_bvp.py
conda activate rppg; python train_regions.py --src 'UBFC_my_rm' -t 'PURE_my_rm'
conda activate mprppg; python eval_from_bvp.py

conda activate rppg; python train_regions.py --src 'BUAA_my_in' -t 'UBFC_my_in' -ui --weight_info 0
conda activate mprppg; python eval_from_bvp.py
conda activate rppg; python train_regions.py --src 'BUAA_my_in' -t 'UBFC_my_in' -ui --weight_info 0.01
conda activate mprppg; python eval_from_bvp.py
conda activate rppg; python train_regions.py --src 'BUAA_my_rm' -t 'UBFC_my_in'
conda activate mprppg; python eval_from_bvp.py
conda activate rppg; python train_regions.py --src 'BUAA_my_rm' -t 'UBFC_my_rm'
conda activate mprppg; python eval_from_bvp.py

# conda activate rppg; python train_regions.py --src 'UBFC_my_in' -t 'BUAA_my_in' -ui --weight_info 0
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'UBFC_my_in' -t 'BUAA_my_in' -ui --weight_info 0.01
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'UBFC_my_rm' -t 'BUAA_my_in'
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'UBFC_my_rm' -t 'BUAA_my_rm'
# conda activate mprppg; python eval_from_bvp.py

# conda activate rppg; python train_regions.py --src 'BUAA_my_in' -t 'PURE_my_in' -ui --weight_info 0
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'BUAA_my_in' -t 'PURE_my_in' -ui --weight_info 0.01
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'BUAA_my_rm' -t 'PURE_my_in'
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'BUAA_my_rm' -t 'PURE_my_rm'
# conda activate mprppg; python eval_from_bvp.py

# conda activate rppg; python train_regions.py --src 'PURE_my_in' -t 'BUAA_my_in' -ui --weight_info 0
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'PURE_my_in' -t 'BUAA_my_in' -ui --weight_info 0.01
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'PURE_my_rm' -t 'BUAA_my_in'
# conda activate mprppg; python eval_from_bvp.py
# conda activate rppg; python train_regions.py --src 'PURE_my_rm' -t 'BUAA_my_rm'
# conda activate mprppg; python eval_from_bvp.py

