#!/usr/bin/env bash
set -euo pipefail

# Run arc_net region training then evaluation in terminal Python.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH."
  exit 1
fi

# Ensure `conda activate` works in non-interactive shells.
source "$(conda info --base)/etc/profile.d/conda.sh"

python train_regions_arc.py --src 'UBFC_my_in' -t 'PURE_my_in' -ui --weight_info 0
python eval_from_bvp.py
python train_regions_arc.py --src 'UBFC_my_in' -t 'PURE_my_in' -ui --weight_info 0.01
python eval_from_bvp.py
python train_regions_arc.py --src 'UBFC_my_rm' -t 'PURE_my_in'
python eval_from_bvp.py
python train_regions_arc.py --src 'UBFC_my_rm' -t 'PURE_my_rm'
python eval_from_bvp.py

python train_regions_arc.py --src 'BUAA_my_in' -t 'UBFC_my_in' -ui --weight_info 0
python eval_from_bvp.py
python train_regions_arc.py --src 'BUAA_my_in' -t 'UBFC_my_in' -ui --weight_info 0.01
python eval_from_bvp.py
python train_regions_arc.py --src 'BUAA_my_rm' -t 'UBFC_my_in'
python eval_from_bvp.py
python train_regions_arc.py --src 'BUAA_my_rm' -t 'UBFC_my_rm'
python eval_from_bvp.py