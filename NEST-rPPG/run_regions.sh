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

echo "[INFO] Activating conda env: rppg"
conda activate rppg

echo "[INFO] Running train_regions.py"
python train_regions.py --src 'UBFC_my_rm' -t 'BUAA_my_in'

echo "[INFO] Activating conda env: mprppg"
conda activate mprppg

echo "[INFO] Running eval_from_bvp.py"
python eval_from_bvp.py

echo "[INFO] Done."
