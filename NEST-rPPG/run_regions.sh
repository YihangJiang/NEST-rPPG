#!/usr/bin/env bash
set -euo pipefail

# Run arc_net region training then evaluation in terminal Python.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


python train_regions_arc.py --src 'PURE_my_in' -t 'UBFC_my_in' -ui --weight_info 0
python eval_from_bvp.py
