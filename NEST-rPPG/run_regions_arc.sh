#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python train_regions_arc.py --src 'PURE_my_in' -t 'UBFC_my_in' -ui --weight_info 0
python eval_from_bvp.py

python train_regions_arc.py --src 'PURE_my_in' -t 'UBFC_my_in' -ui --weight_info 0.01
python eval_from_bvp.py

python train_regions_arc.py --src 'UBFC_my_in' -t 'PURE_my_in' -ui --weight_info 0
python eval_from_bvp.py

python train_regions_arc.py --src 'UBFC_my_in' -t 'PURE_my_in' -ui --weight_info 0.01
python eval_from_bvp.py

python train_regions_arc.py --src 'BUAA_my_in' -t 'UBFC_my_in' -ui --weight_info 0
python eval_from_bvp.py

python train_regions_arc.py --src 'BUAA_my_in' -t 'UBFC_my_in' -ui --weight_info 0.01
python eval_from_bvp.py

python train_regions_arc.py --src 'UBFC_my_in' -t 'BUAA_my_in' -ui --weight_info 0
python eval_from_bvp.py

python train_regions_arc.py --src 'UBFC_my_in' -t 'BUAA_my_in' -ui --weight_info 0.01
python eval_from_bvp.py

python train_regions_arc.py --src 'BUAA_my_in' -t 'PURE_my_in' -ui --weight_info 0
python eval_from_bvp.py

python train_regions_arc.py --src 'BUAA_my_in' -t 'PURE_my_in' -ui --weight_info 0.01
python eval_from_bvp.py

python train_regions_arc.py --src 'PURE_my_in' -t 'BUAA_my_in' -ui --weight_info 0
python eval_from_bvp.py

python train_regions_arc.py --src 'PURE_my_in' -t 'BUAA_my_in' -ui --weight_info 0.01
python eval_from_bvp.py

python collect_results.py
