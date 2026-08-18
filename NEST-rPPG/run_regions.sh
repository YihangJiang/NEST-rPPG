#!/usr/bin/env bash
python train_regions.py --src 'PURE_my_in' -t 'UBFC_my_in' --regions all --tau_info 0.05 --weight_info 0.01
python eval_from_bvp.py