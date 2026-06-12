#!/usr/bin/env python3
# %%
"""
Analyze ALL experiment snapshots (Jupyter cell style).

Input is limited to waveform .mat files under:
  NEST-rPPG/Training_Log/experiments/<experiment>/wave_sort/*.mat

For each experiment folder, recompute metrics using the same `run_eval()` used by eval_from_bvp.py.
"""

# %%
import json
import os

from utils.eval_utils import run_eval

# %%
# ----- Hard-coded root -----
EXPERIMENTS_ROOT = "/home/yj167/Desktop/NEST-rPPG/NEST-rPPG/Training_Log/experiments"
# ---------------------------

# %%
if not os.path.isdir(EXPERIMENTS_ROOT):
    raise FileNotFoundError(f"Experiments root not found: {EXPERIMENTS_ROOT}")

exp_names = sorted(
    d for d in os.listdir(EXPERIMENTS_ROOT)
    if os.path.isdir(os.path.join(EXPERIMENTS_ROOT, d))
)

if not exp_names:
    raise RuntimeError(f"No experiment folders found under: {EXPERIMENTS_ROOT}")

print(f"Found {len(exp_names)} experiments.")

# %%
# Recompute each experiment using the SAME function used by eval_from_bvp.py
all_results = {}

for i, exp_name in enumerate(exp_names, start=1):
    exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
    wave_sort_dir = os.path.join(exp_dir, "wave_sort")
    if not os.path.isdir(wave_sort_dir):
        print(f"[{i}/{len(exp_names)}] {exp_name}: skip (missing wave_sort/)")
        continue

    print(f"\n[{i}/{len(exp_names)}] {exp_name}")
    result = run_eval(wave_sort_dir, return_details=False)
    all_results[exp_name] = result
    hr = result.get("HR", {})
    print(
        "  HR -> "
        f"ME={hr.get('ME', float('nan')):.6f}, "
        f"Std={hr.get('Std', float('nan')):.6f}, "
        f"MAE={hr.get('MAE', float('nan')):.6f}, "
        f"MAE_std={hr.get('MAE_std', float('nan')):.6f}, "
        f"RMSE={hr.get('RMSE', float('nan')):.6f}, "
        f"RMSE_std={hr.get('RMSE_std', float('nan')):.6f}, "
        f"MER={hr.get('MER', float('nan')):.6f}, "
        f"r={hr.get('r', float('nan')):.6f}"
    )

print("\n=== All experiment results (JSON) ===")
print(json.dumps(all_results, indent=2))



# %%
