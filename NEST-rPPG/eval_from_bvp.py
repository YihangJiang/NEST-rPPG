"""
Evaluates BVP model performance from Wave_sort .mat files (gt/pr pairs):
heart rate only, from FFT on raw segments. Reports ME, Std, MAE, RMSE, MER, Pearson r.
When run this script, use interpreter: mprppg
"""
# %%

import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.signal import welch

import config
from utils.eval_utils import (
    FS_BVP,
    hr_from_fft,
    my_eval,
    run_eval,
    visualize_mat_waves,
    estimate_hr_from_psd,
)

save_path = config.EVAL_SAVE_PATH
print(f"Evaluating Wave_sort path: {save_path}")

# %%
# Path to Wave_sort directory (gt/pr .mat pairs). Set Option A or B in config.EVAL_SAVE_PATH.
# Name for first-level visualization subfolder under save_path/vis/.
# Final structure: Wave_sort/<TGT_DOMAIN>/vis/<VIS_RUN_NAME>/<subject_id>/

# %%
# Optional: visualize waves before evaluation (example usage)
pairs = visualize_mat_waves(
    save_path,
    segment_indices=[3, 5],
    vis_run_name=config.LOSS_TYPE,
)
signal = pairs[5]["pred"]
hr_bpm = estimate_hr_from_psd(signal, fs=FS_BVP, f_low=0.7, f_high=4.0)
print(f"Estimated HR (Welch PSD): {hr_bpm:.2f} bpm")

# %%
result = run_eval(save_path)

# Print table: ME, Std, MAE, RMSE, MER, Pearson r
print("Feature    \tME\t\tStd\t\tMAE\t\tRMSE\t\tMER\t\tr")
print("-" * 90)
for name, metrics in result.items():
    print(
        f"{name:10}\t{metrics['ME']:.6f}\t{metrics['Std']:.6f}\t{metrics['MAE']:.6f}\t"
        f"{metrics['RMSE']:.6f}\t{metrics['MER']:.6f}\t{metrics['r']:.6f}"
    )

# %%
