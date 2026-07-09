"""
Baseline eval: always predict the median ground-truth HR from segment_errors.csv.
Reports MAE (and other metrics) for the "always guess the median" baseline.
"""
# %%

import os

import numpy as np
import pandas as pd

from utils.eval_utils import my_eval

csv_path = (
    "Wave_sort/UBFC_my_in/rPPGNet_UBFC_my_in_srcBUAA_my_uiFalse/feature/segment_errors.csv"
)

# %%
# Load segment errors
df = pd.read_csv(csv_path)
hr_gt = df["hr_gt_bpm"].to_numpy(dtype=float)
valid = np.isfinite(hr_gt)
hr_gt = hr_gt[valid]
if hr_gt.size == 0:
    raise ValueError(f"No valid hr_gt_bpm values in {csv_path}")

median_hr = float(np.median(hr_gt))
print(f"CSV: {os.path.abspath(csv_path)}")
print(f"Median GT HR (constant prediction): {median_hr:.6f} bpm")
print(f"Segments: {hr_gt.size}")

# %%
# Segment-level: predict median for every segment
hr_pr = np.full_like(hr_gt, median_hr)
metrics = my_eval(hr_pr, hr_gt)

print("Segment-level (all rows in CSV)")
print(f"  ME:       {metrics['ME']:.6f}")
print(f"  Std:      {metrics['Std']:.6f}")
print(f"  MAE:      {metrics['MAE']:.6f}")
print(f"  MAE_Std:  {metrics['MAE_Std']:.6f}")
print(f"  MAE_SE:   {metrics['MAE_SE']:.6f}")
print(f"  RMSE:     {metrics['RMSE']:.6f}")
print(f"  RMSE_Std: {metrics['RMSE_Std']:.6f}")
print(f"  RMSE_SE:  {metrics['RMSE_SE']:.6f}")

# %%
# Subject-level: same protocol as eval_from_bvp.py (mean HR per subject)
subj_gt = df.loc[valid].groupby("subject_id")["hr_gt_bpm"].mean().to_numpy(dtype=float)
subj_pr = np.full_like(subj_gt, median_hr)
subj_metrics = my_eval(subj_pr, subj_gt)

print(f"Subjects: {subj_gt.size}")
print("Subject-level (mean HR per subject)")
print(f"  ME:       {subj_metrics['ME']:.6f}")
print(f"  Std:      {subj_metrics['Std']:.6f}")
print(f"  MAE:      {subj_metrics['MAE']:.6f}")
print(f"  MAE_Std:  {subj_metrics['MAE_Std']:.6f}")
print(f"  MAE_SE:   {subj_metrics['MAE_SE']:.6f}")
print(f"  RMSE:     {subj_metrics['RMSE']:.6f}")
print(f"  RMSE_Std: {subj_metrics['RMSE_Std']:.6f}")
print(f"  RMSE_SE:  {subj_metrics['RMSE_SE']:.6f}")

# %%
