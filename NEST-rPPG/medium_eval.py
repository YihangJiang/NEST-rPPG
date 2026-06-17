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
me, std, mae, rmse, mer, r = my_eval(hr_pr, hr_gt)

print("Segment-level (all rows in CSV)")
print(f"  ME:   {me:.6f}")
print(f"  Std:  {std:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  MER:  {mer:.6f}")
print(f"  r:    {r:.6f}")

# %%
# Subject-level: same protocol as eval_from_bvp.py (mean HR per subject)
subj_gt = df.loc[valid].groupby("subject_id")["hr_gt_bpm"].mean().to_numpy(dtype=float)
subj_pr = np.full_like(subj_gt, median_hr)
subj_me, subj_std, subj_mae, subj_rmse, subj_mer, subj_r = my_eval(subj_pr, subj_gt)

print(f"Subjects: {subj_gt.size}")
print("Subject-level (mean HR per subject)")
print(f"  ME:   {subj_me:.6f}")
print(f"  Std:  {subj_std:.6f}")
print(f"  MAE:  {subj_mae:.6f}")
print(f"  RMSE: {subj_rmse:.6f}")
print(f"  MER:  {subj_mer:.6f}")
print(f"  r:    {subj_r:.6f}")

# %%
