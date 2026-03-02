# %%
import cv2
import os
import numpy as np
import shutil
import pandas as pd
import scipy.io as scio
from scipy import interpolate
import scipy.io as io
import glob

import config

# -------------------- Config (edit config.py for shared settings) --------------------
TGT_DOMAIN = config.TGT_DOMAIN
SRC_DOMAIN = config.SRC_DOMAIN
SPATIAL_AUG_RATE = config.SPATIAL_AUG_RATE
TEMPORAL_AUG_RATE = config.TEMPORAL_AUG_RATE
LOSS_TYPE = config.LOSS_TYPE
RUN_NAME_OVERRIDE = None  # override run name, or None to use config defaults

INDEX_DIR = config.get_index_dir(TGT_DOMAIN)
savePath = os.path.join(config.WAVE_SORT_ROOT, TGT_DOMAIN)
os.makedirs(savePath, exist_ok=True)
RESULT_DIR = config.RESULT_DIR


def build_run_name():
    return config.build_run_name(override=RUN_NAME_OVERRIDE)


def resolve_result_paths(run_name: str):
    """Return (gt_path, pr_path). If exact files missing, try glob fallback."""
    gt_path = os.path.join(RESULT_DIR, run_name + 'WAVE_ALL.mat')
    pr_path = os.path.join(RESULT_DIR, run_name + 'WAVE_PR_ALL.mat')
    if os.path.isfile(gt_path) and os.path.isfile(pr_path):
        return gt_path, pr_path

    # Fallback: find closest matches (e.g. if loss_type changed)
    pr_matches = sorted(glob.glob(os.path.join(RESULT_DIR, f"rPPGNet_{TGT_DOMAIN}_src{SRC_DOMAIN}*WAVE_PR_ALL.mat")))
    gt_matches = sorted(glob.glob(os.path.join(RESULT_DIR, f"rPPGNet_{TGT_DOMAIN}_src{SRC_DOMAIN}*WAVE_ALL.mat")))
    if pr_matches and gt_matches:
        # pick the newest by mtime
        pr_best = max(pr_matches, key=os.path.getmtime)
        gt_best = max(gt_matches, key=os.path.getmtime)
        print("Warning: exact run files not found; using newest matches:")
        print("  PR:", os.path.basename(pr_best))
        print("  GT:", os.path.basename(gt_best))
        return gt_best, pr_best

    raise FileNotFoundError(
        "Could not find result wave files. Expected:\n"
        f"  {pr_path}\n"
        f"  {gt_path}\n"
        "Available Result files:\n"
        + "\n".join("  " + os.path.basename(p) for p in sorted(glob.glob(os.path.join(RESULT_DIR, "*.mat"))))
    )


run_name = build_run_name()
gt_path, pr_path = resolve_result_paths(run_name)

print("Index dir:", INDEX_DIR)
print("Run name:", run_name)
print("GT wave:", gt_path)
print("PR wave:", pr_path)

pr = scio.loadmat(pr_path)['Wave']
pr = np.squeeze(np.array(pr.astype('float32')))
gt = scio.loadmat(gt_path)['Wave']
gt = np.squeeze(np.array(gt.astype('float32')))

# %%
files_list = os.listdir(INDEX_DIR)
files_list = sorted(files_list)
num_index = len(files_list)
num_samples = pr.shape[0]
if num_index != num_samples:
    print('Warning: index file count (%d) != result sample count (%d). Using min.' % (num_index, num_samples))
n_use = min(num_index, num_samples)
print('Using %d samples (index files: %d, result rows: %d)' % (n_use, num_index, num_samples))
print('Output dir: %s' % savePath)
temp = scio.loadmat(os.path.join(INDEX_DIR, files_list[0]))
lastPath = str(temp['Path'][0])
lastSubject = os.path.basename(os.path.normpath(lastPath))  # Extract subject folder name (e.g., "01-01")
pr_temp = []
gt_temp = []
# %%
# Loop over each sample (row index in pr/gt). Index files are in same order as pr/gt rows.
for HR_index in range(n_use):
    temp = scio.loadmat(os.path.join(INDEX_DIR, files_list[HR_index]))
    nowPath = str(temp['Path'][0])
    nowSubject = os.path.basename(os.path.normpath(nowPath))  # Extract subject folder name
    Step_Index = int(np.asarray(temp['Step_Index']).flat[0])

    if lastPath != nowPath:
        # New subject detected
        if pr_temp is None:
            print('[First subject] Subject: %s (Path: %s)' % (nowSubject, nowPath))
            pr_temp = []
            gt_temp = []
        else:
            # Save previous subject's waves using folder name
            n_seg = len(pr_temp)
            print('  -> Saved subject %s: %d segments to %spr_Wave.mat, %sgt_Wave.mat' % (lastSubject, n_seg, lastSubject, lastSubject))
            io.savemat(os.path.join(savePath, lastSubject + 'pr_Wave.mat'), {'Wave': pr_temp})
            io.savemat(os.path.join(savePath, lastSubject + 'gt_Wave.mat'), {'Wave': gt_temp})
            pr_temp = []
            gt_temp = []
        pr_temp.append(pr[HR_index, :])
        gt_temp.append(gt[HR_index, :])
    else:
        # Same subject: append to buffers
        pr_temp.append(pr[HR_index, :])
        gt_temp.append(gt[HR_index, :])
    lastPath = nowPath
    lastSubject = nowSubject

# Save the last subject's waves
if pr_temp:
    n_seg = len(pr_temp)
    print('  -> Saved subject %s: %d segments to %spr_Wave.mat, %sgt_Wave.mat' % (lastSubject, n_seg, lastSubject, lastSubject))
    io.savemat(os.path.join(savePath, lastSubject + 'pr_Wave.mat'), {'Wave': np.array(pr_temp)})
    io.savemat(os.path.join(savePath, lastSubject + 'gt_Wave.mat'), {'Wave': np.array(gt_temp)})
print('Done. Per-subject waves written to %s' % savePath)

# %%
