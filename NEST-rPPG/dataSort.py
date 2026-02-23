# %%
import cv2
import os
import numpy as np
import shutil
import pandas as pd
import scipy.io as scio
from scipy import interpolate
import scipy.io as io

# Base path: directory containing this script (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

gt_name = 'BVP.mat'
savePath = os.path.join(BASE_DIR, 'Wave_sort', 'UBFC')
if not os.path.exists(savePath):
    os.makedirs(savePath)
Idex_files = os.path.join(BASE_DIR, 'STMap', 'STMap_Index', 'UBFC')
gt_path = os.path.join(BASE_DIR, 'Result', 'rPPGNet_UBFC_srcPURESpatial0.5Temporal0.1WAVE_ALL.mat')
pr_path = os.path.join(BASE_DIR, 'Result', 'rPPGNet_UBFC_srcPURESpatial0.5Temporal0.1WAVE_PR_ALL.mat')
pr = scio.loadmat(pr_path)['Wave']
pr = np.squeeze(np.array(pr.astype('float32')))
gt = scio.loadmat(gt_path)['Wave']
gt = np.squeeze(np.array(gt.astype('float32')))

# %%
files_list = os.listdir(Idex_files)
files_list = sorted(files_list)
num_index = len(files_list)
num_samples = pr.shape[0]
if num_index != num_samples:
    print('Warning: index file count (%d) != result sample count (%d). Using min.' % (num_index, num_samples))
n_use = min(num_index, num_samples)
print('Using %d samples (index files: %d, result rows: %d)' % (n_use, num_index, num_samples))
print('Output dir: %s' % savePath)
temp = scio.loadmat(os.path.join(Idex_files, files_list[0]))
lastPath = str(temp['Path'][0])
pr_temp = []
gt_temp = []
PERSON = 10000
# %%
# Loop over each sample (row index in pr/gt). Index files are in same order as pr/gt rows.
for HR_index in range(n_use):
    temp = scio.loadmat(os.path.join(Idex_files, files_list[HR_index]))
    nowPath = str(temp['Path'][0])
    Step_Index = int(np.asarray(temp['Step_Index']).flat[0])

    if lastPath != nowPath:
        PERSON = PERSON + 1
        if pr_temp is None:
            print('[First subject] Path: %s' % nowPath)
            pr_temp = []
            gt_temp = []
        else:
            n_seg = len(pr_temp)
            print('  -> Saved subject %s: %d segments to %spr_Wave.mat, %sgt_Wave.mat' % (PERSON, n_seg, PERSON, PERSON))
            io.savemat(os.path.join(savePath, str(PERSON) + 'pr_Wave.mat'), {'Wave': pr_temp})
            io.savemat(os.path.join(savePath, str(PERSON) + 'gt_Wave.mat'), {'Wave': gt_temp})
            pr_temp = []
            gt_temp = []
        pr_temp.append(pr[HR_index, :])
        gt_temp.append(gt[HR_index, :])
    else:
        pr_temp.append(pr[HR_index, :])
        gt_temp.append(gt[HR_index, :])
    lastPath = nowPath

# Save the last subject's waves
if pr_temp:
    n_seg = len(pr_temp)
    print('  -> Saved subject %s: %d segments to %spr_Wave.mat, %sgt_Wave.mat' % (PERSON + 1, n_seg, PERSON + 1, PERSON + 1))
    io.savemat(os.path.join(savePath, str(PERSON + 1) + 'pr_Wave.mat'), {'Wave': np.array(pr_temp)})
    io.savemat(os.path.join(savePath, str(PERSON + 1) + 'gt_Wave.mat'), {'Wave': np.array(gt_temp)})
print('Done. Per-subject waves written to %s' % savePath)

# %%
