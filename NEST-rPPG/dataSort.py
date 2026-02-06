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
savePath = os.path.join(BASE_DIR, 'Wave_sort', 'PURE')
if not os.path.exists(savePath):
    os.makedirs(savePath)
Idex_files = os.path.join(BASE_DIR, 'STMap', 'STMap_Index', 'PURE')
gt_path = os.path.join(BASE_DIR, 'Result', 'rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1WAVE_ALL.mat')
pr_path = os.path.join(BASE_DIR, 'Result', 'rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1WAVE_PR_ALL.mat')
pr = scio.loadmat(pr_path)['Wave']
pr = np.squeeze(np.array(pr.astype('float32')))
gt = scio.loadmat(gt_path)['Wave']
gt = np.squeeze(np.array(gt.astype('float32')))

files_list = os.listdir(Idex_files)
files_list = sorted(files_list)
temp = scio.loadmat(os.path.join(Idex_files, files_list[0]))
lastPath = str(temp['Path'][0])
pr_temp = []
gt_temp = []
print(pr.shape)
PERSON = 10000
for HR_index in range(pr.shape[0]):
    temp = scio.loadmat(os.path.join(Idex_files, files_list[HR_index]))
    nowPath = str(temp['Path'][0])
    Step_Index = int(temp['Step_Index'])
    if lastPath != nowPath:
        PERSON = PERSON + 1
        if pr_temp is None:
            print(nowPath)
            print(lastPath)
            pr_temp = []
            gt_temp = []
        else:
            print(lastPath)
            print(PERSON)
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

