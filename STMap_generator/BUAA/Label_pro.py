# %%
import cv2
import os
import numpy as np
import shutil
import pandas as pd
import json
import scipy.io as scio
from scipy import interpolate
import csv
import h5py
from scipy import signal

# %%
LPF = 0.7  # low cutoff frequency(Hz) - specified as 40bpm(~0.667Hz) in reference
HPF = 2.5  # high cutoff frequency(Hz) - specified as 240bpm(~4.0Hz) in reference
NyquistF = 15  # 15fps
FS = 30  # 30fps
B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
z = 0

fileRoot = '/mnt/nvme2/rppg_data/BUAA'
_script_dir = os.path.dirname(os.path.abspath(__file__))
BUAA_MY_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..', 'STMap_my', 'BUAA_my'))

# %%
# Iterate over BUAA_my folders (e.g. Sub_01lux 10.0, Sub_02lux 63.1, etc.)
if not os.path.isdir(BUAA_MY_ROOT):
    print(f"BUAA_my root not found: {BUAA_MY_ROOT}")
else:
    for buaa_my_folder in sorted(os.listdir(BUAA_MY_ROOT)):
        if buaa_my_folder.startswith('.'):  # Skip hidden/system files like .DS_Store
            continue
        buaa_my_path = os.path.join(BUAA_MY_ROOT, buaa_my_folder)
        if not os.path.isdir(buaa_my_path):
            continue
        
        # Parse folder name: Sub_01lux 10.0 -> Sub_01, lux 10.0
        # Find where "lux" starts (after Sub_XX)
        lux_idx = buaa_my_folder.find('lux')
        if lux_idx == -1:
            print(f'  Skip (invalid folder name format): {buaa_my_folder}')
            continue
        
        sub_name = buaa_my_folder[:lux_idx]  # e.g. "Sub_01"
        lux_name = buaa_my_folder[lux_idx:]  # e.g. "lux 10.0"
        
        # Find corresponding PPGData.mat in original BUAA structure
        original_path = os.path.join(fileRoot, sub_name, lux_name)
        data_path = os.path.join(original_path, 'PPGData.mat')
        
        # Read STMap from BUAA_my: Sub_numluxnum/STMap/STMap_RGB.png
        STMap_path = os.path.join(buaa_my_path, 'STMap', 'STMap_RGB.png')
        # Write labels to BUAA_my: Sub_numluxnum/Label/
        save_path = os.path.join(buaa_my_path, 'Label')
        os.makedirs(save_path, exist_ok=True)
        
        if not os.path.isfile(STMap_path):
            print(f'  Skip (no STMap found): {STMap_path}')
            continue
        if not os.path.isfile(data_path):
            print(f'  Skip (no PPGData.mat found): {data_path}')
            continue
        
        print(buaa_my_folder)
        temp = cv2.imread(STMap_path)
        if temp is None:
            print(f'  Skip (cannot read STMap): {STMap_path}')
            continue
        Num = temp.shape[1]

        pulse = scio.loadmat(data_path)['PPG']['data'][0][0]
        print(pulse)
        pulse = np.array(np.array(pulse).astype('float32')).reshape(-1)

        print('len(pulse)', len(pulse))
        Time = np.linspace(0, Num, len(pulse))
        CSI_Time = np.linspace(0, Num, Num)

        t = interpolate.splrep(Time, pulse)
        pulse_csi = interpolate.splev(CSI_Time, t)


        pulse_csi = (pulse_csi - np.min(pulse_csi)) / (np.max(pulse_csi) - np.min(pulse_csi))
        EXG2_f = signal.filtfilt(B, A, pulse_csi)
        EXG2_f = (EXG2_f - np.min(EXG2_f)) / (np.max(EXG2_f) - np.min(EXG2_f))
        scio.savemat(os.path.join(save_path, 'BVP_Filt.mat'), {'BVP': EXG2_f})
        scio.savemat(os.path.join(save_path, 'BVP.mat'), {'BVP': pulse_csi})



# %%
