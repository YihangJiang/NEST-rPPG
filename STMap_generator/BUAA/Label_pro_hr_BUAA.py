# %%
"""
Generate HR.mat in BUAA_my following the approach of Label_pro_PURE.py.

- Label_pro_PURE: HR comes from JSON pulseRate, interpolated to frame timestamps -> HR.mat (1, N).
- BUAA: no JSON; we derive HR from PPGData.mat BVP by sliding-window FFT (peak in 0.7–2.5 Hz -> BPM),
  one value per frame, and save Label/HR.mat with key 'HR', shape (1, N).

Uses same folder structure as Label_pro.py: BUAA_my/Sub_01lux 10.0/Label/, source PPGData.mat
from fileRoot/Sub_01/lux 10.0/PPGData.mat. Requires STMap and PPGData.mat to exist.
"""
import os
import numpy as np
import cv2
import scipy.io as scio
from scipy import interpolate
from scipy import signal

# %%
# Bandpass for PPG (40–240 bpm), same as Label_pro.py
LPF = 0.7   # Hz
HPF = 2.5   # Hz
NyquistF = 15
FS = 30
B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')

# HR frequency band for FFT peak (BPM -> Hz: divide by 60)
HR_HZ_LOW, HR_HZ_HIGH = 0.7, 2.5
WINDOW_FRAMES = 256  # frames per window for FFT (at 30 fps ~ 8.5 s)

_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..'))
fileRoot = '/mnt/nvme2/rppg_data/BUAA'
BUAA_MY_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..', 'STMap_my', 'BUAA_my'))


def hr_from_window(bvp_window, fs=30.0):
    """Peak frequency in [HR_HZ_LOW, HR_HZ_HIGH] from 1D BVP window; return BPM or nan."""
    bvp = np.asarray(bvp_window, dtype=np.float64).ravel()
    n = len(bvp)
    if n < 32:
        return np.nan
    bvp = bvp - np.nanmean(bvp)
    bvp = np.nan_to_num(bvp, nan=0.0)
    window = np.hanning(n)
    bvp = bvp * window
    spec = np.abs(np.fft.rfft(bvp)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mask = (freqs >= HR_HZ_LOW) & (freqs <= HR_HZ_HIGH)
    if not np.any(mask):
        return np.nan
    idx = np.argmax(spec[mask])
    f_peak = freqs[mask][idx]
    return float(f_peak * 60.0)  # Hz -> BPM


# %%
if not os.path.isdir(BUAA_MY_ROOT):
    print(f"BUAA_my root not found: {BUAA_MY_ROOT}")
else:
    for buaa_my_folder in sorted(os.listdir(BUAA_MY_ROOT)):
        if buaa_my_folder.startswith('.'):
            continue
        buaa_my_path = os.path.join(BUAA_MY_ROOT, buaa_my_folder)
        if not os.path.isdir(buaa_my_path):
            continue

        lux_idx = buaa_my_folder.find('lux')
        if lux_idx == -1:
            print(f'  Skip (invalid folder name): {buaa_my_folder}')
            continue

        sub_name = buaa_my_folder[:lux_idx]
        lux_name = buaa_my_folder[lux_idx:]
        original_path = os.path.join(fileRoot, sub_name, lux_name)
        data_path = os.path.join(original_path, 'PPGData.mat')
        STMap_path = os.path.join(buaa_my_path, 'STMap', 'STMap_RGB.png')
        save_path = os.path.join(buaa_my_path, 'Label')
        os.makedirs(save_path, exist_ok=True)

        if not os.path.isfile(STMap_path):
            print(f'  Skip (no STMap): {buaa_my_folder}')
            continue
        if not os.path.isfile(data_path):
            print(f'  Skip (no PPGData.mat): {data_path}')
            continue

        # Load pulse and interpolate to frame grid (same as Label_pro.py)
        pulse = scio.loadmat(data_path)['PPG']['data'][0][0]
        pulse = np.array(pulse, dtype=np.float64).reshape(-1)
        stmap_img = cv2.imread(STMap_path)
        if stmap_img is None:
            print(f'  Skip (cannot read STMap): {STMap_path}')
            continue
        Num = stmap_img.shape[1]
        Time = np.linspace(0, Num, len(pulse))
        CSI_Time = np.linspace(0, Num, Num)
        t = interpolate.splrep(Time, pulse)
        bvp_csi = interpolate.splev(CSI_Time, t)
        bvp_csi = (bvp_csi - np.nanmin(bvp_csi)) / (np.nanmax(bvp_csi) - np.nanmin(bvp_csi) + 1e-12)
        bvp_csi = np.nan_to_num(bvp_csi, nan=0.0)

        # Per-frame HR: sliding window FFT (method analogous to PURE’s per-frame HR)
        half = WINDOW_FRAMES // 2
        HR = np.zeros(Num, dtype=np.float64)
        for i in range(Num):
            lo = max(0, i - half)
            hi = min(Num, i + half)
            w = bvp_csi[lo:hi]
            if len(w) < 32:
                HR[i] = np.nan
            else:
                HR[i] = hr_from_window(w, fs=FS)
        # Fill any remaining nans with median
        med = np.nanmedian(HR)
        HR = np.nan_to_num(HR, nan=med, posinf=med, neginf=med)

        hr_mat_path = os.path.join(save_path, 'HR.mat')
        scio.savemat(hr_mat_path, {'HR': HR.reshape(1, -1)})
        print(buaa_my_folder, '->', hr_mat_path, f'(Num={Num})')

# %%
