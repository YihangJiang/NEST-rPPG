# %%
"""
Generate Label folder for UBFC_my: BVP.mat, HR.mat only.
- Reads ground_truth.txt from UBFC raw (DATASET_2): line1 = BVP waveform, line2 = HR, line3 = timestamps.
- Aligns to video frame count (STMap width or RGB_lmk rows) via spline interpolation.
- BVP: interpolate to frame count only (no normalization). No BVP_Filt.
Run after Landmark_UBFC and Align/STMap so UBFC_my/<subject>/Label/RGB_lmk.csv and optionally STMap exist.
"""
import os
import cv2
import numpy as np
import scipy.io as scio
from scipy import interpolate

# %%
UBFC_RAW_ROOT = '/mnt/nvme2/rppg_data/DATASET_2'
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..'))
UBFC_MY_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'UBFC_my')

# %%
def load_ground_truth(gt_path):
    """
    Load UBFC ground_truth.txt. Typical format: 3 lines (BVP, HR, timestamps), space-separated floats.
    Returns (bvp, hr) as 1d arrays; timestamps optional. Returns (None, None) on error.
    """
    if not os.path.isfile(gt_path):
        return None, None
    try:
        with open(gt_path) as f:
            lines = [L.strip() for L in f.readlines() if L.strip()]
    except Exception:
        return None, None
    if len(lines) < 2:
        return None, None
    try:
        bvp = np.array([float(x) for x in lines[0].split()])
        hr = np.array([float(x) for x in lines[1].split()])
    except ValueError:
        return None, None
    return bvp, hr


def interpolate_to_length(x, target_len):
    """Spline interpolate 1d array x to target_len samples (same idea as BUAA splrep/splev)."""
    n = len(x)
    if n < 4:
        return np.linspace(x[0], x[-1] if n > 1 else x[0], target_len)
    t_orig = np.linspace(0, target_len, n)
    t_target = np.linspace(0, target_len, target_len, endpoint=False)
    spl = interpolate.splrep(t_orig, x, k=min(3, n - 1))
    out = interpolate.splev(t_target, spl)
    if not np.isfinite(out).all():
        fill = np.nanmean(out) if np.isfinite(np.nanmean(out)) else np.nanmean(x)
        out = np.nan_to_num(out, nan=fill if np.isfinite(fill) else 0.0, posinf=0.0, neginf=0.0)
    return np.asarray(out, dtype=np.float64)


# %%
if not os.path.isdir(UBFC_MY_ROOT):
    print(f"UBFC_my root not found: {UBFC_MY_ROOT}")
else:
    for sub_name in sorted(os.listdir(UBFC_MY_ROOT)):
        if sub_name.startswith('.'):
            continue
        save_path = os.path.join(UBFC_MY_ROOT, sub_name, 'Label')
        lmk_path = os.path.join(save_path, 'RGB_lmk.csv')
        stmap_path = os.path.join(UBFC_MY_ROOT, sub_name, 'STMap', 'STMap_RGB.png')

        # Target length: STMap width if available, else landmark row count
        Num = None
        if os.path.isfile(stmap_path):
            temp = cv2.imread(stmap_path)
            if temp is not None:
                Num = temp.shape[1]
        if Num is None and os.path.isfile(lmk_path):
            with open(lmk_path) as f:
                Num = sum(1 for _ in f)
        if Num is None or Num < 1:
            print(f"  Skip (no STMap or RGB_lmk to get frame count): {sub_name}")
            continue

        gt_path = os.path.join(UBFC_RAW_ROOT, sub_name, 'ground_truth.txt')
        bvp_raw, hr_raw = load_ground_truth(gt_path)
        if bvp_raw is None or hr_raw is None:
            print(f"  Skip (no/invalid ground_truth.txt): {gt_path}")
            continue

        os.makedirs(save_path, exist_ok=True)

        # Interpolate to Num frames (same as BUAA)
        pulse_csi = interpolate_to_length(bvp_raw, Num)
        hr_csi = interpolate_to_length(hr_raw, Num)

        # BVP only: interpolate to frame count (no normalization)
        scio.savemat(os.path.join(save_path, 'BVP.mat'), {'BVP': pulse_csi.reshape(1, -1)})

        # HR: save per frame (interpolated)
        scio.savemat(os.path.join(save_path, 'HR.mat'), {'HR': hr_csi.reshape(1, -1)})

        print(sub_name, '->', save_path, f'({Num} frames)')

# %%
