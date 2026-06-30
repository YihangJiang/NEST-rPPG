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
_PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..'))
_STMAP_MY_ROOT = os.path.join(_PROJECT_ROOT, 'STMap_my')
BUAA_MY_ROOT = os.path.join(_STMAP_MY_ROOT, 'BUAA_my')

# All BUAA_my region trees (same subject folder names, e.g. Sub_10lux 15.8)
BUAA_MY_VARIANTS = {
    'BUAA_my': os.path.join(_STMAP_MY_ROOT, 'BUAA_my'),
    'BUAA_my_in': os.path.join(_STMAP_MY_ROOT, 'BUAA_my_in'),
    'BUAA_my_rm': os.path.join(_STMAP_MY_ROOT, 'BUAA_my_rm'),
    'BUAA_my_eye': os.path.join(_STMAP_MY_ROOT, 'BUAA_my_eye'),
}
# Reference tree for cross-region equality checks
LABEL_COMPARE_REFERENCE = 'BUAA_my'

# Mat files to compare under each subject's Label/ folder: (filename, mat key)
LABEL_MAT_SPECS = [
    ('BVP.mat', 'BVP'),
    ('BVP_Filt.mat', 'BVP'),
    ('HR.mat', 'HR'),
]

# Set to a folder name (e.g. "Sub_10lux 15.8") to process one subject only; None = batch.
ONLY_SUBJECT = None

# Set True to regenerate BVP.mat / BVP_Filt.mat (run summary cell first).
RUN_LABEL_PRO = False

# %%
def bvp_array_stats(bvp, name='BVP'):
    """Return a one-line stats dict for a 1D BVP array."""
    x = np.asarray(bvp, dtype=np.float64).reshape(-1)
    n_nan = int(np.isnan(x).sum())
    x_finite = x[np.isfinite(x)]
    if x_finite.size == 0:
        return {
            'name': name,
            'len': len(x),
            'nan': n_nan,
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
        }
    return {
        'name': name,
        'len': len(x),
        'nan': n_nan,
        'min': float(np.min(x_finite)),
        'max': float(np.max(x_finite)),
        'mean': float(np.mean(x_finite)),
        'std': float(np.std(x_finite)),
    }


def print_bvp_stats(stats):
    print(
        f"    {stats['name']:10s} len={stats['len']:5d}  "
        f"nan={stats['nan']:3d}  "
        f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
        f"mean={stats['mean']:.4f}  std={stats['std']:.4f}"
    )


def summarize_bvp_mats(label_dir, stmap_width=None):
    """Load BVP.mat and BVP_Filt.mat under label_dir and print summary lines."""
    paths = {
        'BVP': os.path.join(label_dir, 'BVP.mat'),
        'BVP_Filt': os.path.join(label_dir, 'BVP_Filt.mat'),
    }
    for label_name, mat_path in paths.items():
        if not os.path.isfile(mat_path):
            print(f"    {label_name:10s} MISSING ({mat_path})")
            continue
        bvp = scio.loadmat(mat_path)['BVP']
        stats = bvp_array_stats(bvp, name=label_name)
        print_bvp_stats(stats)
    if stmap_width is not None:
        for label_name, mat_path in paths.items():
            if not os.path.isfile(mat_path):
                continue
            bvp_len = int(np.asarray(scio.loadmat(mat_path)['BVP']).reshape(-1).shape[0])
            match = 'OK' if bvp_len == stmap_width else 'MISMATCH'
            print(f"    vs STMap width ({stmap_width}): {label_name} len={bvp_len} -> {match}")


def run_bvp_summary():
    """Print stats for existing BVP.mat / BVP_Filt.mat (read-only)."""
    print("BVP label summary (read-only)")
    print("BUAA_MY_ROOT:", BUAA_MY_ROOT)
    if ONLY_SUBJECT is not None:
        print("ONLY_SUBJECT:", ONLY_SUBJECT)
    if not os.path.isdir(BUAA_MY_ROOT):
        print(f"BUAA_my root not found: {BUAA_MY_ROOT}")
        return

    n_ok = 0
    n_missing = 0
    for buaa_my_folder in sorted(os.listdir(BUAA_MY_ROOT)):
        if ONLY_SUBJECT is not None and buaa_my_folder != ONLY_SUBJECT:
            continue
        if buaa_my_folder.startswith('.'):
            continue
        buaa_my_path = os.path.join(BUAA_MY_ROOT, buaa_my_folder)
        if not os.path.isdir(buaa_my_path):
            continue

        label_dir = os.path.join(buaa_my_path, 'Label')
        bvp_path = os.path.join(label_dir, 'BVP.mat')
        filt_path = os.path.join(label_dir, 'BVP_Filt.mat')
        stmap_path = os.path.join(buaa_my_path, 'STMap', 'STMap_RGB.png')

        stmap_width = None
        if os.path.isfile(stmap_path):
            stmap_img = cv2.imread(stmap_path)
            if stmap_img is not None:
                stmap_width = stmap_img.shape[1]

        print("=" * 72)
        print(buaa_my_folder)
        if stmap_width is not None:
            print(f"  STMap width: {stmap_width}")
        else:
            print("  STMap width: (no STMap_RGB.png)")
        if not os.path.isfile(bvp_path) and not os.path.isfile(filt_path):
            print("  No BVP.mat or BVP_Filt.mat")
            n_missing += 1
            continue

        summarize_bvp_mats(label_dir, stmap_width=stmap_width)
        n_ok += 1

    print("=" * 72)
    print(f"Subjects with labels: {n_ok}  |  missing both mats: {n_missing}")


def _load_label_vector(mat_path, key):
    """Load one 1D label vector from a .mat file, or None if missing."""
    if not os.path.isfile(mat_path):
        return None
    data = scio.loadmat(mat_path)
    if key not in data:
        return None
    return np.asarray(data[key], dtype=np.float64).reshape(-1)


def _compare_label_vectors(a, b, rtol=1e-5, atol=1e-8):
    """Compare two 1D arrays; return (equal, max_abs_diff, reason)."""
    if a is None and b is None:
        return True, 0.0, 'both missing'
    if a is None:
        return False, np.nan, 'A missing'
    if b is None:
        return False, np.nan, 'B missing'
    if a.shape != b.shape:
        return False, np.nan, f'shape {a.shape} vs {b.shape}'
    if np.array_equal(a, b):
        return True, 0.0, 'exact'
    if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        max_diff = float(np.nanmax(np.abs(a - b)))
        return True, max_diff, 'close'
    max_diff = float(np.nanmax(np.abs(a - b)))
    return False, max_diff, 'diff'


def _iter_subject_ids(variant_roots):
    """Subject folder names present in at least one variant root."""
    subjects = set()
    for root in variant_roots.values():
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.startswith('.'):
                continue
            if os.path.isdir(os.path.join(root, name)):
                subjects.add(name)
    return sorted(subjects)


def run_cross_region_label_check(
    variant_roots=None,
    reference_name=None,
    rtol=1e-5,
    atol=1e-8,
):
    """
    Double-check BVP.mat, BVP_Filt.mat, HR.mat are identical across BUAA_my variants.
    Compares every non-reference variant to LABEL_COMPARE_REFERENCE (or reference_name).
    """
    variant_roots = variant_roots or BUAA_MY_VARIANTS
    reference_name = reference_name or LABEL_COMPARE_REFERENCE
    if reference_name not in variant_roots:
        raise ValueError(f'Unknown reference variant: {reference_name!r}')

    ref_root = variant_roots[reference_name]
    other_names = [n for n in variant_roots if n != reference_name]

    print('Cross-region label check (read-only)')
    print('Reference:', reference_name, '->', ref_root)
    for name in other_names:
        print(f'  vs {name}: {variant_roots[name]}')
    if ONLY_SUBJECT is not None:
        print('ONLY_SUBJECT:', ONLY_SUBJECT)

    missing_roots = [n for n, p in variant_roots.items() if not os.path.isdir(p)]
    if missing_roots:
        print('Missing roots:', ', '.join(missing_roots))
    if not os.path.isdir(ref_root):
        print(f'Reference root not found: {ref_root}')
        return

    subjects = _iter_subject_ids(variant_roots)
    if ONLY_SUBJECT is not None:
        subjects = [s for s in subjects if s == ONLY_SUBJECT]

    n_all_match = 0
    n_mismatch = 0
    n_partial = 0

    for subject_id in subjects:
        ref_label = os.path.join(ref_root, subject_id, 'Label')
        row_issues = []

        for mat_name, mat_key in LABEL_MAT_SPECS:
            ref_path = os.path.join(ref_label, mat_name)
            ref_vec = _load_label_vector(ref_path, mat_key)
            if ref_vec is None:
                row_issues.append(f'{mat_name}: missing in {reference_name}')
                continue

            for var_name in other_names:
                var_root = variant_roots[var_name]
                if not os.path.isdir(var_root):
                    row_issues.append(f'{mat_name}: root missing ({var_name})')
                    continue
                var_path = os.path.join(var_root, subject_id, 'Label', mat_name)
                var_vec = _load_label_vector(var_path, mat_key)
                same, max_diff, reason = _compare_label_vectors(
                    ref_vec, var_vec, rtol=rtol, atol=atol
                )
                if not same:
                    row_issues.append(
                        f'{mat_name} vs {var_name}: {reason}'
                        + (f' (max_diff={max_diff:.6g})' if np.isfinite(max_diff) else '')
                    )

        if not row_issues:
            n_all_match += 1
            if ONLY_SUBJECT is not None:
                print(f'{subject_id}: ALL MATCH ({reference_name} == others for BVP, BVP_Filt, HR)')
            continue

        # Only print subjects with at least one file in reference or any variant
        any_present = any(
            os.path.isfile(os.path.join(variant_roots[n], subject_id, 'Label', m))
            for n in variant_roots
            if os.path.isdir(variant_roots[n])
            for m, _ in LABEL_MAT_SPECS
        )
        if not any_present:
            continue

        print('=' * 72)
        print(subject_id)
        for issue in row_issues:
            print(' ', issue)
        if any('missing in' in x for x in row_issues):
            n_partial += 1
        else:
            n_mismatch += 1

    print('=' * 72)
    print(
        f'Checked {len(subjects)} subject(s): '
        f'all match={n_all_match}, mismatch={n_mismatch}, partial/missing={n_partial}'
    )
    if ONLY_SUBJECT is None and n_all_match == len(subjects) and n_mismatch == 0 and n_partial == 0:
        print('All label mats match across regions.')


# %%
# --- Inspect existing labels (run this before RUN_LABEL_PRO) ---
run_bvp_summary()

# %%
# --- Generate / overwrite BVP.mat and BVP_Filt.mat ---
if not RUN_LABEL_PRO:
    print("RUN_LABEL_PRO=False — set RUN_LABEL_PRO=True to regenerate labels.")
elif not os.path.isdir(BUAA_MY_ROOT):
    print(f"BUAA_my root not found: {BUAA_MY_ROOT}")
else:
    for buaa_my_folder in sorted(os.listdir(BUAA_MY_ROOT)):
        if ONLY_SUBJECT is not None and buaa_my_folder != ONLY_SUBJECT:
            continue
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
        print(f"  -> saved BVP.mat, BVP_Filt.mat  (STMap width={Num}, raw PPG len={len(pulse)})")

# %%
# --- Cross-region check: BVP / BVP_Filt / HR identical across BUAA_my variants ---
run_cross_region_label_check()

# %%
