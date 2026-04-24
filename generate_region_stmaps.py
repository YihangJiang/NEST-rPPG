#!/usr/bin/env python3
"""
Generate region-specific STMap_my datasets from pre-cropped raw data.

Outputs (under /mnt/nvme2/rppg_data/STMap_my/):
  UBFC_my_in/   <- DATASET_2IN   (infranasal, video)
  UBFC_my_rm/   <- DATASET_2RM   (cheek,      video)
  UBFC_my_eye/  <- DATASET_2LEFT_EYE (eye,    video)
  PURE_my_in/   <- DATASET_3IN   (infranasal, PNG sequence)
  PURE_my_rm/   <- DATASET_3RM   (cheek,      PNG sequence)
  PURE_my_eye/  <- DATASET_3LEFT_EYE (eye,    PNG sequence)

Each subject folder contains:
  STMap/STMap_RGB.png   (25 rows x T frames x 3 channels, uint8, normalized per-row)
  Label/BVP_Filt.mat    {'BVP': (1, T)}
  Label/HR.mat          {'HR': (1, T)}
"""

import os
import re
import json
import numpy as np
import cv2
import scipy.io as scio
from scipy import interpolate, signal

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = '/mnt/nvme2/rppg_data'
STMAP_ROOT  = os.path.join(DATA_ROOT, 'STMap_my')

UBFC_REGIONS = {
    'UBFC_my_in':  'DATASET_2IN',
    'UBFC_my_rm':  'DATASET_2RM',
    'UBFC_my_eye': 'DATASET_2LEFT_EYE',
}
PURE_REGIONS = {
    'PURE_my_in':  'DATASET_3IN',
    'PURE_my_rm':  'DATASET_3RM',
    'PURE_my_eye': 'DATASET_3LEFT_EYE',
}

# Ground-truth sources
UBFC_GT_ROOT = os.path.join(DATA_ROOT, 'DATASET_2')   # subject*/ground_truth.txt
PURE_RAW_ROOT = os.path.join(DATA_ROOT, 'DATASET_3')  # session*/session.json + PNG frames

# BVP bandpass filter (same as Label_pro_PURE.py)
LPF, HPF, FS = 0.7, 2.5, 30
B, A = signal.butter(3, [LPF / (FS / 2), HPF / (FS / 2)], 'bandpass')

GRID = 5   # 5x5 spatial grid -> 25 STMap rows
STMAP_NAME = 'STMap_RGB.png'


# ── Core helpers ──────────────────────────────────────────────────────────────

def build_stmap_from_frames(frames):
    """
    frames: list of (H, W, 3) uint8 BGR images
    Returns (25, T, 3) uint8 STMap, normalized per spatial row across time.
    """
    T = len(frames)
    raw = np.zeros((T, GRID * GRID, 3), dtype=np.float32)
    for t, frame in enumerate(frames):
        h, w = frame.shape[:2]
        rh, rw = h // GRID, w // GRID
        idx = 0
        for ri in range(GRID):
            for ci in range(GRID):
                patch = frame[ri*rh:(ri+1)*rh, ci*rw:(ci+1)*rw]
                raw[t, idx] = patch.mean(axis=(0, 1))
                idx += 1

    # raw: (T, 25, 3) — normalize each spatial position across time to [0, 255]
    for sp in range(GRID * GRID):
        for c in range(3):
            col = raw[:, sp, c]
            lo, hi = col.min(), col.max()
            raw[:, sp, c] = 255.0 * (col - lo) / (hi - lo + 0.001)

    stmap = raw.swapaxes(0, 1)        # (25, T, 3)
    return np.rint(stmap).astype(np.uint8)


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def read_png_sequence(folder):
    """Read sorted PNG frames from a DATASET_3* session folder (nested or flat)."""
    session_name = os.path.basename(folder.rstrip('/'))
    # Try nested folder first (DATASET_3/01-01/01-01/Image*.png)
    nested = os.path.join(folder, session_name)
    src = nested if os.path.isdir(nested) else folder
    files = sorted(
        f for f in os.listdir(src)
        if f.lower().endswith('.png')
    )
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(src, f))
        if img is not None:
            frames.append(img)
    return frames


def save_subject(out_dir, stmap, bvp_filt, hr):
    os.makedirs(os.path.join(out_dir, 'STMap'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'Label'), exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, 'STMap', STMAP_NAME), stmap)
    T = stmap.shape[1]
    scio.savemat(os.path.join(out_dir, 'Label', 'BVP_Filt.mat'),
                 {'BVP': bvp_filt[:T].reshape(1, -1)})
    scio.savemat(os.path.join(out_dir, 'Label', 'HR.mat'),
                 {'HR': hr[:T].reshape(1, -1)})


# ── UBFC region generation ─────────────────────────────────────────────────────

def load_ubfc_gt(subject):
    gt_path = os.path.join(UBFC_GT_ROOT, subject, 'ground_truth.txt')
    if not os.path.isfile(gt_path):
        return None, None
    with open(gt_path) as f:
        lines = f.readlines()
    bvp = np.array(lines[0].split(), dtype=np.float64)
    hr  = np.array(lines[1].split(), dtype=np.float64)
    return bvp, hr


def gen_ubfc_region(region_name, src_folder):
    out_root = os.path.join(STMAP_ROOT, region_name)
    subjects = sorted(s for s in os.listdir(src_folder)
                      if os.path.isdir(os.path.join(src_folder, s)))
    print(f'\n=== {region_name} ({len(subjects)} subjects) ===')

    for subj in subjects:
        out_dir = os.path.join(out_root, subj)
        stmap_png = os.path.join(out_dir, 'STMap', STMAP_NAME)
        if os.path.isfile(stmap_png):
            print(f'  skip (exists): {subj}')
            continue

        vid = os.path.join(src_folder, subj, 'vid.avi')
        if not os.path.isfile(vid):
            print(f'  skip (no video): {subj}')
            continue

        bvp, hr = load_ubfc_gt(subj)
        if bvp is None:
            print(f'  skip (no GT): {subj}')
            continue

        frames = read_video_frames(vid)
        if not frames:
            print(f'  skip (empty video): {subj}')
            continue

        stmap = build_stmap_from_frames(frames)
        save_subject(out_dir, stmap, bvp, hr)
        print(f'  {subj}: {stmap.shape[1]} frames -> {stmap_png}')


# ── PURE region generation ─────────────────────────────────────────────────────

def load_pure_json_signals(session_path, session_name):
    json_path = os.path.join(session_path, session_name + '.json')
    if not os.path.isfile(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    pack = data.get('/FullPackage', data.get('FullPackage', []))
    if not pack:
        return None
    ts  = np.array([int(x['Timestamp']) for x in pack], dtype=np.float64)
    hr  = np.array([float(x['Value']['pulseRate']) for x in pack], dtype=np.float64)
    wav = np.array([float(x['Value']['waveform']) for x in pack], dtype=np.float64)
    return ts, hr, wav


def get_frame_timestamps(session_path, session_name):
    """Extract sorted nanosecond timestamps from Image<ts>.png filenames."""
    nested = os.path.join(session_path, session_name)
    src = nested if os.path.isdir(nested) else session_path
    pat = re.compile(r'Image(\d+)\.png', re.IGNORECASE)
    ts = []
    for f in os.listdir(src):
        m = pat.match(f)
        if m:
            ts.append(int(m.group(1)))
    return sorted(ts) if ts else None


def interp(t_orig, y, t_target):
    spl = interpolate.splrep(t_orig, y, k=min(3, len(t_orig) - 1))
    t_target = np.clip(t_target, t_orig.min(), t_orig.max())
    return interpolate.splev(t_target, spl)


def make_bvp_filt(waveform):
    filt = signal.filtfilt(B, A, waveform)
    lo, hi = filt.min(), filt.max()
    return (filt - lo) / (hi - lo + 1e-9)


def gen_pure_region(region_name, src_folder):
    out_root = os.path.join(STMAP_ROOT, region_name)
    sessions = sorted(s for s in os.listdir(src_folder)
                      if os.path.isdir(os.path.join(src_folder, s)))
    print(f'\n=== {region_name} ({len(sessions)} sessions) ===')

    for sess in sessions:
        out_dir = os.path.join(out_root, sess)
        stmap_png = os.path.join(out_dir, 'STMap', STMAP_NAME)
        if os.path.isfile(stmap_png):
            print(f'  skip (exists): {sess}')
            continue

        raw_sess = os.path.join(PURE_RAW_ROOT, sess)
        signals = load_pure_json_signals(raw_sess, sess)
        if signals is None:
            print(f'  skip (no JSON): {sess}')
            continue
        ts_json, hr_json, wav_json = signals

        frame_ts = get_frame_timestamps(raw_sess, sess)

        # Read frames from the region-cropped source
        region_sess = os.path.join(src_folder, sess)
        frames = read_png_sequence(region_sess)
        if not frames:
            print(f'  skip (no frames): {sess}')
            continue

        T = len(frames)

        # Interpolate labels to frame timestamps (or use first T json values)
        if frame_ts is not None and len(frame_ts) >= T:
            t = np.array(frame_ts[:T], dtype=np.float64)
            hr_out  = interp(ts_json, hr_json, t)
            wav_out = interp(ts_json, wav_json, t)
        else:
            n = min(T, len(wav_json))
            hr_out  = hr_json[:n]
            wav_out = wav_json[:n]
            T = n
            frames  = frames[:T]

        bvp_filt = make_bvp_filt(wav_out)
        stmap = build_stmap_from_frames(frames)
        save_subject(out_dir, stmap, bvp_filt, hr_out)
        print(f'  {sess}: {T} frames -> {stmap_png}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Generate region STMaps for UBFC and PURE')
    p.add_argument('--datasets', nargs='+',
                   choices=list(UBFC_REGIONS) + list(PURE_REGIONS) + ['all'],
                   default=['all'])
    args = p.parse_args()

    targets = (list(UBFC_REGIONS) + list(PURE_REGIONS)) if 'all' in args.datasets else args.datasets

    for name in targets:
        if name in UBFC_REGIONS:
            gen_ubfc_region(name, os.path.join(DATA_ROOT, UBFC_REGIONS[name]))
        elif name in PURE_REGIONS:
            gen_pure_region(name, os.path.join(DATA_ROOT, PURE_REGIONS[name]))

    print('\nDone.')
