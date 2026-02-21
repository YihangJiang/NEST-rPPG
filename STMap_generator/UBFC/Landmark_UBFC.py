# %%
"""
Extract 2D face landmarks for UBFC videos and save as Label/RGB_lmk.csv per subject.
Output: STMap_my/UBFC_my/<subject>/Label/RGB_lmk.csv (one row per frame, 136 values).
Run this before the UBFC STMap generation in test_stmap.py.
"""
import face_alignment
import cv2
import os
import numpy as np
import csv

def get_file(dir_path, suffix):
    """Return first filename in dir_path ending with suffix, or None."""
    if not os.path.isdir(dir_path):
        return None
    for f in os.listdir(dir_path):
        if f.endswith(suffix):
            return f
    return None


# Raw UBFC dataset root: one folder per subject (subject1, subject2, ...), each with a video
UBFC_RAW_ROOT = '/mnt/nvme2/rppg_data/DATASET_2'

# Output: STMap_my/UBFC_my/<subject>/Label/RGB_lmk.csv
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..'))
UBFC_MY_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'UBFC_my')

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device='cuda:0',
)

# %%
# For each subject folder under UBFC_RAW_ROOT, find video and write landmarks to UBFC_my
if not os.path.isdir(UBFC_RAW_ROOT):
    print(f"UBFC raw root not found: {UBFC_RAW_ROOT}")
else:
    os.makedirs(UBFC_MY_ROOT, exist_ok=True)
    for sub_name in sorted(os.listdir(UBFC_RAW_ROOT)):
        if sub_name.startswith('.'):
            continue
        subj_dir = os.path.join(UBFC_RAW_ROOT, sub_name)
        if not os.path.isdir(subj_dir):
            continue

        video_path = None
        for ext in ('.avi', '.mp4', '.mov'):
            f = get_file(subj_dir, ext)
            if f:
                video_path = os.path.join(subj_dir, f)
                break
        if not video_path:
            print('  Skip (no video):', subj_dir)
            continue

        save_dir = os.path.join(UBFC_MY_ROOT, sub_name, 'Label')
        os.makedirs(save_dir, exist_ok=True)
        out_csv = os.path.join(save_dir, 'RGB_lmk.csv')

        print(sub_name, '->', out_csv)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('  Skip (cannot open video):', video_path)
            continue

        lmk = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            preds = fa.get_landmarks(frame)
            if preds is None:
                lmk.append([0 for _ in range(136)])
            else:
                lmk.append(preds[0].reshape(136).tolist())
        cap.release()

        with open(out_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in lmk:
                writer.writerow(row)
        print('  ->', out_csv, f'({len(lmk)} frames)')

# %%
