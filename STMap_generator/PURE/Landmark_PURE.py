# %%
"""
Extract 2D face landmarks for the PURE dataset (stored as PNG image sequences, not videos)
and save them as Label/RGB_lmk.csv per subject.

Input layout (on this machine):
    /mnt/nvme2/rppg_data/DATASET_3/
        01-01/
            01-01/              # PNG frames: ImageXXXXXXXXXXXXXXX.png
            01-01.json          # metadata
        01-02/
            01-02/
            01-02.json
        ...

Output layout (in this repo):
    STMap_my/PURE_my/<subject>/Label/RGB_lmk.csv
Each CSV row = 136 values (68 facial landmarks * 2 coordinates) per frame.

Run this before the PURE STMap generation (align face, then STMap).
"""
import face_alignment
import cv2
import os
import numpy as np
import csv


# Raw PURE dataset root (DATASET_3 on disk): one folder per subject/session (01-01, 01-02, ...)
# containing a nested folder with PNG frames.
PURE_RAW_ROOT = '/mnt/nvme2/rppg_data/DATASET_3'

# Output: STMap_my/PURE_my/<subject>/Label/RGB_lmk.csv
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..'))
PURE_MY_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'PURE_my')

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device='cuda:0',
)

# %%
# For each subject folder under PURE_RAW_ROOT, read PNG frames and write landmarks to PURE_my
if not os.path.isdir(PURE_RAW_ROOT):
    print(f"PURE raw root not found: {PURE_RAW_ROOT}")
else:
    os.makedirs(PURE_MY_ROOT, exist_ok=True)
    for sub_name in sorted(os.listdir(PURE_RAW_ROOT)):
        if sub_name.startswith('.'):
            continue
        subj_dir = os.path.join(PURE_RAW_ROOT, sub_name)  # e.g. DATASET_3/01-01
        if not os.path.isdir(subj_dir):
            continue

        # PURE frames are stored as PNGs in a nested folder with the same name, e.g.
        #   DATASET_3/01-01/01-01/ImageXXXXXXXXXXXX.png
        # Fallback: if that folder is missing, look for PNGs directly under subj_dir.
        frames_root = os.path.join(subj_dir, sub_name)
        if not os.path.isdir(frames_root):
            frames_root = subj_dir

        frame_files = sorted(
            f for f in os.listdir(frames_root)
            if f.lower().endswith('.png')
        )
        if not frame_files:
            print('  Skip (no PNG frames):', frames_root)
            continue

        save_dir = os.path.join(PURE_MY_ROOT, sub_name, 'Label')
        os.makedirs(save_dir, exist_ok=True)
        out_csv = os.path.join(save_dir, 'RGB_lmk.csv')

        print(sub_name, '->', out_csv)
        lmk = []
        for fname in frame_files:
            img_path = os.path.join(frames_root, fname)
            frame = cv2.imread(img_path)
            if frame is None:
                print('    Warning: cannot read image:', img_path)
                lmk.append([0 for _ in range(136)])
                continue
            preds = fa.get_landmarks(frame)
            if preds is None:
                lmk.append([0 for _ in range(136)])
            else:
                lmk.append(preds[0].reshape(136).tolist())

        with open(out_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in lmk:
                writer.writerow(row)
        print('  ->', out_csv, f'({len(lmk)} frames)')

# %%
