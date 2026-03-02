# %%
"""Align face from PURE PNG frames using landmarks; save cropped 128x128 frames to Align/ and masks to Mask/."""
import os
import csv
import cv2
import numpy as np

# %%
# Config
PURE_RAW_ROOT = '/mnt/nvme2/rppg_data/DATASET_3'

# Base folder in this repo where aligned PNGs will be saved,
# e.g. STMap_my/PURE_my/01-01/Align/xxxxx.png
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
PURE_MY_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'PURE_my')

OUTPUT_SIZE = 128
FRAME_NAME_START = 10000  # first frame saved as 10000.png, 10001.png, ...

# ROI indices (68-point face): face outline, then exclude eyes, nose, mouth
ROI_FACE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 22, 21, 17]
ROI_EXCLUDE = [
    [36, 17, 21, 39, 40, 41],  # left eye
    [42, 22, 26, 45, 46, 47],  # right eye
    [31, 33, 35, 30],          # nose
    [51, 53, 54, 55, 57, 59, 48, 49],  # mouth
]
# 3-point affine: left face (1), right face (15), chin (8) -> fixed triangle
AFFINE_SRC_INDICES = [1, 15, 8]
AFFINE_DST = np.array([[0, 48], [128, 48], [64, 128]], dtype=np.float32)

# %%
def lmk_roi_points(lmk, indices):
    """Return (N, 1, 2) int32 array for cv2.fillPoly from landmark indices."""
    pts = np.array([[lmk[i, 0], lmk[i, 1]] for i in indices], dtype=np.float32)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def process_frames(frames_root, lmk_path, align_path, mask_path):
    """Read PNG frames and landmarks; write aligned face crops and masks per frame."""
    os.makedirs(align_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    # Load landmarks (one row per frame)
    with open(lmk_path, 'r') as f:
        lmk_all = list(csv.reader(f))

    # Get sorted list of PNG frames
    frame_files = sorted(
        f for f in os.listdir(frames_root)
        if f.lower().endswith('.png')
    )
    if not frame_files:
        print('  Skip (no PNG frames):', frames_root)
        return

    # Ensure frame count matches landmark count
    if len(frame_files) != len(lmk_all):
        print(f'  Warning: frame count ({len(frame_files)}) != landmark count ({len(lmk_all)})')

    lmk_index = 0
    z = FRAME_NAME_START
    for fname in frame_files:
        if lmk_index >= len(lmk_all):
            break
        
        img_path = os.path.join(frames_root, fname)
        img = cv2.imread(img_path)
        if img is None:
            print('    Warning: cannot read image:', img_path)
            lmk_index += 1
            continue

        lmk = np.array(lmk_all[lmk_index], dtype=np.float32).reshape(-1, 2)
        lmk_index += 1

        h, w = img.shape[:2]
        # Face mask: full face polygon minus eyes, nose, mouth
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.fillPoly(mask, [lmk_roi_points(lmk, ROI_FACE)], (255, 255, 255))
        for indices in ROI_EXCLUDE:
            cv2.fillPoly(mask, [lmk_roi_points(lmk, indices)], (0, 0, 0))
        img_masked = cv2.bitwise_and(img, mask)

        # Affine warp: 3-point alignment (same as UBFC/BUAA)
        old_pts = lmk[AFFINE_SRC_INDICES, :].astype(np.float32)
        M = cv2.getAffineTransform(old_pts, AFFINE_DST)
        out_size = (max(w, 128), max(h, 128))
        face_align = cv2.warpAffine(img, M, out_size)
        face_align = face_align[0:OUTPUT_SIZE, 0:OUTPUT_SIZE, :]

        out_name = f'{z}.png'
        cv2.imwrite(os.path.join(align_path, out_name), face_align)
        cv2.imwrite(os.path.join(mask_path, f'{z}.png'), mask)  # full-frame mask
        z += 1

    print('  ->', align_path, f'({z - FRAME_NAME_START} frames)')


# %%
# Run over all subject folders under PURE_RAW_ROOT
if not os.path.isdir(PURE_RAW_ROOT):
    print(f"PURE raw root not found: {PURE_RAW_ROOT}")
else:
    os.makedirs(PURE_MY_ROOT, exist_ok=True)
    
    for sub_name in sorted(os.listdir(PURE_RAW_ROOT)):
        if sub_name.startswith('.'):  # Skip hidden/system files like .DS_Store
            continue
        raw_subject_path = os.path.join(PURE_RAW_ROOT, sub_name)
        if not os.path.isdir(raw_subject_path):
            continue
        
        # PURE frames are stored as PNGs in a nested folder with the same name, e.g.
        #   DATASET_3/01-01/01-01/ImageXXXXXXXXXXXX.png
        # Fallback: if that folder is missing, look for PNGs directly under raw_subject_path.
        frames_root = os.path.join(raw_subject_path, sub_name)
        if not os.path.isdir(frames_root):
            frames_root = raw_subject_path
        
        # Check if frames exist
        frame_files = sorted(
            f for f in os.listdir(frames_root)
            if f.lower().endswith('.png')
        )
        if not frame_files:
            print('  Skip (no PNG frames):', frames_root)
            continue
        
        # Landmarks are in PURE_my (e.g. 01-01/Label/RGB_lmk.csv)
        lmk_path = os.path.join(PURE_MY_ROOT, sub_name, 'Label', 'RGB_lmk.csv')
        if not os.path.isfile(lmk_path):
            print('  Skip (no Label/RGB_lmk.csv in PURE_my):', lmk_path)
            continue
        
        # Output paths: STMap_my/PURE_my/<subject>/Align/ and Mask/
        align_path = os.path.join(PURE_MY_ROOT, sub_name, 'Align')
        mask_path = os.path.join(PURE_MY_ROOT, sub_name, 'Mask')

        print(sub_name, '->', align_path)
        process_frames(frames_root, lmk_path, align_path, mask_path)

# %%
