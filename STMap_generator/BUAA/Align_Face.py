# %%
"""Align face from video using landmarks; save cropped 128x128 frames to Align/ and masks to Mask/."""
import os
import csv
import cv2
import numpy as np

# %%
# Config
fileRoot = '/mnt/nvme2/rppg_data/BUAA'

# Base folder in this repo where aligned PNGs will be mirrored,
# e.g. BUAA_my/Sub_01lux10.0/Align/xxxxx.png
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
BUAA_MY_ROOT = os.path.join(_REPO_ROOT, 'STMap_my', 'BUAA_my')
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
def get_file(dir_path, suffix):
    """Return first filename in dir_path ending with suffix, or None."""
    if not os.path.isdir(dir_path):
        return None
    for f in os.listdir(dir_path):
        if f.endswith(suffix):
            return f
    return None


def lmk_roi_points(lmk, indices):
    """Return (N, 1, 2) int32 array for cv2.fillPoly from landmark indices."""
    pts = np.array([[lmk[i, 0], lmk[i, 1]] for i in indices], dtype=np.float32)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def process_video(now_path, video_path, lmk_path, align_path, mask_path, buaa_align_path=None):
    """Read video and landmarks; write aligned face crops and masks per frame.

    If buaa_align_path is provided, aligned crops are also mirrored there
    (BUAA_my/Sub_*lux*/Align/xxxxx.png structure).
    """
    os.makedirs(align_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    if buaa_align_path is not None:
        os.makedirs(buaa_align_path, exist_ok=True)

    with open(lmk_path, 'r') as f:
        lmk_all = list(csv.reader(f))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('  Skip (cannot open video):', video_path)
        return

    lmk_index = 0
    z = FRAME_NAME_START
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if lmk_index >= len(lmk_all):
            break
        lmk = np.array(lmk_all[lmk_index], dtype=np.float32).reshape(-1, 2)
        lmk_index += 1

        h, w = img.shape[:2]
        # Face mask: full face polygon minus eyes, nose, mouth
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.fillPoly(mask, [lmk_roi_points(lmk, ROI_FACE)], (255, 255, 255))
        for indices in ROI_EXCLUDE:
            cv2.fillPoly(mask, [lmk_roi_points(lmk, indices)], (0, 0, 0))
        img_masked = cv2.bitwise_and(img, mask)

        # Affine warp: 3-point alignment
        old_pts = lmk[AFFINE_SRC_INDICES, :].astype(np.float32)
        M = cv2.getAffineTransform(old_pts, AFFINE_DST)
        out_size = (max(w, 128), max(h, 128))
        face_align = cv2.warpAffine(img, M, out_size)
        face_align = face_align[0:OUTPUT_SIZE, 0:OUTPUT_SIZE, :]

        out_name = f'{z}.png'
        cv2.imwrite(os.path.join(align_path, out_name), face_align)
        if buaa_align_path is not None:
            cv2.imwrite(os.path.join(buaa_align_path, out_name), face_align)
        cv2.imwrite(os.path.join(mask_path, f'{z}.png'), mask)  # full-frame mask
        z += 1

    cap.release()
    print('  ->', align_path, f'({z - FRAME_NAME_START} frames)')


# %%
# Run over all subject folders under fileRoot
for subfile_p in sorted(os.listdir(fileRoot)):
    if subfile_p.startswith('.'):  # Skip hidden/system files like .DS_Store
        continue
    now_path_p = os.path.join(fileRoot, subfile_p)
    if not os.path.isdir(now_path_p):
        continue
    for subfile in sorted(os.listdir(now_path_p)):
        if subfile.startswith('.'):  # Skip hidden/system files like .DS_Store
            continue
        now_path = os.path.join(now_path_p, subfile)
        if not os.path.isdir(now_path):
            continue
        avi_name = get_file(now_path, 'avi')
        if not avi_name:
            continue
        video_path = os.path.join(now_path, avi_name)
        # Landmarks are in BUAA_my (e.g. Sub_01lux 10.0/Label/RGB_lmk.csv), not in original BUAA
        buaa_my_folder = f"{subfile_p}{subfile}"  # e.g. Sub_01 + "lux 10.0" -> Sub_01lux 10.0
        lmk_path = os.path.join(BUAA_MY_ROOT, buaa_my_folder, 'Label', 'RGB_lmk.csv')
        if not os.path.isfile(lmk_path):
            print('  Skip (no Label/RGB_lmk.csv in BUAA_my):', lmk_path)
            continue
        align_path = os.path.join(now_path, 'Align')
        mask_path = os.path.join(now_path, 'Mask')

        # Mirror aligned PNGs into BUAA_my: Sub_numluxnum/Align/
        buaa_align_path = os.path.join(BUAA_MY_ROOT, buaa_my_folder, 'Align')

        print(now_path, '->', buaa_align_path)
        process_video(now_path, video_path, lmk_path, align_path, mask_path, buaa_align_path=buaa_align_path)

# %%
