# %%
"""
Generate one STMap using only BUAA_my (video) and Label (RGB_lmk.csv).
Run cells in order in Jupyter or VS Code.
"""
import os
import sys
import csv
import math
import cv2
import numpy as np
from scipy import signal

# %%
# Config: set paths (works when run as script or in Jupyter)
if '__file__' in dir():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = os.getcwd()  # Jupyter: set this to NEST-rPPG folder if needed
BUAA_MY_DIR = os.path.join(SCRIPT_DIR, 'STMap_my', 'BUAA_my')
LABEL_DIR = os.path.join(SCRIPT_DIR, 'Label')
STMAP_NAME = 'STMap_RGB.png'
# None = use all frames; set to int (e.g. 500) for a shorter run
MAX_FRAMES = None

# %%
def find_video_path(dir_path, extensions=('.avi', '.mp4', '.mov')):
    """Return path to first video file in dir_path, or None."""
    if not os.path.isdir(dir_path):
        return None
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith(extensions):
            return os.path.join(dir_path, f)
    return None


def extract_frames_to_folder(video_path, out_folder, max_frames=None):
    """Extract video frames to out_folder as 00000.png, 00001.png, ... Returns number written."""
    if not os.path.isfile(video_path):
        return 0
    os.makedirs(out_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and n >= max_frames):
            break
        path = os.path.join(out_folder, f'{n:05d}.png')
        cv2.imwrite(path, frame)
        n += 1
    cap.release()
    return n


def PointRotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    Rotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    Rotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return Rotatex, Rotatey


def getValue(img, lmk=[], type=2, lmk_type=2, channels='YUV'):
    """Extract 5x5 ROI values; type=2 uses face landmarks for alignment."""
    Value = []
    h, w, c = img.shape
    if type == 1:
        w_step = int(w / 5)
        h_step = int(h / 5)
        for w_index in range(5):
            for h_index in range(5):
                temp = img[h_index * h_step: (h_index + 1) * h_step, w_index * w_step:(w_index + 1) * w_step, :]
                temp1 = np.nanmean(np.nanmean(temp, axis=0), axis=0)
                Value.append(temp1)
    elif type == 2:
        lmk = np.array(lmk, np.float32).reshape(-1, 2)
        min_p = np.min(lmk, 0)
        max_p = np.max(lmk, 0)
        min_p = np.maximum(min_p, 0)
        max_p = np.minimum(max_p, [w - 1, h - 1])
        if lmk_type == 1:
            left_eye = lmk[0:8]
            right_eye = lmk[9:17]
            left = np.array([lmk[60], lmk[62], lmk[65]])
            right = np.array([lmk[61], lmk[63], lmk[73]])
        else:
            left_eye = lmk[36:41]
            right_eye = lmk[42:47]
            left = np.array([lmk[0], lmk[1], lmk[2]])
            right = np.array([lmk[14], lmk[15], lmk[16]])
        left_eye = np.nanmean(left_eye, 0)
        right_eye = np.nanmean(right_eye, 0)
        left = np.nanmean(left, 0)
        right = np.nanmean(right, 0)
        top = max((left[1] + right[1]) / 2 - 0.5 * (max_p[1] - (left[1] + right[1]) / 2), 0)
        rotate_angular = math.atan((right_eye[1] - left_eye[1]) / (0.00001 + right_eye[0] - left_eye[0])) * (180 / math.pi)
        cent_point = [w / 2, h / 2]
        matRotation = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_angular, 1)
        face_rotate = cv2.warpAffine(img, matRotation, (w, h))
        left[0], left[1] = PointRotate(math.radians(rotate_angular), left[0], left[1], cent_point[0], cent_point[1])
        right[0], right[1] = PointRotate(math.radians(rotate_angular), right[0], right[1], cent_point[0], cent_point[1])
        face_crop = face_rotate[int(top):int(max_p[1]), int(left[0]):int(right[0]), :]
        h, w, c = face_crop.shape
        w_step = max(1, int(w / 5))
        h_step = max(1, int(h / 5))
        for w_index in range(5):
            for h_index in range(5):
                temp = face_crop[h_index * h_step: (h_index + 1) * h_step, w_index * w_step:(w_index + 1) * w_step, :]
                if temp.size == 0:
                    temp1 = np.array([100, 100, 100], dtype=np.float64)
                else:
                    temp1 = np.mean(np.mean(temp, axis=0), axis=0)
                Value.append(temp1)
    return np.array(Value)


def mySTMap(imglist_root, lmk_all, use_landmarks=True):
    """Build STMap from image folder and landmark list. use_landmarks=True => getValue(..., type=2)."""
    img_list = sorted([f for f in os.listdir(imglist_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    n_use = min(len(img_list), len(lmk_all))
    img_list = img_list[:n_use]
    lmk_use = lmk_all[:n_use]
    STMap = []
    for z, imgPath_sub in enumerate(img_list):
        now_path = os.path.join(imglist_root, imgPath_sub)
        img = cv2.imread(now_path)
        if img is None:
            continue
        Value = getValue(img, lmk=lmk_use[z], type=2 if use_landmarks else 1)
        if np.isnan(Value).any():
            Value = np.nan_to_num(Value, nan=100.0)
        STMap.append(Value)
    STMap = np.array(STMap)
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            mn, mx = np.nanmin(STMap[:, w, c]), np.nanmax(STMap[:, w, c])
            STMap[:, w, c] = 255 * ((STMap[:, w, c] - mn) / (0.001 + mx - mn))
    STMap = np.swapaxes(STMap, 0, 1)
    STMap = np.rint(STMap)
    STMap = np.array(STMap, dtype='uint8')
    return STMap

# %%
# Run: load Label, get frames (extract or use existing), build STMap, save in BUAA_my
lmk_path = os.path.join(LABEL_DIR, 'RGB_lmk.csv')
if not os.path.isfile(lmk_path):
    raise FileNotFoundError('Label file not found: ' + lmk_path)

video_path = find_video_path(BUAA_MY_DIR)
if not video_path:
    raise FileNotFoundError('No video (.avi/.mp4/.mov) found in: ' + BUAA_MY_DIR)

lmk_all = []
with open(lmk_path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        lmk_all.append(line)
n_landmarks = len(lmk_all)
print('Landmark rows:', n_landmarks, '| Video:', video_path)

# %%
align_folder = os.path.join(BUAA_MY_DIR, 'Align')
to_extract = n_landmarks if MAX_FRAMES is None else min(MAX_FRAMES, n_landmarks)
existing = [f for f in os.listdir(align_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.isdir(align_folder) else []
n_existing = len(existing)
if n_existing >= to_extract:
    print('Using existing frames in Align:', n_existing)
    n_frames = n_existing
else:
    n_frames = extract_frames_to_folder(video_path, align_folder, max_frames=to_extract)
    print('Frames extracted:', n_frames)

n_use = min(n_frames, len(lmk_all))
if n_use == 0:
    raise ValueError('No frames or landmarks.')
lmk_use = lmk_all[:n_use]

# %%
STMap = mySTMap(align_folder, lmk_use, use_landmarks=False)
stmap_dir = os.path.join(BUAA_MY_DIR, 'STMap')
os.makedirs(stmap_dir, exist_ok=True)
out_path = os.path.join(stmap_dir, STMAP_NAME)
cv2.imwrite(out_path, STMap)
print('Saved:', out_path)
