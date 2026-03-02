# %%
"""STMap generation for UBFC. Run cells in order in Jupyter or VS Code."""
import os
import math
import csv
import cv2
import numpy as np
from math import *

# %%
# Config: set paths for your environment (edit and run this cell first)
STMap_name = 'STMap_RGB.png'
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..'))
UBFC_MY_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'UBFC_my_eye')

# %%
def PointRotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    Rotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    Rotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return Rotatex, Rotatey


def getValue(img, lmk=[], type = 1, lmk_type=2, channels='BGR'):
    Value = []
    # 1.三点对齐 2.两点对齐
    # 1.81点 2.68 点
    h, w, c = img.shape
    # Currently using BGR (no conversion)
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
        max_p = np.minimum(max_p, [w - 1, h-1])
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
        top = max((left[1] + right[1])/2 - 0.5*(max_p[1] - (left[1] + right[1])/2), 0)
        rotate_angular = math.atan((right_eye[1] - left_eye[1]) / (0.00001+right_eye[0] - left_eye[0])) * (180 / math.pi)
        # 旋转
        cent_point = [w/2, h/2]
        matRotation = cv2.getRotationMatrix2D((w/2, h/2), rotate_angular, 1)
        face_rotate = cv2.warpAffine(img, matRotation, (w, h))
        left[0], left[1] = PointRotate(math.radians(rotate_angular), left[0], left[1], cent_point[0], cent_point[1])
        right[0], right[1] = PointRotate(math.radians(rotate_angular), right[0], right[1], cent_point[0], cent_point[1])
        # 截取
        face_crop = face_rotate[int(top):int(max_p[1]), int(left[0]):int(right[0]), :]
        h, w, c = face_crop.shape
        w_step = int(w / 5)
        h_step = int(h / 5)
        for w_index in range(5):
            for h_index in range(5):
                temp = face_crop[h_index * h_step: (h_index + 1) * h_step, w_index * w_step:(w_index + 1) * w_step, :]
                temp1 = np.mean(np.mean(temp, axis=0), axis=0)
                Value.append(temp1)
    return np.array(Value)


def mySTMap(imglist_root, lmk_all=[]):
    # Only use pre-aligned frames (10000.png, 10001.png, ...) from Align_Face_UBFC.py
    # Exclude old raw frames (00000.png, 00001.png, ...) from Align_UBFC.py
    all_files = os.listdir(imglist_root)
    img_list = []
    for f in all_files:
        if f.endswith('.png'):
            name_no_ext = f[:-4]  # remove .png
            # Only include files >= 10000 (pre-aligned from Align_Face_UBFC.py)
            try:
                frame_num = int(name_no_ext)
                if frame_num >= 10000:
                    img_list.append(f)
            except ValueError:
                continue  # Skip non-numeric filenames
    img_list = sorted(img_list)  # Sort numerically by filename
    
    z = 0
    STMap = []
    for imgPath_sub in img_list:
        now_path = os.path.join(imglist_root, imgPath_sub)
        img = cv2.imread(now_path)
        # Use type=1 (simple 5x5 grid) since images are already pre-aligned 128x128 crops (same as BUAA)
        # Landmarks are passed but not used when type=1 (kept for consistency with BUAA)
        Value = getValue(img, lmk=lmk_all[z] if z < len(lmk_all) else [], type=1)
        if np.isnan(Value).any():
            Value[:, :] = 100
        STMap.append(Value)
        z = z + 1
    STMap = np.array(STMap)
    # Normal
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            STMap[:, w, c] = 255 * ((STMap[:, w, c] - np.nanmin(STMap[:, w, c])) / (
                    0.001 + np.nanmax(STMap[:, w, c]) - np.nanmin(STMap[:, w, c])))
    STMap = np.swapaxes(STMap, 0, 1)
    STMap = np.rint(STMap)
    STMap = np.array(STMap, dtype='uint8')
    return STMap


# %%
# Run STMap generation: read Align + Label from UBFC_my, write STMap to UBFC_my
if not os.path.isdir(UBFC_MY_ROOT):
    print(f"UBFC_my root not found: {UBFC_MY_ROOT}")
else:
    z = 0
    for sub_name in sorted(os.listdir(UBFC_MY_ROOT)):
        if sub_name.startswith('.'):  # Skip hidden/system files like .DS_Store
            continue
        subject_path = os.path.join(UBFC_MY_ROOT, sub_name)
        if not os.path.isdir(subject_path):
            continue
        
        lmk_path = os.path.join(subject_path, 'Label', 'RGB_lmk.csv')
        RGB_path = os.path.join(subject_path, 'Align')
        STMap_path = os.path.join(subject_path, 'STMap')
        
        if not os.path.exists(lmk_path):
            print('  Skip (no landmarks):', lmk_path)
            continue
        if not os.path.exists(RGB_path):
            print('  Skip (no Align folder):', RGB_path)
            continue
        
        os.makedirs(STMap_path, exist_ok=True)
        print(z, sub_name)
        
        # Load landmarks (one row per frame)
        lmk_all = []
        with open(lmk_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lmk_all.append(line)
        
        # Build STMap from Align frames and landmarks
        # Note: mySTMap filters for files >= 10000 and uses type=1 (simple grid on pre-aligned crops)
        STMap = mySTMap(RGB_path, lmk_all=lmk_all)
        
        # Warn if frame count mismatch (should match if Align_Face_UBFC.py processed all frames)
        n_align_frames = len([f for f in os.listdir(RGB_path) if f.endswith('.png') and f[:-4].isdigit() and int(f[:-4]) >= 10000])
        if n_align_frames != len(lmk_all):
            print(f'  Warning: Align frames ({n_align_frames}) != landmark rows ({len(lmk_all)})')
        out_path = os.path.join(STMap_path, STMap_name)
        cv2.imwrite(out_path, STMap, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print('  ->', out_path)
        z += 1

# %%
