# %%
"""STMap generation for BUAA. Run cells in order in Jupyter or VS Code."""
import sys
import re
import os
import shutil
import scipy.io as io
import xlrd
import math
import csv
import cv2
import numpy as np
from math import *
from scipy import signal
import scipy.io as scio
from scipy import interpolate
from scipy import signal

# %%
# Config: set paths for your environment (edit and run this cell first)
fileRoot = '/mnt/nvme2/rppg_data/BUAA_EYE'
STMap_name = 'STMap_RGB.png'
_script_dir = os.path.dirname(os.path.abspath(__file__))
BUAA_MY_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..', 'STMap_my', 'BUAA_my_eye'))

# %%
def PointRotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    Rotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    Rotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return Rotatex, Rotatey


def getValue(img, lmk=[], type = 1, lmk_type=2, channels='YUV'):
    Value = []
    # 1.三点对齐 2.两点对齐
    # 1.81点 2.68 点
    h, w, c = img.shape
    # if channels == 'YUV':
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
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
        # cv2.imshow('a', face_crop)
        # cv2.waitKey(0)
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
    # b, a = signal.butter(5, 0.12 / (30 / 2), 'highpass')
    # b, a = signal.butter(5, [0.5 / (30 / 2), 3 / (30 / 2)], 'bandpass')
    img_list = sorted(os.listdir(imglist_root))
    z = 0
    STMap = []
    for imgPath_sub in img_list:
        now_path = os.path.join(imglist_root, imgPath_sub)
        img = cv2.imread(now_path)
        Value = getValue(img, lmk=lmk_all[z])
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
# Run STMap generation: read Align + Label from BUAA_my, write STMap to BUAA_my
z = 0
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
        buaa_my_folder = f"{subfile_p}{subfile}"
        lmk_path = os.path.join(BUAA_MY_ROOT, buaa_my_folder, 'Label', 'RGB_lmk.csv')
        RGB_path = os.path.join(BUAA_MY_ROOT, buaa_my_folder, 'Align')
        STMap_path = os.path.join(BUAA_MY_ROOT, buaa_my_folder, 'STMap')
        if not os.path.exists(lmk_path):
            print('  Skip (no landmarks):', lmk_path)
            continue
        if not os.path.exists(RGB_path):
            print('  Skip (no Align folder):', RGB_path)
            continue
        os.makedirs(STMap_path, exist_ok=True)
        print(z, buaa_my_folder)
        lmk_all = []
        with open(lmk_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lmk_all.append(line)
        STMap = mySTMap(RGB_path, lmk_all=lmk_all)
        out_path = os.path.join(STMap_path, STMap_name)
        cv2.imwrite(out_path, STMap, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print('  ->', out_path)
        z += 1

# %%
