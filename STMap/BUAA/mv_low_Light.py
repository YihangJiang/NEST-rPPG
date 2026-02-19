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


def get_file(dir_path, file_type):
    for subfile in os.listdir(dir_path):
            if subfile.endswith(file_type):
                return subfile
    print(subfile)
    print('*************')
    return 0

# 功能： lmk for UBFC
# 注释： 1.大约有5个文件时间对不齐需要删除  2.有的检测不到lmk

fileRoot = '/mnt/nvme2/rppg_data/BUAA'
newRoot = '/mnt/nvme2/rppg_data/BUAA_low_light/'
file_list_p = os.listdir(fileRoot)
z = 0
for subfile_p in file_list_p:
    now_path_p = os.path.join(fileRoot, subfile_p)
    file_list = os.listdir(now_path_p)
    for subfile in file_list:
        if subfile in ['lux 1.0', 'lux 1.6', 'lux 2.5', 'lux 4.0', 'lux 6.3']:
            print(subfile)
            now_path = os.path.join(now_path_p, subfile)
            print(now_path)
            new_path = os.path.join(newRoot, subfile_p, subfile)
            if os.path.exists(new_path):
                print('  Skip (exists):', new_path)
                continue
            os.makedirs(os.path.join(newRoot, subfile_p), exist_ok=True)
            shutil.copytree(now_path, new_path)
            shutil.rmtree(now_path)

        # print(z)
        # z = z + 1
        # now_path = os.path.join(now_path_p, subfile)
        # video_path = os.path.join(now_path, get_file(now_path, 'avi'))
        # print('video_path')
        # print(video_path)
        # save_path = os.path.join(now_path, 'Label')
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # # 读取 video
        # cap = cv2.VideoCapture(video_path)
        # Num = cap.get(7)
        # Num = int(Num)
        # # 读取 time
        # # 计算lmk
        # # 计算lmk
        # lmk = []
        # while (cap.isOpened()):
        #     ret, frame = cap.read()
        #     if ret:
        #         preds = fa.get_landmarks(frame)
        #         if preds is None:
        #             lmk.append([0 for _ in range(136)])
        #         else:
        #             preds = preds[0]
        #             lmk.append(preds.reshape(136))
        #     else:
        #         break
        # with open(os.path.join(save_path, 'RGB_lmk.csv'), 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for row in lmk:
        #         writer.writerow(row)
        # print(now_path)
        # cap.release()



# %%
