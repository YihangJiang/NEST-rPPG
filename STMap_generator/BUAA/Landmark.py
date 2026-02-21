# %%
import face_alignment
import cv2
import os
import numpy as np
import csv
# Function: calculate 2D landmarks and save RGB_lmk.csv


def get_file(dir_path, suffix):
    """Return first filename in dir_path ending with suffix, or None."""
    if not os.path.isdir(dir_path):
        return None
    for subfile in os.listdir(dir_path):
        if subfile.endswith(suffix):
            return subfile
    return None


# Raw dataset root: fileRoot/Subject/lux_xx.xx/*.avi
fileRoot = '/mnt/nvme2/rppg_data/BUAA'

# Preprocessing output root: BUAA_my/Sub_numlux_num/Label/RGB_lmk.csv
if '__file__' in dir():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    saveRoot = os.path.normpath(os.path.join(_script_dir, '..', '..', 'STMap_my', 'BUAA_my'))
else:
    saveRoot = os.path.join(os.getcwd(), 'STMap_my', 'BUAA_my')

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device='cuda:0',
)

# %%
# For each subject and lux folder under fileRoot, find the .avi and write:
#   BUAA_my/Sub_numlux_num/Label/RGB_lmk.csv
idx = 0
for sub_name in sorted(os.listdir(fileRoot)):
    if sub_name.startswith('.'):  # Skip hidden/system files like .DS_Store
        continue
    subj_dir = os.path.join(fileRoot, sub_name)
    if not os.path.isdir(subj_dir):
        continue

    for lux_name in sorted(os.listdir(subj_dir)):
        if lux_name.startswith('.'):  # Skip hidden/system files like .DS_Store
            continue
        lux_dir = os.path.join(subj_dir, lux_name)
        if not os.path.isdir(lux_dir):
            continue

        avi_name = get_file(lux_dir, '.avi')
        if not avi_name:
            continue

        video_path = os.path.join(lux_dir, avi_name)
        out_folder_name = f"{sub_name}{lux_name}"  # e.g. Sub_01 + 'lux 10.0' -> 'Sub_01lux 10.0'
        print(idx, out_folder_name)
        idx += 1

        save_dir = os.path.join(saveRoot, out_folder_name, 'Label')
        os.makedirs(save_dir, exist_ok=True)
        out_csv = os.path.join(save_dir, 'RGB_lmk.csv')

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
                preds = preds[0]
                lmk.append(preds.reshape(136))
        cap.release()

        with open(out_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in lmk:
                writer.writerow(row)
        print('  ->', out_csv)

# %%
