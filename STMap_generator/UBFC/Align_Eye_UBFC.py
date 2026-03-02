#!/usr/bin/env python
# %%
"""
Align_Eye_UBFC
---------------

Prepare eye-region frames for STMap generation by extracting frames from
eye-only UBFC videos into the `Align/` folders of
`STMap_my/UBFC_my_eye/<subject_id>/`.

Unlike `Align_Face_UBFC.py`, this script:
- does NOT use landmarks
- does NOT warp or crop faces
- simply saves raw eye video frames as `10000.png, 10001.png, ...` under Align/.

Assumptions:
- Eye videos are stored per subject under:
      /mnt/nvme2/rppg_data/DATASET_2LEFT_EYE/<subject_id>/*.avi|*.mp4|*.mov
- You have duplicated UBFC_my into UBFC_my_eye, so folders like:
      STMap_my/UBFC_my_eye/<subject_id>/
  already exist (for labels/BVP/etc.).
"""

import os
import shutil
import cv2

# %% config paths
UBFC_EYE_ROOT = '/mnt/nvme2/rppg_data/DATASET_2RM'

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
UBFC_MY_EYE_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'UBFC_my_rm')

FRAME_NAME_START = 10000  # first frame saved as 10000.png


def get_video_file(dir_path):
    """Return first video file in dir_path with common extensions, or None."""
    if not os.path.isdir(dir_path):
        return None
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith(('.avi', '.mp4', '.mov')):
            return os.path.join(dir_path, f)
    return None


def extract_eye_video_to_align(video_path, align_root):
    """
    Read an eye-only UBFC video and save each frame into `align_root`,
    renaming them sequentially as 10000.png, 10001.png, ...
    """
    os.makedirs(align_root, exist_ok=True)

    # Clear existing Align PNGs to avoid mixing old/new frames.
    for f in os.listdir(align_root):
        if f.lower().endswith('.png'):
            try:
                os.remove(os.path.join(align_root, f))
            except OSError:
                pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('  Skip (cannot open eye video):', video_path)
        return 0

    z = FRAME_NAME_START
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(align_root, f'{z}.png')
        cv2.imwrite(out_path, frame)
        z += 1
        frame_count += 1

    cap.release()
    return frame_count


# %%
# Run over all subjects (script or Jupyter cell)
if not os.path.isdir(UBFC_EYE_ROOT):
    print('UBFC eye root not found:', UBFC_EYE_ROOT)
elif not os.path.isdir(UBFC_MY_EYE_ROOT):
    print('UBFC_my_eye root not found:', UBFC_MY_EYE_ROOT)
else:
    print('UBFC_EYE_ROOT   :', UBFC_EYE_ROOT)
    print('UBFC_MY_EYE_ROOT:', UBFC_MY_EYE_ROOT)

    processed = 0
    for subj_id in sorted(os.listdir(UBFC_MY_EYE_ROOT)):
        if subj_id.startswith('.'):
            continue
        subject_root = os.path.join(UBFC_MY_EYE_ROOT, subj_id)
        if not os.path.isdir(subject_root):
            continue

        eye_subject_dir = os.path.join(UBFC_EYE_ROOT, subj_id)
        video_path = get_video_file(eye_subject_dir)
        if video_path is None:
            print(f'{subj_id}: no eye video found under {eye_subject_dir}')
            continue

        align_root = os.path.join(subject_root, 'Align')
        print(f'{processed:03d} {subj_id}  eye_video={video_path}')
        n_frames = extract_eye_video_to_align(video_path, align_root)
        print(f'  -> {align_root} ({n_frames} frames)')
        processed += 1

    print('Done. Subjects processed:', processed)

# %%

