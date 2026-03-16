#!/usr/bin/env python
# %%
"""
Align_Eye_BUAA
---------------

Prepare eye-region frames for STMap generation by extracting frames from
eye-only BUAA videos into the `Align/` folders of
`STMap_my/BUAA_my_eye/<subject_id>/`.

Unlike the face-alignment script, this script:
- does NOT use landmarks
- does NOT warp or crop faces
- simply saves raw eye video frames as `10000.png, 10001.png, ...` under Align/.

Assumptions:
- Eye videos are stored per subject under:
      /mnt/nvme2/rppg_data/BUAA_EYE/<subject_id>/*.avi|*.mp4|*.mov
- You have duplicated BUAA_my into BUAA_my_eye (or a similar eye-only copy), so folders like:
      STMap_my/BUAA_my_eye/<subject_id>/
  already exist (for labels/BVP/etc.).
"""

import os
import shutil
import cv2

# %% config paths (edit these to match your BUAA eye data layout)
BUAA_EYE_ROOT = '/mnt/nvme2/rppg_data/BUAA_RM'

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
# Default: write Align frames into STMap_my/BUAA_my_eye/<subject_id>/Align
BUAA_MY_EYE_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'BUAA_my_rm') 

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
if not os.path.isdir(BUAA_EYE_ROOT):
    print('BUAA eye root not found:', BUAA_EYE_ROOT)
elif not os.path.isdir(BUAA_MY_EYE_ROOT):
    print('BUAA_my_eye root not found:', BUAA_MY_EYE_ROOT)
else:
    print('BUAA_EYE_ROOT   :', BUAA_EYE_ROOT)
    print('BUAA_MY_EYE_ROOT:', BUAA_MY_EYE_ROOT)

    processed = 0
    for subj_id in sorted(os.listdir(BUAA_MY_EYE_ROOT)):
        if subj_id.startswith('.'):
            continue
        subject_root = os.path.join(BUAA_MY_EYE_ROOT, subj_id)
        if not os.path.isdir(subject_root):
            continue

        # BUAA_my_eye subject folders were created by concatenating two levels
        # from the original BUAA tree, e.g. "Sub_01" + "lux 10.0" -> "Sub_01lux 10.0".
        # To find the corresponding eye-video folder under BUAA_EYE_ROOT, we
        # split the subject id back into these two components.
        #
        # Assumes BUAA subject IDs start with a fixed-length "Sub_XX" prefix.
        subj_prefix = subj_id[:6]           # e.g. "Sub_01"
        subj_suffix = subj_id[6:]           # e.g. "lux 10.0"
        eye_subject_dir = os.path.join(BUAA_EYE_ROOT, subj_prefix, subj_suffix)
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
