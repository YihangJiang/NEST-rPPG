#!/usr/bin/env python
# %%
"""
Align_Eye_PURE
---------------

Prepare eye-region frames for STMap generation by copying eye PNG frames from
`/mnt/nvme2rppg_data/DATASET_3LEFT_EYE` into the `Align/` folders of
`STMap_my/PURE_my_eye/<nest_id>/`.

Unlike `Align_Face_PURE.py`, this script:
- does NOT use landmarks
- does NOT warp or crop faces
- simply renames eye frames to `10000.png, 10001.png, ...` under Align/.

Inputs:
  - Mapping CSV: `pure_nest_to_dataset3_mapping.csv` in this folder
        nest_id,dataset3_id,match_method,overlap_info
  - Eye frames:
        /mnt/nvme2rppg_data/DATASET_3LEFT_EYE/<dataset3_id>/<dataset3_id>/*.png
    or, if the nested folder does not exist:
        /mnt/nvme2rppg_data/DATASET_3LEFT_EYE/<dataset3_id>/*.png

Output:
  - For each mapped `nest_id`, fills:
        STMap_my/PURE_my_eye/<nest_id>/Align/10000.png, 10001.png, ...
"""

import os
import csv
import shutil

# %% config paths
LEFT_EYE_ROOT = '/mnt/nvme2/rppg_data/DATASET_3IN'

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
PURE_MY_EYE_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'PURE_my_in')

FRAME_NAME_START = 10000  # first frame saved as 10000.png


def find_eye_frames_root(session_id):
    """
    For a DATASET_3LEFT_EYE session like '02-02', return the directory
    containing PNG frames:
        /.../DATASET_3LEFT_EYE/02-02/02-02
    or, if that does not exist:
        /.../DATASET_3LEFT_EYE/02-02
    """
    base = os.path.join(LEFT_EYE_ROOT, session_id)
    if not os.path.isdir(base):
        return None
    nested = os.path.join(base, session_id)
    if os.path.isdir(nested):
        return nested
    return base


def copy_eye_frames_to_align(eye_root, align_root):
    """
    Copy all PNG frames from `eye_root` into `align_root`, renaming them
    sequentially as 10000.png, 10001.png, ...
    """
    os.makedirs(align_root, exist_ok=True)

    frame_files = sorted(
        f for f in os.listdir(eye_root)
        if f.lower().endswith('.png')
    )
    if not frame_files:
        print('  Skip (no PNG frames):', eye_root)
        return 0

    # Optionally clear existing Align files to avoid mixing old/new frames.
    for f in os.listdir(align_root):
        if f.lower().endswith('.png'):
            try:
                os.remove(os.path.join(align_root, f))
            except OSError:
                pass

    z = FRAME_NAME_START
    for fname in frame_files:
        src = os.path.join(eye_root, fname)
        dst = os.path.join(align_root, f'{z}.png')
        shutil.copy2(src, dst)
        z += 1

    return z - FRAME_NAME_START


# %%
# Run over all subjects (works both as a script and when executed as a Jupyter cell)
if not os.path.isdir(LEFT_EYE_ROOT):
    print('DATASET_3LEFT_EYE root not found:', LEFT_EYE_ROOT)
elif not os.path.isdir(PURE_MY_EYE_ROOT):
    print('PURE_my_eye root not found:', PURE_MY_EYE_ROOT)
else:
    print('LEFT_EYE_ROOT   :', LEFT_EYE_ROOT)
    print('PURE_MY_EYE_ROOT:', PURE_MY_EYE_ROOT)

    processed = 0
    # Folder names like '01-01', '02-02', ... are assumed to match
    # the DATASET_3LEFT_EYE session IDs directly.
    for ds_id in sorted(os.listdir(PURE_MY_EYE_ROOT)):
        if ds_id.startswith('.'):
            continue
        subject_root = os.path.join(PURE_MY_EYE_ROOT, ds_id)
        if not os.path.isdir(subject_root):
            continue

        eye_frames_root = find_eye_frames_root(ds_id)
        if not eye_frames_root:
            print(f'{ds_id}: eye session folder not found under DATASET_3LEFT_EYE')
            continue

        align_root = os.path.join(subject_root, 'Align')
        print(f'{processed:03d} {ds_id}  eye_root={eye_frames_root}')
        n_frames = copy_eye_frames_to_align(eye_frames_root, align_root)
        print(f'  -> {align_root} ({n_frames} frames)')
        processed += 1

    print('Done. Subjects processed:', processed)


# %%
