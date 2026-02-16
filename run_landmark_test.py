#%%
#!/usr/bin/env python3
"""
Standalone test script: extract face landmarks from lux100.0_WQT.avi only.

This DOES NOT import or modify STMap/BUAA/Landmark.py. It simply replicates the
single-video landmark extraction logic here.

Output: ./Label/RGB_lmk.csv (same directory as this script/video).
"""
import os
import sys
import csv
# %%
import face_alignment
import cv2
import matplotlib.pyplot as plt
import numpy as np

# %%
# Path to this script (and the test video)
SCRIPT_DIR =  '/home/yj167/Desktop/NEST-rPPG'
TEST_VIDEO = '/home/yj167/Desktop/NEST-rPPG/lux100.0_WQT.avi'
device_name = 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                  flip_input=False,
                                  device=device_name)



def extract_landmarks_from_video(video_path, save_dir=None, device="cpu"):
    """
    Extract 2D face landmarks for every frame of a video and save as RGB_lmk.csv.

    - One row per frame
    - 136 values per row (68 landmarks x 2 coordinates)
    """
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        print("Video not found:", video_path)
        return None

    if save_dir is None:
        save_dir = os.path.dirname(video_path)

    label_dir = os.path.join(save_dir, "Label")
    os.makedirs(label_dir, exist_ok=True)
    csv_path = os.path.join(label_dir, "RGB_lmk.csv")

    print("Loading face alignment model on", device)
    # Hard-code 2D landmarks type and device as requested
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                      flip_input=False,
                                      device=device)

    print("Reading video:", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return None

    # Try to get total frame count for progress estimation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    if total_frames > 0:
        print("Total frames (approx):", total_frames)


    lmk_rows = []
    frame_idx = 0
    last_print_pct = -1.0

    while cap.isOpened():
        print(frame_idx)
        ret, frame = cap.read()

        if not ret:
            break
        preds = fa.get_landmarks(frame)
        if preds is None:
            lmk_rows.append([0.0 for _ in range(136)])
        else:
            lmk_rows.append(preds[0].reshape(136).tolist())
        frame_idx += 1

        # Progress estimate: print every ~5% or every 50 frames when frame count unknown
        if total_frames > 0:
            pct = 100.0 * frame_idx / total_frames
            if pct - last_print_pct >= 5.0 or frame_idx == total_frames:
                print(f"Processed {frame_idx}/{total_frames} frames ({pct:.1f}%)")
                last_print_pct = pct
        else:
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames (total unknown)")
    
    print("out")

    print("Writing landmarks to:", csv_path)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in lmk_rows:
            writer.writerow(row)

    print("Saved", len(lmk_rows), "frames to", csv_path)
    cap.release()
    return csv_path


# %%
print("start extracting landmarks")
if not os.path.isfile(TEST_VIDEO):
    print("Test video not found:", TEST_VIDEO)
    sys.exit(1)

# %%
# Extract landmarks from video
out = extract_landmarks_from_video(
    TEST_VIDEO,
    save_dir=SCRIPT_DIR,
    device="cpu",
)
if out:
    print("Done. Landmarks:", out)
else:
    sys.exit(1)

# %%
