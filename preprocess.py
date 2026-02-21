#%%
#!/usr/bin/env python3
"""
Preprocessing script: extract face landmarks and preprocess PPG labels.

1. Extract 2D face landmarks from video and save as RGB_lmk.csv
2. Preprocess PPGData.mat label and save BVP.mat and BVP_Filt.mat to Label/

Output: ./Label/RGB_lmk.csv, BVP.mat, BVP_Filt.mat
"""
import os
import sys
import csv
# %%
import cv2
import face_alignment
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import interpolate
from scipy import signal

# %%
# Path to this script (and the test video)
SCRIPT_DIR = '/home/yj167/Desktop/NEST-rPPG'
TEST_VIDEO = '/home/yj167/Desktop/NEST-rPPG/BUAA_my/lux100.0_WQT.avi'
PPG_DATA_PATH = '/home/yj167/Desktop/NEST-rPPG/PPGData.mat'
device_name = 'cpu'

# Filter parameters (from Label_pro.py)
LPF = 0.7  # low cutoff frequency(Hz) - specified as 40bpm(~0.667Hz) in reference
HPF = 2.5  # high cutoff frequency(Hz) - specified as 240bpm(~4.0Hz) in reference
NyquistF = 15  # 15fps
FS = 30  # 30fps
B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')

# %%
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
    return csv_path, frame_idx  # Return frame count for preprocessing


def preprocess_ppg_label(ppg_data_path, num_frames, stmap_path=None, save_dir=None):
    """
    Preprocess PPGData.mat label:
    1. Extract raw pulse data
    2. Interpolate to match target length (STMap width if available, else video frame count)
    3. Normalize
    4. Apply bandpass filter
    5. Save BVP.mat and BVP_Filt.mat to Label/
    
    Args:
        ppg_data_path: Path to PPGData.mat
        num_frames: Number of video frames (fallback target length)
        stmap_path: Optional path to STMap image (if provided, uses its width as target)
        save_dir: Directory to save processed labels (default: same as script)
    """
    if save_dir is None:
        save_dir = SCRIPT_DIR
    
    label_dir = os.path.join(save_dir, "Label")
    os.makedirs(label_dir, exist_ok=True)
    
    # Determine target length: use STMap width if available, else video frame count
    if stmap_path and os.path.isfile(stmap_path):
        temp = cv2.imread(stmap_path)
        target_length = temp.shape[1]  # STMap width
        print(f"\nUsing STMap width as target length: {target_length}")
    else:
        target_length = num_frames
        print(f"\nUsing video frame count as target length: {target_length}")
    
    print(f"Preprocessing PPG label from {ppg_data_path}")
    
    # Load PPGData.mat
    if not os.path.isfile(ppg_data_path):
        print(f"PPGData.mat not found: {ppg_data_path}")
        return None
    
    ppg_data = scio.loadmat(ppg_data_path)
    ppg_struct = ppg_data['PPG']
    
    # Extract raw data from structured array
    if isinstance(ppg_struct, np.ndarray) and ppg_struct.dtype.names:
        # Structured array: extract 'data' field
        pulse = ppg_struct['data'][0, 0]
    else:
        # Simple array
        pulse = ppg_struct
    
    pulse = np.array(np.array(pulse).astype('float32')).reshape(-1)
    print(f'Loaded pulse signal length: {len(pulse)}')
    
    # CSI interpolation: align pulse signal to target length
    Time = np.linspace(0, target_length, len(pulse))
    CSI_Time = np.linspace(0, target_length, target_length)
    
    t = interpolate.splrep(Time, pulse)
    pulse_csi = interpolate.splev(CSI_Time, t)
    
    # Normalize
    pulse_csi = (pulse_csi - np.min(pulse_csi)) / (np.max(pulse_csi) - np.min(pulse_csi))
    
    # Apply bandpass filter
    EXG2_f = signal.filtfilt(B, A, pulse_csi)
    EXG2_f = (EXG2_f - np.min(EXG2_f)) / (np.max(EXG2_f) - np.min(EXG2_f))
    
    # Save processed labels
    bvp_path = os.path.join(label_dir, 'BVP.mat')
    bvp_filt_path = os.path.join(label_dir, 'BVP_Filt.mat')
    
    scio.savemat(bvp_path, {'BVP': pulse_csi})
    scio.savemat(bvp_filt_path, {'BVP': EXG2_f})
    
    print(f"Saved BVP.mat: {bvp_path}")
    print(f"Saved BVP_Filt.mat: {bvp_filt_path}")
    
    return bvp_path, bvp_filt_path


# %%
print("=" * 60)
print("PART 1: Extracting landmarks from video")
print("=" * 60)
if not os.path.isfile(TEST_VIDEO):
    print("Test video not found:", TEST_VIDEO)
    sys.exit(1)

# Extract landmarks from video
out, num_frames = extract_landmarks_from_video(
    TEST_VIDEO,
    save_dir=SCRIPT_DIR,
    device="cpu",
)
if out:
    print("Done. Landmarks:", out)
    print(f"Total frames processed: {num_frames}")
else:
    sys.exit(1)

# %%
print("\n" + "=" * 60)
print("PART 2: Preprocessing PPG label")
print("=" * 60)
# Check for STMap (optional - if exists, use its width as target length)
STMAP_PATH = None  # Set to STMap path if available, e.g., './STMap/.../STMap_RGB.png'
# Preprocess PPG label
preprocess_ppg_label(PPG_DATA_PATH, num_frames, stmap_path=STMAP_PATH, save_dir=SCRIPT_DIR)

print("\n" + "=" * 60)
print("Preprocessing complete!")
print("=" * 60)

# %%
