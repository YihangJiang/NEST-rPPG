#%%
#!/usr/bin/env python3
"""
Preprocessing script PART 2: Preprocess PPG label.

Preprocess PPGData.mat label and save BVP.mat and BVP_Filt.mat to Label/

Output: ./Label/BVP.mat, BVP_Filt.mat

Required environment: scipy, numpy, cv2 (for STMap reading)
"""
import os
import sys
import numpy as np
import scipy.io as scio
from scipy import interpolate
from scipy import signal
import cv2
import matplotlib.pyplot as plt

# %%
# Path configuration
SCRIPT_DIR = '/home/yj167/Desktop/NEST-rPPG'
PPG_DATA_PATH = '/home/yj167/Desktop/NEST-rPPG/PPGData.mat'
LABEL_DIR = os.path.join(SCRIPT_DIR, "Label")
RGB_LMK_CSV = os.path.join(LABEL_DIR, "RGB_lmk.csv")
FRAME_COUNT_FILE = os.path.join(LABEL_DIR, "frame_count.txt")

# Filter parameters (from Label_pro.py)
LPF = 0.7  # low cutoff frequency(Hz) - specified as 40bpm(~0.667Hz) in reference
HPF = 2.5  # high cutoff frequency(Hz) - specified as 240bpm(~4.0Hz) in reference
NyquistF = 15  # 15fps
FS = 30  # 30fps
B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')


def get_frame_count(stmap_path=None, frame_count_file=None, csv_path=None):
    """
    Determine target frame count from:
    1. STMap width (if STMap path provided)
    2. Frame count file (if exists)
    3. CSV row count (if CSV exists)
    
    Returns:
        frame_count: Target number of frames
    """
    # Priority 1: Use STMap width if available
    if stmap_path and os.path.isfile(stmap_path):
        temp = cv2.imread(stmap_path)
        if temp is not None:
            target_length = temp.shape[1]  # STMap width
            print(f"Using STMap width as target length: {target_length}")
            return target_length
    
    # Priority 2: Read from frame count file
    if frame_count_file and os.path.isfile(frame_count_file):
        try:
            with open(frame_count_file, "r") as f:
                target_length = int(f.read().strip())
            print(f"Using frame count from file: {target_length}")
            return target_length
        except (ValueError, IOError) as e:
            print(f"Could not read frame count file: {e}")
    
    # Priority 3: Count rows in CSV
    if csv_path and os.path.isfile(csv_path):
        try:
            with open(csv_path, "r") as f:
                target_length = sum(1 for _ in f)
            print(f"Using CSV row count as target length: {target_length}")
            return target_length
        except IOError as e:
            print(f"Could not read CSV file: {e}")
    
    print("Warning: Could not determine frame count. Please provide STMap path or ensure frame_count.txt exists.")
    return None


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
        num_frames: Number of video frames (target length for interpolation)
        stmap_path: Optional path to STMap image (if provided, uses its width as target)
        save_dir: Directory to save processed labels (default: same as script)
    
    Returns:
        bvp_path, bvp_filt_path: Paths to saved files
    """
    if save_dir is None:
        save_dir = SCRIPT_DIR
    
    label_dir = os.path.join(save_dir, "Label")
    os.makedirs(label_dir, exist_ok=True)
    
    if num_frames is None:
        print("Error: Could not determine target frame count")
        return None, None
    
    print(f"\nPreprocessing PPG label from {ppg_data_path}")
    print(f"Target frame count: {num_frames}")
    
    # Load PPGData.mat
    if not os.path.isfile(ppg_data_path):
        print(f"PPGData.mat not found: {ppg_data_path}")
        return None, None
    
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
    Time = np.linspace(0, num_frames, len(pulse))
    CSI_Time = np.linspace(0, num_frames, num_frames)
    
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


def compare_bvp_files(bvp_path, comp_bvp_path):
    """
    Compare two BVP .mat files and generate comparison plots.
    
    Args:
        bvp_path: Path to Label/BVP.mat (processed)
        comp_bvp_path: Path to comp_BVP.mat (reference)
    
    Returns:
        mae, rmse, max_diff: Comparison statistics
    """
    print("\n" + "=" * 60)
    print("PART 3: Comparing BVP files")
    print("=" * 60)
    
    if not os.path.isfile(bvp_path):
        print(f"BVP file not found: {bvp_path}")
        return None, None, None
    
    if not os.path.isfile(comp_bvp_path):
        print(f"Reference BVP file not found: {comp_bvp_path}")
        return None, None, None
    
    print(f"Loading {bvp_path}")
    bvp_data = scio.loadmat(bvp_path)
    bvp_signal = np.squeeze(bvp_data['BVP']).astype(float)
    
    print(f"Loading {comp_bvp_path}")
    comp_bvp_data = scio.loadmat(comp_bvp_path)
    comp_bvp_signal = np.squeeze(comp_bvp_data['BVP']).astype(float)
    
    n_bvp = len(bvp_signal)
    n_comp_bvp = len(comp_bvp_signal)
    print(f"\nSignal lengths: Label/BVP.mat={n_bvp}, comp_BVP.mat={n_comp_bvp}")
    
    # Compare over common length
    n_common = min(n_bvp, n_comp_bvp)
    bvp_c = bvp_signal[:n_common]
    comp_bvp_c = comp_bvp_signal[:n_common]
    
    diff_bvp = np.abs(bvp_c - comp_bvp_c)
    
    # Statistics
    mae = np.mean(diff_bvp)
    max_diff = np.max(diff_bvp)
    rmse = np.sqrt(np.mean(diff_bvp ** 2))
    
    print(f"\nComparison over first {n_common} samples:")
    print(f"  Mean absolute error (MAE):  {mae:.6f}")
    print(f"  Root mean square error (RMSE): {rmse:.6f}")
    print(f"  Max absolute difference:    {max_diff:.6f}")
    
    # Plot: signals overlay
    plt.figure(figsize=(12, 5))
    t = np.arange(n_common)
    plt.plot(t, bvp_c, label="Label/BVP.mat (processed)", alpha=0.7, linewidth=1)
    plt.plot(t, comp_bvp_c, label="comp_BVP.mat (reference)", alpha=0.7, linewidth=1)
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.title(f"BVP signal comparison (first {n_common} samples)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    compare_overlay_path = os.path.join(SCRIPT_DIR, "compare_bvp_overlay.png")
    plt.savefig(compare_overlay_path, dpi=150)
    print(f"\nSaved: compare_bvp_overlay.png")
    plt.show()
    
    # Plot: difference
    plt.figure(figsize=(12, 4))
    plt.plot(t, diff_bvp, label="Absolute difference", color='red', alpha=0.7, linewidth=1)
    plt.xlabel("Sample index")
    plt.ylabel("|Label/BVP - comp_BVP|")
    plt.title(f"BVP difference (MAE={mae:.6f}, RMSE={rmse:.6f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    compare_diff_path = os.path.join(SCRIPT_DIR, "compare_bvp_diff.png")
    plt.savefig(compare_diff_path, dpi=150)
    print(f"Saved: compare_bvp_diff.png")
    plt.show()
    
    return mae, rmse, max_diff


# %%
print("=" * 60)
print("PART 2: Preprocessing PPG label")
print("=" * 60)

# Determine target frame count
STMAP_PATH = None  # Set to STMap path if available, e.g., './STMap/.../STMap_RGB.png'
num_frames = get_frame_count(stmap_path=STMAP_PATH, 
                             frame_count_file=FRAME_COUNT_FILE,
                             csv_path=RGB_LMK_CSV)

if num_frames is None:
    print("Error: Could not determine frame count. Please run preprocess_landmarks.py first or provide STMap path.")
    sys.exit(1)

# Preprocess PPG label
bvp_path, bvp_filt_path = preprocess_ppg_label(
    PPG_DATA_PATH, 
    num_frames, 
    stmap_path=STMAP_PATH, 
    save_dir=SCRIPT_DIR
)

if bvp_path and bvp_filt_path:
    print("\n" + "=" * 60)
    print("PPG label preprocessing complete!")
    print("=" * 60)
else:
    print("Error: Failed to preprocess PPG label")
    sys.exit(1)

# %%
# Compare with reference BVP file
COMP_BVP_PATH = os.path.join(SCRIPT_DIR, "comp_BVP.mat")
if os.path.isfile(COMP_BVP_PATH):
    mae, rmse, max_diff = compare_bvp_files(bvp_path, COMP_BVP_PATH)
    if mae is not None:
        print("\n" + "=" * 60)
        print("BVP comparison complete!")
        print("=" * 60)
else:
    print(f"\nReference file not found: {COMP_BVP_PATH}")
    print("Skipping comparison. To compare, ensure comp_BVP.mat exists in the script directory.")

# %%
