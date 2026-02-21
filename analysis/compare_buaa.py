# %%
"""
Compare landmark CSVs and BVP .mat files.
- Label/RGB_lmk.csv vs comp_RGB_lmk.csv (landmarks)
- PPGData.mat vs comp_BVP.mat (BVP signals)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cv2

# Paths: script lives in analysis/, project root is parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# %%
# ===== Part 1: Compare landmark CSVs (NEST-rPPG/STMap/BUAA vs BUAA_my) =====
print("=" * 60)
print("PART 1: Comparing landmark CSVs (NEST-rPPG/STMap/BUAA vs BUAA_my)")
print("=" * 60)

STMAP_BUAA_ROOT = os.path.join(PROJECT_ROOT, "NEST-rPPG", "STMap", "BUAA")
BUAA_MY_ROOT = os.path.join(PROJECT_ROOT, "STMap_my", "BUAA_my")


def load_lmk_csv(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]
            rows.append(vals)
    return np.array(rows, dtype=float)


def compare_lmk_pair(path_a, path_b, tag):
    print("\n---", tag, "---")
    print("A:", path_a)
    print("B:", path_b)
    lmk_a = load_lmk_csv(path_a)
    lmk_b = load_lmk_csv(path_b)

    n_a, n_b = lmk_a.shape[0], lmk_b.shape[0]
    print(f"Row counts: A={n_a}, B={n_b}")

    n_common = min(n_a, n_b)
    lmk_a_c = lmk_a[:n_common]
    lmk_b_c = lmk_b[:n_common]

    diff = np.abs(lmk_a_c - lmk_b_c)

    exact_matches = np.sum(np.all(diff == 0, axis=1))
    print(f"Exactly matching rows: {exact_matches}/{n_common} ({100.0*exact_matches/n_common:.2f}%)")

    mae_per_frame = np.mean(diff, axis=1)
    mae_overall = float(np.mean(diff))
    max_diff = float(np.max(diff))

    print(f"  MAE overall: {mae_overall:.6f}")
    print(f"  Max diff:    {max_diff:.6f}")
    print(f"  MAE per frame: min={mae_per_frame.min():.6f}, max={mae_per_frame.max():.6f}, mean={mae_per_frame.mean():.6f}")

# Iterate over all folders and compare landmarks between NEST-rPPG/STMap/BUAA and BUAA_my
if not os.path.isdir(STMAP_BUAA_ROOT):
    print("STMap BUAA root not found:", STMAP_BUAA_ROOT)
elif not os.path.isdir(BUAA_MY_ROOT):
    print("BUAA_my root not found:", BUAA_MY_ROOT)
else:
    # Get all folders from both directories
    stmap_folders = set()
    if os.path.isdir(STMAP_BUAA_ROOT):
        for name in os.listdir(STMAP_BUAA_ROOT):
            if not name.startswith('.') and os.path.isdir(os.path.join(STMAP_BUAA_ROOT, name)):
                stmap_folders.add(name)

    my_folders = set()
    if os.path.isdir(BUAA_MY_ROOT):
        for name in os.listdir(BUAA_MY_ROOT):
            if not name.startswith('.') and os.path.isdir(os.path.join(BUAA_MY_ROOT, name)):
                my_folders.add(name)

    # Compare folders that exist in both
    common_folders = sorted(stmap_folders & my_folders)
    print(f"\nFound {len(common_folders)} common folders to compare")

    for name in common_folders:
        stmap_label = os.path.join(STMAP_BUAA_ROOT, name, "Label", "RGB_lmk.csv")
        my_label = os.path.join(BUAA_MY_ROOT, name, "Label", "RGB_lmk.csv")
        if os.path.isfile(stmap_label) and os.path.isfile(my_label):
            compare_lmk_pair(stmap_label, my_label, tag=f"{name} (NEST-rPPG/STMap/BUAA vs BUAA_my)")
        elif not os.path.isfile(stmap_label):
            print(f"\n--- {name} ---")
            print(f"Missing NEST-rPPG/STMap/BUAA label: {stmap_label}")
        elif not os.path.isfile(my_label):
            print(f"\n--- {name} ---")
            print(f"Missing BUAA_my label: {my_label}")

    # Report folders only in one directory
    only_stmap = sorted(stmap_folders - my_folders)
    only_my = sorted(my_folders - stmap_folders)
    if only_stmap:
        print(f"\nFolders only in NEST-rPPG/STMap/BUAA ({len(only_stmap)}): {only_stmap[:10]}{'...' if len(only_stmap) > 10 else ''}")
    if only_my:
        print(f"\nFolders only in BUAA_my ({len(only_my)}): {only_my[:10]}{'...' if len(only_my) > 10 else ''}")

# %%
# ===== Part 2: Compare BVP .mat files =====
print("\n" + "=" * 60)
print("PART 2: Comparing BVP .mat files")
print("=" * 60)

PATH_PPGDATA = os.path.join(PROJECT_ROOT, "PPGData.mat")
PATH_COMP_BVP = os.path.join(PROJECT_ROOT, "comp_BVP.mat")

if os.path.isfile(PATH_PPGDATA) and os.path.isfile(PATH_COMP_BVP):
    print("Loading", PATH_PPGDATA)
    ppg_data = scio.loadmat(PATH_PPGDATA)
    # Extract raw data from structured array: PPG is (fs, data, peaks), we want 'data'
    ppg_struct = ppg_data['PPG']
    if isinstance(ppg_struct, np.ndarray) and ppg_struct.dtype.names:
        # Structured array: extract 'data' field
        ppg_signal = np.squeeze(ppg_struct['data'][0, 0]).astype(float)
    else:
        # Simple array
        ppg_signal = np.squeeze(ppg_struct).astype(float)

    print("Loading", PATH_COMP_BVP)
    comp_bvp_data = scio.loadmat(PATH_COMP_BVP)
    comp_bvp_signal = np.squeeze(comp_bvp_data['BVP']).astype(float)

    n_ppg = len(ppg_signal)
    n_comp_bvp = len(comp_bvp_signal)
    print(f"\nSignal lengths: PPGData={n_ppg}, comp_BVP={n_comp_bvp}")

    # Compare over common length
    n_common_bvp = min(n_ppg, n_comp_bvp)
    ppg_c = ppg_signal[:n_common_bvp]
    comp_bvp_c = comp_bvp_signal[:n_common_bvp]

    diff_bvp = np.abs(ppg_c - comp_bvp_c)

    # Statistics
    mae_bvp = np.mean(diff_bvp)
    max_diff_bvp = np.max(diff_bvp)
    rmse_bvp = np.sqrt(np.mean(diff_bvp ** 2))

    print(f"\nComparison over first {n_common_bvp} samples:")
    print(f"  Mean absolute error (MAE):  {mae_bvp:.6f}")
    print(f"  Root mean square error (RMSE): {rmse_bvp:.6f}")
    print(f"  Max absolute difference:    {max_diff_bvp:.6f}")
else:
    print("Skipping Part 2: PPGData.mat or comp_BVP.mat not found in project root.")

# %%
# ===== Part 3: Compare STMap PNG images =====
print("\n" + "=" * 60)
print("PART 3: Comparing STMap PNG images")
print("=" * 60)

def compare_two_images(path_a, path_b):
    """
    Load two images, resize second to first if needed,
    and compute MSE, MAE, and max absolute difference.
    Returns: (img_a, img_b, mse, mae, max_diff, h, w)
    """
    a = cv2.imread(path_a)
    b = cv2.imread(path_b)
    if a is None:
        raise FileNotFoundError(f'Could not load image A: {path_a}')
    if b is None:
        raise FileNotFoundError(f'Could not load image B: {path_b}')

    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    diff = np.abs(a_f - b_f)

    mse = np.mean((a_f - b_f) ** 2)
    mae = np.mean(diff)
    max_diff = np.max(diff)
    h, w = a.shape[:2]
    return a, b, mse, mae, max_diff, h, w


def compare_stmap_pair(path_a, path_b, tag):
    """Compare two STMap images and print statistics."""
    print("\n---", tag, "---")
    print("A:", path_a)
    print("B:", path_b)
    try:
        a_img, b_img, mse, mae, max_diff, h, w = compare_two_images(path_a, path_b)
        print(f"  Size: {w}x{h}")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Max diff (per channel): {max_diff:.4f}")
        return mse, mae, max_diff
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return None, None, None


# Compare STMaps: NEST-rPPG/STMap/BUAA vs BUAA_my
if os.path.isdir(STMAP_BUAA_ROOT) and os.path.isdir(BUAA_MY_ROOT):
    # Get common folders
    stmap_folders = set()
    for name in os.listdir(STMAP_BUAA_ROOT):
        if not name.startswith('.') and os.path.isdir(os.path.join(STMAP_BUAA_ROOT, name)):
            stmap_folders.add(name)

    my_folders = set()
    for name in os.listdir(BUAA_MY_ROOT):
        if not name.startswith('.') and os.path.isdir(os.path.join(BUAA_MY_ROOT, name)):
            my_folders.add(name)

    common_folders = sorted(stmap_folders & my_folders)
    stmap_count = 0

    for name in common_folders:
        stmap_stmap = os.path.join(STMAP_BUAA_ROOT, name, "STMap", "STMap_RGB.png")
        my_stmap = os.path.join(BUAA_MY_ROOT, name, "STMap", "STMap_RGB.png")

        if os.path.isfile(stmap_stmap) and os.path.isfile(my_stmap):
            compare_stmap_pair(stmap_stmap, my_stmap, tag=f"{name} (NEST-rPPG/STMap/BUAA vs BUAA_my)")
            stmap_count += 1
        elif not os.path.isfile(stmap_stmap):
            print(f"\n--- {name} ---")
            print(f"  STMap found in BUAA_my: {my_stmap}")
            print(f"  Missing in NEST-rPPG/STMap/BUAA: {stmap_stmap}")
        elif not os.path.isfile(my_stmap):
            print(f"\n--- {name} ---")
            print(f"  STMap found in NEST-rPPG/STMap/BUAA: {stmap_stmap}")
            print(f"  Missing in BUAA_my: {my_stmap}")

    if stmap_count == 0:
        print("No STMap pairs found for comparison.")
    else:
        print(f"\nCompared {stmap_count} STMap pairs.")

# %%
# ===== Part 4: Compare BVP label .mat files (NEST-rPPG/STMap/BUAA vs BUAA_my) =====
print("\n" + "=" * 60)
print("PART 4: Comparing BVP label .mat files (NEST-rPPG/STMap/BUAA vs BUAA_my)")
print("=" * 60)

def compare_bvp_mat_pair(path_a, path_b, tag, signal_name='BVP'):
    """Compare two BVP .mat files and print statistics. Returns (mae, rmse, max_diff, signal_a_c, signal_b_c)."""
    print("\n---", tag, "---")
    print("A:", path_a)
    print("B:", path_b)

    try:
        data_a = scio.loadmat(path_a)
        data_b = scio.loadmat(path_b)

        if signal_name not in data_a or signal_name not in data_b:
            print(f"  Error: '{signal_name}' key not found in one or both files")
            return None, None, None, None, None

        signal_a = np.squeeze(data_a[signal_name]).astype(float)
        signal_b = np.squeeze(data_b[signal_name]).astype(float)

        n_a, n_b = len(signal_a), len(signal_b)
        print(f"  Signal lengths: A={n_a}, B={n_b}")

        n_common = min(n_a, n_b)
        signal_a_c = signal_a[:n_common]
        signal_b_c = signal_b[:n_common]

        diff = np.abs(signal_a_c - signal_b_c)
        mae = np.mean(diff)
        rmse = np.sqrt(np.mean(diff ** 2))
        max_diff = np.max(diff)

        print(f"  Comparison over first {n_common} samples:")
        print(f"    MAE:  {mae:.6f}")
        print(f"    RMSE: {rmse:.6f}")
        print(f"    Max diff: {max_diff:.6f}")

        return mae, rmse, max_diff, signal_a_c, signal_b_c
    except Exception as e:
        print(f"  Error: {e}")
        return None, None, None, None, None


def plot_bvp_comparison(signal_a, signal_b, subject_name, signal_type, save_dir):
    """Create two plots: overlay of both signals, and difference plot."""
    n = len(signal_a)
    t = np.arange(n)
    
    # Figure 1: Overlay plot
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, signal_a, label='NEST-rPPG/STMap/BUAA', alpha=0.7, linewidth=1)
    ax1.plot(t, signal_b, label='BUAA_my', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Frame index')
    ax1.set_ylabel('BVP value')
    ax1.set_title(f'{subject_name} - {signal_type} Overlay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    overlay_path = os.path.join(save_dir, f'{subject_name}_{signal_type}_overlay.png')
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Figure 2: Difference plot
    diff = np.abs(signal_a - signal_b)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(t, diff, label='Absolute difference', color='red', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Frame index')
    ax2.set_ylabel('|A - B|')
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    ax2.set_title(f'{subject_name} - {signal_type} Difference (MAE={mae:.6f}, RMSE={rmse:.6f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    diff_path = os.path.join(save_dir, f'{subject_name}_{signal_type}_diff.png')
    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    return overlay_path, diff_path


# Compare BVP.mat and BVP_Filt.mat files between NEST-rPPG/STMap/BUAA and BUAA_my
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'buaa_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

if os.path.isdir(STMAP_BUAA_ROOT) and os.path.isdir(BUAA_MY_ROOT):
    # Get common folders
    stmap_folders = set()
    for name in os.listdir(STMAP_BUAA_ROOT):
        if not name.startswith('.') and os.path.isdir(os.path.join(STMAP_BUAA_ROOT, name)):
            stmap_folders.add(name)

    my_folders = set()
    for name in os.listdir(BUAA_MY_ROOT):
        if not name.startswith('.') and os.path.isdir(os.path.join(BUAA_MY_ROOT, name)):
            my_folders.add(name)

    common_folders = sorted(stmap_folders & my_folders)
    bvp_count = 0
    bvp_filt_count = 0

    for name in common_folders:
        stmap_label_dir = os.path.join(STMAP_BUAA_ROOT, name, "Label")
        my_label_dir = os.path.join(BUAA_MY_ROOT, name, "Label")

        # Compare BVP.mat
        stmap_bvp = os.path.join(stmap_label_dir, "BVP.mat")
        my_bvp = os.path.join(my_label_dir, "BVP.mat")
        if os.path.isfile(stmap_bvp) and os.path.isfile(my_bvp):
            mae, rmse, max_d, sig_a, sig_b = compare_bvp_mat_pair(
                stmap_bvp, my_bvp, tag=f"{name} (BVP.mat: NEST-rPPG/STMap/BUAA vs BUAA_my)", signal_name='BVP')
            if sig_a is not None and sig_b is not None:
                overlay_path, diff_path = plot_bvp_comparison(sig_a, sig_b, name, 'BVP', FIGURES_DIR)
                print(f"    Saved: {overlay_path}")
                print(f"    Saved: {diff_path}")
            bvp_count += 1

        # Compare BVP_Filt.mat
        stmap_bvp_filt = os.path.join(stmap_label_dir, "BVP_Filt.mat")
        my_bvp_filt = os.path.join(my_label_dir, "BVP_Filt.mat")
        if os.path.isfile(stmap_bvp_filt) and os.path.isfile(my_bvp_filt):
            mae, rmse, max_d, sig_a, sig_b = compare_bvp_mat_pair(
                stmap_bvp_filt, my_bvp_filt, tag=f"{name} (BVP_Filt.mat: NEST-rPPG/STMap/BUAA vs BUAA_my)", signal_name='BVP')
            if sig_a is not None and sig_b is not None:
                overlay_path, diff_path = plot_bvp_comparison(sig_a, sig_b, name, 'BVP_Filt', FIGURES_DIR)
                print(f"    Saved: {overlay_path}")
                print(f"    Saved: {diff_path}")
            bvp_filt_count += 1

    print(f"\nCompared {bvp_count} BVP.mat pairs and {bvp_filt_count} BVP_Filt.mat pairs.")
    print(f"Figures saved to: {FIGURES_DIR}")

    if bvp_count == 0 and bvp_filt_count == 0:
        print("No BVP label pairs found for comparison.")

print("\n" + "=" * 60)
print("All comparisons complete!")
print("=" * 60)


# %%
