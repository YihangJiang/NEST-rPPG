# %%
"""
Compare landmark CSVs and BVP .mat files between STMap/UBFC and STMap_my/UBFC_my.
"""
import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

# Paths: script lives in analysis/, project root is parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# %%
# ===== Part 1: Compare landmark CSVs (STMap/UBFC vs STMap_my/UBFC_my) =====
print("=" * 60)
print("PART 1: Comparing landmark CSVs (STMap/UBFC vs STMap_my/UBFC_my)")
print("=" * 60)

STMAP_UBFC_ROOT = os.path.join(PROJECT_ROOT, "STMap", "UBFC")
UBFC_MY_ROOT = os.path.join(PROJECT_ROOT, "STMap_my", "UBFC_my")


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


# Iterate over all folders and compare landmarks between STMap/UBFC and STMap_my/UBFC_my
if not os.path.isdir(STMAP_UBFC_ROOT):
    print("STMap UBFC root not found:", STMAP_UBFC_ROOT)
elif not os.path.isdir(UBFC_MY_ROOT):
    print("UBFC_my root not found:", UBFC_MY_ROOT)
else:
    # Get all folders from both directories
    stmap_folders = set()
    for name in os.listdir(STMAP_UBFC_ROOT):
        if not name.startswith('.') and os.path.isdir(os.path.join(STMAP_UBFC_ROOT, name)):
            stmap_folders.add(name)

    my_folders = set()
    for name in os.listdir(UBFC_MY_ROOT):
        if not name.startswith('.') and os.path.isdir(os.path.join(UBFC_MY_ROOT, name)):
            my_folders.add(name)

    # Compare folders that exist in both
    common_folders = sorted(stmap_folders & my_folders)
    print(f"\nFound {len(common_folders)} common folders to compare")

    for name in common_folders:
        stmap_label = os.path.join(STMAP_UBFC_ROOT, name, "Label", "RGB_lmk.csv")
        my_label = os.path.join(UBFC_MY_ROOT, name, "Label", "RGB_lmk.csv")
        if os.path.isfile(stmap_label) and os.path.isfile(my_label):
            compare_lmk_pair(stmap_label, my_label, tag=f"{name} (STMap/UBFC vs UBFC_my)")
        elif not os.path.isfile(stmap_label):
            print(f"\n--- {name} ---")
            print(f"Missing STMap/UBFC label: {stmap_label}")
        elif not os.path.isfile(my_label):
            print(f"\n--- {name} ---")
            print(f"Missing UBFC_my label: {my_label}")

    # Report folders only in one directory
    only_stmap = sorted(stmap_folders - my_folders)
    only_my = sorted(my_folders - stmap_folders)
    if only_stmap:
        print(f"\nFolders only in STMap/UBFC ({len(only_stmap)}): {only_stmap[:10]}{'...' if len(only_stmap) > 10 else ''}")
    if only_my:
        print(f"\nFolders only in UBFC_my ({len(only_my)}): {only_my[:10]}{'...' if len(only_my) > 10 else ''}")

# %%
# ===== Part 2: Compare BVP label .mat files (STMap/UBFC vs STMap_my/UBFC_my) =====
print("\n" + "=" * 60)
print("PART 2: Comparing BVP label .mat files (STMap/UBFC vs STMap_my/UBFC_my)")
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
    ax1.plot(t, signal_a, label='STMap/UBFC', alpha=0.7, linewidth=1)
    ax1.plot(t, signal_b, label='UBFC_my', alpha=0.7, linewidth=1)
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


# Compare BVP.mat and BVP_Filt.mat files between STMap/UBFC and UBFC_my
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'ubfc_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

if os.path.isdir(STMAP_UBFC_ROOT) and os.path.isdir(UBFC_MY_ROOT):
    # Get common folders
    stmap_folders = set()
    for name in os.listdir(STMAP_UBFC_ROOT):
        if not name.startswith('.') and os.path.isdir(os.path.join(STMAP_UBFC_ROOT, name)):
            stmap_folders.add(name)

    my_folders = set()
    for name in os.listdir(UBFC_MY_ROOT):
        if not name.startswith('.') and os.path.isdir(os.path.join(UBFC_MY_ROOT, name)):
            my_folders.add(name)

    common_folders = sorted(stmap_folders & my_folders)
    bvp_count = 0
    bvp_filt_count = 0

    for name in common_folders:
        stmap_label_dir = os.path.join(STMAP_UBFC_ROOT, name, "Label")
        my_label_dir = os.path.join(UBFC_MY_ROOT, name, "Label")

        # Compare BVP.mat
        stmap_bvp = os.path.join(stmap_label_dir, "BVP.mat")
        my_bvp = os.path.join(my_label_dir, "BVP.mat")
        if os.path.isfile(stmap_bvp) and os.path.isfile(my_bvp):
            mae, rmse, max_d, sig_a, sig_b = compare_bvp_mat_pair(
                stmap_bvp, my_bvp, tag=f"{name} (BVP.mat: STMap/UBFC vs UBFC_my)", signal_name='BVP')
            if sig_a is not None and sig_b is not None:
                overlay_path, diff_path = plot_bvp_comparison(sig_a, sig_b, name, 'BVP', FIGURES_DIR)
                print(f"    Saved: {overlay_path}")
                print(f"    Saved: {diff_path}")
            bvp_count += 1

        # Compare UBFC BVP.mat vs UBFC_my BVP_Filt.mat
        stmap_bvp = os.path.join(stmap_label_dir, "BVP.mat")
        my_bvp_filt = os.path.join(my_label_dir, "BVP_Filt.mat")
        if os.path.isfile(stmap_bvp) and os.path.isfile(my_bvp_filt):
            mae, rmse, max_d, sig_a, sig_b = compare_bvp_mat_pair(
                stmap_bvp, my_bvp_filt,
                tag=f"{name} (UBFC BVP.mat vs UBFC_my BVP_Filt.mat)", signal_name='BVP')
            if sig_a is not None and sig_b is not None:
                overlay_path, diff_path = plot_bvp_comparison(sig_a, sig_b, name, 'UBFC_BVP_vs_my_BVP_Filt', FIGURES_DIR)
                print(f"    Saved: {overlay_path}")
                print(f"    Saved: {diff_path}")
            bvp_filt_count += 1

    print(f"\nCompared {bvp_count} BVP.mat pairs and {bvp_filt_count} (UBFC BVP vs UBFC_my BVP_Filt) pairs.")
    print(f"Figures saved to: {FIGURES_DIR}")

    if bvp_count == 0 and bvp_filt_count == 0:
        print("No BVP label pairs found for comparison.")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)

# %%
