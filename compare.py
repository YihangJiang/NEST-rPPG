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

# Paths (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# %%
# ===== Part 1: Compare landmark CSVs (NEST-rPPG/STMap/BUAA vs BUAA_my) =====
print("=" * 60)
print("PART 1: Comparing landmark CSVs (BUAA original vs BUAA_my)")
print("=" * 60)

STMAP_BUAA_ROOT = os.path.join(SCRIPT_DIR, "NEST-rPPG", "STMap", "BUAA")
BUAA_MY_ROOT = os.path.join(SCRIPT_DIR, "BUAA_my")


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

# Iterate over all BUAA STMap folders and compare with matching BUAA_my folders
if not os.path.isdir(STMAP_BUAA_ROOT):
    print("STMap BUAA root not found:", STMAP_BUAA_ROOT)
else:
    buaa_subs = sorted(os.listdir(STMAP_BUAA_ROOT))
    for name in buaa_subs:
        stmap_label = os.path.join(STMAP_BUAA_ROOT, name, "Label", "RGB_lmk.csv")
        my_label = os.path.join(BUAA_MY_ROOT, name, "Label", "RGB_lmk.csv")
        if not os.path.isfile(stmap_label):
            continue
        if not os.path.isfile(my_label):
            print("\n---", name, "---")
            print("Missing BUAA_my label:", my_label)
            continue
        compare_lmk_pair(stmap_label, my_label, tag=name)

# %%
# ===== Part 2: Compare BVP .mat files =====
print("\n" + "=" * 60)
print("PART 2: Comparing BVP .mat files")
print("=" * 60)

PATH_PPGDATA = os.path.join(SCRIPT_DIR, "PPGData.mat")
PATH_COMP_BVP = os.path.join(SCRIPT_DIR, "comp_BVP.mat")

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


