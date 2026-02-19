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
# ===== Part 1: Compare landmark CSVs =====
print("=" * 60)
print("PART 1: Comparing landmark CSVs")
print("=" * 60)

PATH_LABEL = os.path.join(SCRIPT_DIR, "Label", "RGB_lmk.csv")
PATH_COMP = os.path.join(SCRIPT_DIR, "comp_RGB_lmk.csv")

def load_lmk_csv(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]
            rows.append(vals)
    return np.array(rows)

print("Loading", PATH_LABEL)
lmk_label = load_lmk_csv(PATH_LABEL)
print("Loading", PATH_COMP)
lmk_comp = load_lmk_csv(PATH_COMP)

n_label, n_comp = lmk_label.shape[0], lmk_comp.shape[0]
print(f"\nRow counts: Label={n_label}, comp={n_comp}")

# Compare over common rows
n_common = min(n_label, n_comp)
lmk_label_c = lmk_label[:n_common]
lmk_comp_c = lmk_comp[:n_common]

diff = np.abs(lmk_label_c.astype(float) - lmk_comp_c.astype(float))

# Count exactly matching rows (all 136 values match)
exact_matches = np.sum(np.all(diff == 0, axis=1))
print(f"\nExactly matching rows: {exact_matches}/{n_common} ({100.0*exact_matches/n_common:.2f}%)")

# Per-frame: mean absolute error (over 136 values)
mae_per_frame = np.mean(diff, axis=1)
# Per-value (over all frames): mean and max
mae_overall = np.mean(diff)
max_diff = np.max(diff)

print(f"\nComparison over first {n_common} frames:")
print(f"  Mean absolute error (overall): {mae_overall:.6f}")
print(f"  Max absolute difference:       {max_diff:.6f}")
print(f"  MAE per frame: min={mae_per_frame.min():.6f}, max={mae_per_frame.max():.6f}, mean={mae_per_frame.mean():.6f}")

# Plot: MAE per frame
plt.figure(figsize=(10, 4))
plt.plot(mae_per_frame, label="MAE per frame (136 coords)")
plt.xlabel("Frame index")
plt.ylabel("MAE")
plt.title("Label/RGB_lmk.csv vs comp_RGB_lmk.csv")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "compare_lmk_mae_per_frame.png"), dpi=150)
print(f"\nSaved: compare_lmk_mae_per_frame.png")
plt.show()

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

# Plot: signals overlay
plt.figure(figsize=(12, 5))
t = np.arange(n_common_bvp)
plt.plot(t, ppg_c, label="PPGData.mat (PPG)", alpha=0.7, linewidth=1)
plt.plot(t, comp_bvp_c, label="comp_BVP.mat (BVP)", alpha=0.7, linewidth=1)
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.title(f"BVP signal comparison (first {n_common_bvp} samples)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "compare_bvp_signals.png"), dpi=150)
print(f"\nSaved: compare_bvp_signals.png")
plt.show()

# Plot: difference
plt.figure(figsize=(12, 4))
plt.plot(t, diff_bvp, label="Absolute difference", color='red', alpha=0.7, linewidth=1)
plt.xlabel("Sample index")
plt.ylabel("|PPG - BVP|")
plt.title(f"BVP difference (MAE={mae_bvp:.6f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "compare_bvp_diff.png"), dpi=150)
print(f"Saved: compare_bvp_diff.png")
plt.show()

# %%
