# %%
"""
Compare Label/RGB_lmk.csv with comp_RGB_lmk.csv.
Both files: one row per frame, 136 values per row (68 landmarks x 2).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_LABEL = os.path.join(SCRIPT_DIR, "Label", "RGB_lmk.csv")
PATH_COMP = os.path.join(SCRIPT_DIR, "comp_RGB_lmk.csv")

# Load both CSVs
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
# %%
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
