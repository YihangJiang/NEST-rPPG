# %%
"""
Compare STMap image *content* (not just names) between STMap/UBFC and STMap_my/UBFC_my.
Reports shape, frame count, and pixel-level differences; saves pixel value distribution comparison figures.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
NEST_UBFC_ROOT = os.path.join(PROJECT_ROOT, "STMap", "UBFC")
UBFC_MY_ROOT = os.path.join(PROJECT_ROOT, "STMap_my", "UBFC_my")


def load_stmap(subject_root, prefer_name="STMap.png"):
    """Load STMap from subject_root/STMap/ (try STMap.png then STMap_RGB.png). Return (array, path) or (None, None)."""
    stmap_dir = os.path.join(subject_root, "STMap")
    for name in (prefer_name, "STMap_RGB.png", "STMap.png"):
        path = os.path.join(stmap_dir, name)
        if os.path.isfile(path):
            img = cv2.imread(path)
            if img is not None:
                return img, path
    return None, None


def plot_pixel_distribution(nest_img, my_img, subject_name, save_dir, w_common):
    """
    Plot pixel value distribution comparison (histograms per channel) for NEST vs UBFC_my.
    Uses first w_common frames from both. Saves to save_dir/<subject>_pixel_distribution.png.
    """
    nest_c = nest_img[:, :w_common, :].astype(np.float64)
    my_c = my_img[:, :w_common, :].astype(np.float64)
    ch_names = ["B", "G", "R"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    bins = np.linspace(0, 256, 51)
    for ch, (ax, label) in enumerate(zip(axes, ch_names)):
        ax.hist(nest_c[:, :, ch].ravel(), bins=bins, alpha=0.6, label="NEST UBFC", color="C0", density=True)
        ax.hist(my_c[:, :, ch].ravel(), bins=bins, alpha=0.6, label="UBFC_my", color="C1", density=True)
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Density")
        ax.set_title(f"Channel {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{subject_name} STMap pixel distribution (first {w_common} frames)")
    plt.tight_layout()
    safe_name = subject_name.replace("/", "-")
    out_path = os.path.join(save_dir, f"{safe_name}_pixel_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# %%
print("=" * 60)
print("STMap content comparison: NEST UBFC vs UBFC_my")
print("=" * 60)

nest_subjects = sorted(
    [d for d in os.listdir(NEST_UBFC_ROOT) if not d.startswith(".") and os.path.isdir(os.path.join(NEST_UBFC_ROOT, d))]
)
my_subjects = sorted(
    [d for d in os.listdir(UBFC_MY_ROOT) if not d.startswith(".") and os.path.isdir(os.path.join(UBFC_MY_ROOT, d))]
)
common = sorted(set(nest_subjects) & set(my_subjects))
print(f"Common subjects: {len(common)} (NEST: {len(nest_subjects)}, UBFC_my: {len(my_subjects)})")

FIGURES_DIR = os.path.join(SCRIPT_DIR, "ubfc_stmap_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Distribution figures will be saved to: {FIGURES_DIR}\n")

# STMap layout: (H=25 spatial, W=time/frames, C=3). Compare over overlapping time.
for name in common:
    nest_path = os.path.join(NEST_UBFC_ROOT, name)
    my_path = os.path.join(UBFC_MY_ROOT, name)
    nest_img, nest_f = load_stmap(nest_path)
    my_img, my_f = load_stmap(my_path)
    if nest_img is None:
        print(f"\n{name}: NEST STMap not found")
        continue
    if my_img is None:
        print(f"\n{name}: UBFC_my STMap not found")
        continue

    # Shapes: (25, T, 3)
    h_n, w_n, c_n = nest_img.shape
    h_m, w_m, c_m = my_img.shape
    print(f"\n--- {name} ---")
    print(f"  NEST:    shape={nest_img.shape} (H={h_n}, time={w_n}, C={c_n})")
    print(f"  UBFC_my: shape={my_img.shape} (H={h_m}, time={w_m}, C={c_m})")

    if (h_n, c_n) != (h_m, c_m):
        print("  -> Spatial/channel mismatch (different grid or channels).")
        continue
    # Same height (25) and channels (3); compare over common time length
    w = min(w_n, w_m)
    # Pixel distribution comparison figure
    dist_path = plot_pixel_distribution(nest_img, my_img, name, FIGURES_DIR, w)
    print(f"  Saved: {dist_path}")

    nest_c = nest_img[:, :w, :].astype(np.float64)
    my_c = my_img[:, :w, :].astype(np.float64)
    diff = np.abs(nest_c - my_c)
    mae = np.mean(diff)
    max_diff = np.max(diff)
    # Correlation per channel (flatten spatial and time)
    corr = []
    for ch in range(3):
        a, b = nest_c[:, :, ch].ravel(), my_c[:, :, ch].ravel()
        if np.std(a) > 0 and np.std(b) > 0:
            corr.append(np.corrcoef(a, b)[0, 1])
        else:
            corr.append(np.nan)
    print(f"  Over first {w} frames: MAE={mae:.4f}, max_diff={max_diff:.1f}, corr(R,G,B)={[f'{c:.4f}' for c in corr]}")
    if w_n != w_m:
        print(f"  -> Different frame count (NEST={w_n}, UBFC_my={w_m}) -> different STMap width.")
    if mae > 1.0:
        print("  -> Large pixel difference: content differs (likely different alignment or getValue type).")

print("\n" + "=" * 60)
print("Summary: STMap content differs because input frames and/or pipeline differ.")
print("  - Different frame count -> different width.")
print("  - Different alignment (NEST vs Align_Face_UBFC) or getValue (type=1 vs type=2) -> different pixel values.")
print(f"  Pixel distribution figures saved to: {FIGURES_DIR}")
print("=" * 60)

# %%
