"""
mat_visualization: Read and visualize .mat files and example STMaps.
Edit variables in the config cell (# %%) and run cells.
"""

# %%
import os
import glob
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cv2

# %%
def load_mat(path):
    """Load .mat file; return dict of arrays (no __* keys)."""
    data = scio.loadmat(path)
    return {k: data[k] for k in data if not k.startswith("__")}


def list_mat_keys(path):
    """Return variable names in a .mat file."""
    return list(load_mat(path).keys())


def infer_sampling_hz(arr):
    """Guess BVP sampling rate (256 samples → 60 Hz for PURE)."""
    n = np.size(arr)
    return 60.0 if n == 256 else 256.0


def plot_array(arr, ax=None, fs=None, title=""):
    """Plot 1D or 2D array with optional time axis."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_seg, n_samp = arr.shape
    fs = fs if fs is not None else infer_sampling_hz(arr)
    t = np.arange(n_samp) / fs
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 3))
    for i in range(n_seg):
        label = f"seg {i}" if n_seg > 1 else None
        ax.plot(t, arr[i, :], label=label)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if n_seg > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def visualize_mat_file(mat_path, keys=None, fs=None, max_segments=5, figsize=(12, 4)):
    """Load one .mat and plot selected keys. Returns dict of raw arrays {key: array}."""
    mat_path = os.path.abspath(mat_path)
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(mat_path)
    data = load_mat(mat_path)
    if not data:
        print("No arrays in", mat_path)
        return None
    keys = keys if keys is not None else list(data.keys())
    keys = [k for k in keys if k in data]
    if not keys:
        print("Available:", list(data.keys()))
        return None
    base = os.path.splitext(os.path.basename(mat_path))[0]
    raw_signals = {}
    for key in keys:
        v = np.squeeze(data[key]).copy()
        if np.ndim(v) == 0:
            print(key, "=", float(v))
            raw_signals[key] = np.array(float(v))
            continue
        if np.ndim(v) == 1:
            v = v.reshape(1, -1)
        raw_signals[key] = v
        n_plot = min(v.shape[0], max_segments)
        _fs = fs if fs is not None else infer_sampling_hz(v)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_array(v[:n_plot, :], ax=ax, fs=_fs, title=f"{base} — {key}")
        plt.tight_layout()
        plt.show()
    return raw_signals


def visualize_mat_dir(dir_path, pattern="*.mat", keys=None, fs=None, max_files=4, max_segments=2):
    """Load .mat files in directory and plot."""
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(dir_path)
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))[:max_files]
    if not files:
        print("No files", pattern, "in", dir_path)
        return
    for f in files:
        visualize_mat_file(f, keys=keys, fs=fs, max_segments=max_segments)


# %%
def load_stmap(path, bgr_to_rgb=False):
    """
    Load an STMap PNG. Returns (H, W, C) with H=regions, W=time, C=BGR.
    If bgr_to_rgb=True, convert to RGB for display.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def visualize_stmap(stmap_path, bgr_to_rgb=True, fs=30.0):
    """
    Load and visualize one STMap with original width/height ratio.
    Shape (H, W, C): H=regions, W=time, C=channels. Single axes, aspect='equal'
    so display ratio matches original (W wide × H tall). No title or axis labels.
    """
    stmap = load_stmap(stmap_path, bgr_to_rgb=bgr_to_rgb)
    H, W, C = stmap.shape
    # One figure, original aspect ratio (width:height = W:H); base size 10 inches on longer side
    scale = 10.0 / max(W, H, 1)
    fig, ax = plt.subplots(1, 1, figsize=(W * scale, H * scale))
    ax.imshow(stmap, aspect="equal", interpolation="nearest")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return stmap


def read_example_stmaps(stmap_paths, bgr_to_rgb=True, fs=30.0):
    """
    Read and visualize a list of example STMap paths (one figure per STMap, original ratio).
    """
    for path in stmap_paths:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            print("Skip (not found):", path)
            continue
        print("Reading:", path)
        visualize_stmap(path, bgr_to_rgb=bgr_to_rgb, fs=fs)


# %%
# Config: edit paths and options here
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mat_path = os.path.join(BASE_DIR, "Wave_sort", "PURE", "10001gt_Wave.mat")
# dir_path = os.path.join(BASE_DIR, "Wave_sort", "PURE")
keys = ["Wave"]
fs = 60.0
max_segments = 2
max_files = 3

# Example STMap paths (edit to match your STMap layout)
STMAP_DIR = os.path.join(BASE_DIR, "STMap")
example_stmap_paths = [
    os.path.join(STMAP_DIR, "PURE", "10003", "STMap", "STMap.png"),
    os.path.join(STMAP_DIR, "UBFC", "subject5", "STMap", "STMap.png"),
]

# %%
# List variables in the .mat file
list_mat_keys(mat_path)

# %%
# Visualize one .mat file; returns raw visualized mat signals {key: array}
raw_mat_signals = visualize_mat_file(mat_path, keys=keys, fs=fs, max_segments=max_segments)

# %%
# Read and visualize example STMaps
read_example_stmaps(example_stmap_paths, bgr_to_rgb=True, fs=30.0)

# %%
# Or visualize all .mat files in a directory (set dir_path in config and uncomment)
# visualize_mat_dir(dir_path, pattern="*gt_Wave.mat", keys=keys, fs=fs, max_files=max_files, max_segments=max_segments)

# %%
# Visualize distribution of all Wave values in Wave_sort (using plt.hist)
def plot_wave_distribution(dir_path=None, pattern="*gt_Wave.mat", key="Wave", bins=100, fs_for_info=None):
    """
    Aggregate all values from 'key' in matching .mat files under dir_path and
    plot a 1D histogram using plt.hist.
    """
    if dir_path is None:
        dir_path = os.path.join(BASE_DIR, "Wave_sort", "PURE")
    dir_path = os.path.abspath(dir_path)
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    if not files:
        print("No files", pattern, "in", dir_path)
        return
    all_vals = []
    for f in files:
        data = load_mat(f)
        if key not in data:
            continue
        v = np.asarray(data[key], dtype=float).ravel()
        all_vals.append(v)
    if not all_vals:
        print("No key", key, "found in any", pattern, "under", dir_path)
        return
    all_vals = np.concatenate(all_vals)
    print("Collected", all_vals.size, "samples from", len(files), "files.")
    plt.figure(figsize=(8, 4))
    plt.hist(all_vals, bins=bins, density=False, alpha=0.7)
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    title = f"Distribution of '{key}' over {len(files)} {pattern} files"
    if fs_for_info is not None:
        title += f" (fs={fs_for_info} Hz)"
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage (PURE gt_Wave.mat files):
# plot_wave_distribution(os.path.join(BASE_DIR, "Wave_sort", "PURE"),
#                        pattern="*gt_Wave.mat", key="Wave", bins=100, fs_for_info=fs)

# %%
plot_wave_distribution(os.path.join(BASE_DIR, "Wave_sort", "PURE"),
                       pattern="*gt_Wave.mat",
                       key="Wave",
                       bins=100,
                       fs_for_info=fs)
# %%
