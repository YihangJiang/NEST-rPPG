"""
met_visualization: Read and visualize contents of .mat files.
Edit variables in the config cell (# %%) and run cells.
"""

# %%
import os
import glob
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

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
    """Load one .mat and plot selected keys."""
    mat_path = os.path.abspath(mat_path)
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(mat_path)
    data = load_mat(mat_path)
    if not data:
        print("No arrays in", mat_path)
        return
    keys = keys if keys is not None else list(data.keys())
    keys = [k for k in keys if k in data]
    if not keys:
        print("Available:", list(data.keys()))
        return
    base = os.path.splitext(os.path.basename(mat_path))[0]
    for key in keys:
        v = np.squeeze(data[key])
        if np.ndim(v) == 0:
            print(key, "=", float(v))
            continue
        if np.ndim(v) == 1:
            v = v.reshape(1, -1)
        n_plot = min(v.shape[0], max_segments)
        _fs = fs if fs is not None else infer_sampling_hz(v)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_array(v[:n_plot, :], ax=ax, fs=_fs, title=f"{base} — {key}")
        plt.tight_layout()
        plt.show()


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
# Config: edit paths and options here
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mat_path = os.path.join(BASE_DIR, "Wave_sort", "PURE", "10001gt_Wave.mat")
# dir_path = os.path.join(BASE_DIR, "Wave_sort", "PURE")
keys = ["Wave"]
fs = 60.0
max_segments = 2
max_files = 3

# %%
# List variables in the .mat file
list_mat_keys(mat_path)

# %%
# Visualize one .mat file
visualize_mat_file(mat_path, keys=keys, fs=fs, max_segments=max_segments)

# %%
# Or visualize all .mat files in a directory (set dir_path in config and uncomment)
# visualize_mat_dir(dir_path, pattern="*gt_Wave.mat", keys=keys, fs=fs, max_files=max_files, max_segments=max_segments)
