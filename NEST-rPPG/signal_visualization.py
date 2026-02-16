"""
signal_visualization: Visualize signals from .mat files (BVP, Wave, HR, etc.)
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
    """Guess signal sampling rate (256 samples → 60 Hz for PURE)."""
    n = np.size(arr)
    return 60.0 if n == 256 else 256.0


# %%
def plot_signal(signal_1d, fs=None, ax=None, title="", label=None, color=None):
    """
    Plot a 1D signal with time axis.
    
    Parameters:
    -----------
    signal_1d : array-like, shape (n_samples,)
        1D signal to plot
    fs : float, optional
        Sampling frequency [Hz]. If None, inferred from length.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title
    label : str, optional
        Line label for legend
    color : str or tuple, optional
        Line color
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes used for plotting
    """
    signal_1d = np.asarray(signal_1d, dtype=float).ravel()
    n_samples = signal_1d.size
    fs = fs if fs is not None else infer_sampling_hz(signal_1d)
    t = np.arange(n_samples) / fs
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    
    ax.plot(t, signal_1d, label=label, color=color, alpha=0.8)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if label:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_multiple_signals(signals_dict, fs=None, figsize=(12, 6), sharex=True):
    """
    Plot multiple signals in subplots.
    
    Parameters:
    -----------
    signals_dict : dict
        {label: signal_array} pairs to plot
    fs : float, optional
        Sampling frequency [Hz]
    figsize : tuple, optional
        Figure size (width, height)
    sharex : bool, optional
        Share x-axis across subplots
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    n_signals = len(signals_dict)
    fig, axes = plt.subplots(n_signals, 1, figsize=figsize, sharex=sharex)
    if n_signals == 1:
        axes = [axes]
    
    for idx, (label, signal) in enumerate(signals_dict.items()):
        plot_signal(signal, fs=fs, ax=axes[idx], title=label, label=label)
    
    plt.tight_layout()
    return fig, axes


# %%
def visualize_mat_file(mat_path, keys=None, fs=None, max_segments=5, figsize=(12, 4)):
    """
    Load one .mat file and plot selected signal keys.
    
    Parameters:
    -----------
    mat_path : str
        Path to .mat file
    keys : list of str, optional
        Keys to visualize. If None, plots all arrays.
    fs : float, optional
        Sampling frequency [Hz]
    max_segments : int, optional
        Max number of segments to plot if signal is 2D
    figsize : tuple, optional
        Figure size per key
    
    Returns:
    --------
    raw_signals : dict
        {key: array} of loaded signals
    """
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
        for i in range(n_plot):
            label = f"seg {i}" if n_plot > 1 else None
            plot_signal(v[i, :], fs=_fs, ax=ax, title=f"{base} — {key}", label=label)
        plt.tight_layout()
        plt.show()
    
    return raw_signals


def visualize_mat_dir(dir_path, pattern="*.mat", keys=None, fs=None, max_files=4, max_segments=2):
    """
    Load and visualize multiple .mat files in a directory.
    
    Parameters:
    -----------
    dir_path : str
        Directory containing .mat files
    pattern : str, optional
        Glob pattern to match files (e.g., "*gt_Wave.mat")
    keys : list of str, optional
        Keys to visualize in each file
    fs : float, optional
        Sampling frequency [Hz]
    max_files : int, optional
        Maximum number of files to process
    max_segments : int, optional
        Max segments per file
    """
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
def plot_signal_distribution(dir_path=None, pattern="*gt_Wave.mat", key="Wave", 
                            bins=100, fs_for_info=None, figsize=(8, 4)):
    """
    Aggregate all values from 'key' in matching .mat files and plot histogram.
    
    Parameters:
    -----------
    dir_path : str, optional
        Directory containing .mat files. If None, uses Wave_sort/PURE.
    pattern : str, optional
        Glob pattern to match files
    key : str, optional
        Key in .mat files to aggregate
    bins : int, optional
        Number of histogram bins
    fs_for_info : float, optional
        Sampling frequency for title (informational)
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    all_vals : array
        Concatenated values from all files
    """
    if dir_path is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(BASE_DIR, "Wave_sort", "PURE")
    
    dir_path = os.path.abspath(dir_path)
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    
    if not files:
        print("No files", pattern, "in", dir_path)
        return None
    
    all_vals = []
    for f in files:
        data = load_mat(f)
        if key not in data:
            continue
        v = np.asarray(data[key], dtype=float).ravel()
        all_vals.append(v)
    
    if not all_vals:
        print("No key", key, "found in any", pattern, "under", dir_path)
        return None
    
    all_vals = np.concatenate(all_vals)
    print(f"Collected {all_vals.size} samples from {len(files)} files.")
    
    plt.figure(figsize=figsize)
    plt.hist(all_vals, bins=bins, density=False, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    title = f"Distribution of '{key}' over {len(files)} {pattern} files"
    if fs_for_info is not None:
        title += f" (fs={fs_for_info} Hz)"
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return all_vals


def compare_signals(signal1, signal2, fs=None, labels=("Signal 1", "Signal 2"), 
                   figsize=(12, 4)):
    """
    Plot two signals on the same axes for comparison.
    
    Parameters:
    -----------
    signal1, signal2 : array-like
        1D signals to compare
    fs : float, optional
        Sampling frequency [Hz]
    labels : tuple of str, optional
        Labels for the two signals
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_signal(signal1, fs=fs, ax=ax, label=labels[0], color='C0')
    plot_signal(signal2, fs=fs, ax=ax, label=labels[1], color='C1')
    ax.set_title(f"Comparison: {labels[0]} vs {labels[1]}")
    plt.tight_layout()
    return fig, ax


# %%
# Config: edit paths and options here
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mat_path = os.path.join(BASE_DIR, "Wave_sort", "PURE", "10001gt_Wave.mat")
dir_path = os.path.join(BASE_DIR, "Wave_sort", "PURE")
keys = ["Wave"]
fs = 30.0
max_segments = 2
max_files = 3

# %%
# List variables in a .mat file
list_mat_keys(mat_path)

# %%
# Visualize one .mat file; returns raw signals {key: array}
raw_signals = visualize_mat_file(mat_path, keys=keys, fs=fs, max_segments=max_segments)

# %%
# Visualize distribution of all Wave values in Wave_sort (histogram)
plot_signal_distribution(dir_path=dir_path, pattern="*gt_Wave.mat", 
                         key="Wave", bins=100, fs_for_info=fs)

# %%
# Visualize all .mat files in a directory
visualize_mat_dir(dir_path, pattern="*gt_Wave.mat", keys=keys, fs=fs, 
                  max_files=max_files, max_segments=max_segments)
