# %%
# Plot the BUAA PPG/BVP waveform for subject 12.
#
# By default, all available `Sub_12lux ...` recordings are plotted using the
# filtered reference signal in BVP_Filt.mat. Set RAW = True to plot the raw
# BVP.mat signal instead, or set CONDITION to a specific illumination folder
# name (e.g. "Sub_12lux 63.1") to plot just one recording.
#
# Run cell-by-cell in Jupyter / VS Code interactive window, or:
#   python figures/plot_buaa_subject12_ppg.py

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "STMap" / "BUAA"
DEFAULT_OUTPUT = SCRIPT_DIR / "buaa_subject12_ppg.png"
FS_HZ = 30.0  # BUAA videos are sampled at 60 frames per second.

# %%
# Configuration
DATA_ROOT = DEFAULT_DATA_ROOT
OUTPUT = DEFAULT_OUTPUT
CONDITION: str | None = "Sub_12lux 63.1"  # e.g. "Sub_12lux 63.1"; None plots all conditions
RAW = False  # plot BVP.mat instead of BVP_Filt.mat


# %%
def load_bvp(path: Path) -> np.ndarray:
    """Load and flatten the BVP array from a MATLAB file."""
    data = loadmat(path)
    if "BVP" not in data:
        raise KeyError(f"{path} does not contain a 'BVP' variable")
    signal = np.asarray(data["BVP"], dtype=float).squeeze()
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"Expected a non-empty 1-D BVP signal in {path}")
    return signal


def plot_subject12_ppg(
    data_root: Path = DATA_ROOT,
    output: Path = OUTPUT,
    condition: str | None = CONDITION,
    raw: bool = RAW,
) -> Path:
    pattern = "Sub_12lux *"
    conditions = sorted(data_root.glob(pattern))
    if condition:
        conditions = [data_root / condition]
    conditions = [path for path in conditions if path.is_dir()]
    if not conditions:
        raise FileNotFoundError(f"No subject-12 recordings found under {data_root}")

    filename = "BVP.mat" if raw else "BVP_Filt.mat"
    fig, axes = plt.subplots(len(conditions), 1, figsize=(11, 2.4 * len(conditions)), squeeze=False)
    axes = axes[:, 0]

    for ax, cond_path in zip(axes, conditions):
        signal_path = cond_path / "Label" / filename
        signal = load_bvp(signal_path)
        print(f"{cond_path.name}: {filename} samples = {signal.size}")
        time = np.arange(signal.size) / FS_HZ
        ax.plot(time, signal, color="#1769aa", linewidth=0.8)
        ax.set_title(cond_path.name)
        ax.set_ylabel("PPG amplitude")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"BUAA Subject 12 PPG waveform ({'raw' if raw else 'filtered'})", y=0.995)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output} ({len(conditions)} recording(s), {filename})")
    return output


# %%
if __name__ == "__main__":
    plot_subject12_ppg()

# %%
