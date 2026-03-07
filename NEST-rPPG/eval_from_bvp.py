"""
Evaluates BVP model performance from Wave_sort .mat files (gt/pr pairs):
heart rate only, from FFT on raw segments. Reports ME, Std, MAE, RMSE, MER, Pearson r.
When run this script, use interpreter: mprppg
"""

# %%
import os
import numpy as np
import scipy.io as scio
from scipy import signal as scipy_signal
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Optional

import config

FS_BVP = 30  # BVP sampling rate [Hz] in .mat (raw segments)
HR_FREQ_LOW, HR_FREQ_HIGH = 0.7, 3.0  # HR band [Hz] for FFT peak


# %%
def bpfilter64(sig: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass filter [0.8, 3] Hz using FIR (Hamming), filtfilt."""
    n = max(round(len(sig) / 10), 1)
    b = scipy_signal.firwin(n + 1, [0.8, 3.0], pass_zero=False, fs=fs)
    return scipy_signal.filtfilt(b, 1.0, sig)


def hr_from_fft(signal_1d: np.ndarray, fs: float = FS_BVP) -> float:
    """
    Heart rate from FFT: bandpass filter, then peak frequency in [HR_FREQ_LOW, HR_FREQ_HIGH] Hz.
    Returns HR in BPM, or np.nan if signal too short or no valid peak.
    """
    signal_1d = np.asarray(signal_1d, dtype=float).ravel()
    n = len(signal_1d)
    if n < 64:
        return np.nan
    filtered = bpfilter64(signal_1d, fs)
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-12)
    # FFT (real signal -> rfft)
    fft_vals = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    power = np.abs(fft_vals) ** 2
    mask = (freqs >= HR_FREQ_LOW) & (freqs <= HR_FREQ_HIGH)
    if not np.any(mask):
        return np.nan
    idx_max = np.argmax(power[mask])
    f_peak = freqs[mask][idx_max]
    return float(f_peak * 60.0)  # Hz -> BPM


def my_eval(hr_pr: np.ndarray, hr_gt: np.ndarray) -> tuple:
    """Matches MyEval.m: me, E_std, mae, rmse, mer, p (Pearson r)."""
    hr_pr = np.asarray(hr_pr).ravel()
    hr_gt = np.asarray(hr_gt).ravel()
    n = len(hr_pr)
    if n != len(hr_gt) or n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    temp = hr_pr - hr_gt
    valid = ~(np.isnan(hr_pr) | np.isnan(hr_gt))
    n_valid = np.sum(valid)
    if n_valid == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # ME: mean error
    me = float(np.nanmean(temp)) if n_valid > 0 else np.nan
    # E_std: std of error
    e_std = float(np.nanstd(temp, ddof=0)) if n_valid > 1 else (0.0 if n_valid == 1 else np.nan)
    # MAE: mean absolute error (use nanmean to handle NaNs properly)
    mae = float(np.nanmean(np.abs(temp))) if n_valid > 0 else np.nan
    # RMSE: root mean squared error
    rmse = float(np.sqrt(np.nanmean(temp * temp))) if n_valid > 0 else np.nan
    # MER: mean absolute relative error
    mer = np.abs(temp) / (hr_gt + 0.01)
    mer = float(np.nanmean(mer)) if n_valid > 0 else np.nan
    # Pearson correlation
    if n_valid < 2:
        p = np.nan
    else:
        p = float(np.corrcoef(hr_pr[valid], hr_gt[valid])[0, 1])
    return me, e_std, mae, rmse, mer, p

# %%
def run_eval(save_path: str) -> dict:
    """
    Load gt/pr .mat pairs, compute FFT heart rate per segment (no interpolation, fs=FS_BVP).
    Average HR per subject, then compute ME, Std, MAE, RMSE, MER, Pearson r.
    save_path: directory with '*gt_Wave.mat' and '*pr_Wave.mat' pairs (e.g. Wave_sort/PURE).
    """
    save_path = os.path.abspath(save_path)
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"Directory not found: {save_path}")
    all_files = sorted([f for f in os.listdir(save_path) if f.endswith(".mat")])
    gt_files = [f for f in all_files if "gt_Wave" in f]
    pr_files = [f for f in all_files if "pr_Wave" in f]
    def subject_id(s):
        return s.replace("gt_Wave.mat", "").replace("pr_Wave.mat", "").strip()
    gt_by_id = {subject_id(f): f for f in gt_files}
    pr_by_id = {subject_id(f): f for f in pr_files}
    common_ids = sorted(set(gt_by_id) & set(pr_by_id))
    if not common_ids:
        raise FileNotFoundError(f"No gt/pr .mat pairs found in {save_path}")

    hr_pr_per_subject = []
    hr_gt_per_subject = []
    print(common_ids)
    print(f"Found {len(common_ids)} subject pairs. Computing FFT HR only (fs={FS_BVP} Hz, no interpolation)...")
    for sid in tqdm(common_ids, desc="Subjects", unit="subject"):
        gt_mat = scio.loadmat(os.path.join(save_path, gt_by_id[sid]))
        pr_mat = scio.loadmat(os.path.join(save_path, pr_by_id[sid]))
        Wave_gt = np.squeeze(gt_mat["Wave"])
        Wave_pr = np.squeeze(pr_mat["Wave"])
        if Wave_gt.ndim == 1:
            Wave_gt = Wave_gt[None, :]
        if Wave_pr.ndim == 1:
            Wave_pr = Wave_pr[None, :]
        num = Wave_gt.shape[0]
        if num == 0:
            print(f"Warning: Subject {sid} has no segments, skipping...")
            continue
        hr_pr_list = []
        hr_gt_list = []
        for n in range(num):
            wave_pr_one = np.asarray(Wave_pr[n, :], dtype=float)
            wave_gt_one = np.asarray(Wave_gt[n, :], dtype=float)
            hr_pr_list.append(hr_from_fft(wave_pr_one, fs=FS_BVP))
            hr_gt_list.append(hr_from_fft(wave_gt_one, fs=FS_BVP))
        with np.errstate(invalid="ignore"):
            hr_pr_per_subject.append(np.nanmean(hr_pr_list))
            hr_gt_per_subject.append(np.nanmean(hr_gt_list))

    if len(hr_pr_per_subject) == 0:
        raise ValueError("No valid data found - all subjects have empty arrays")

    hr_pr = np.array(hr_pr_per_subject)
    hr_gt = np.array(hr_gt_per_subject)
    me, e_std, mae, rmse, mer, p = my_eval(hr_pr, hr_gt)
    result = {"HR": {"ME": me, "Std": e_std, "MAE": mae, "RMSE": rmse, "MER": mer, "r": p}}
    return result


def visualize_mat_waves(
    save_path: str,
    subject_ids: Optional[List[str]] = None,
    segment_indices: Optional[List[int]] = None,
    max_segments_per_subject: int = 3,
    figsize: tuple = (12, 4),
    fs: float = 30.0,
    vis_run_name: Optional[str] = None,
):
    """
    Visualize BVP waves from Wave_sort .mat files (gt vs predicted).
    Returns a list of pairs of raw arrays: each item is {"subject_id", "segment", "gt", "pred"}
    with "gt" and "pred" as 1D numpy arrays.

    Args:
        save_path: Directory with *gt_Wave.mat and *pr_Wave.mat pairs (e.g. Wave_sort/PURE).
        subject_ids: List of subject ids to plot (e.g. ['01-01', '01-02']). If None, plot first 5 subjects.
        segment_indices: Which segment index(es) to plot per subject (e.g. [0, 1]). If None, plot first max_segments_per_subject.
        max_segments_per_subject: Max number of segments to show per subject when segment_indices is None.
        figsize: (width, height) per subplot figure.
        fs: BVP sampling frequency [Hz] for time axis (e.g. 30 for .mat).
        vis_run_name: Optional name for first-level subfolder under 'vis/'. If provided,
            per-subject folders are created at save_path/vis/<vis_run_name>/<subject_id>/.
    """
    save_path = os.path.abspath(save_path)
    all_files = [f for f in os.listdir(save_path) if f.endswith(".mat")]
    gt_files = [f for f in all_files if "gt_Wave" in f]
    pr_files = [f for f in all_files if "pr_Wave" in f]

    def _subject_id(s):
        return s.replace("gt_Wave.mat", "").replace("pr_Wave.mat", "").strip()

    gt_by_id = {_subject_id(f): f for f in gt_files}
    pr_by_id = {_subject_id(f): f for f in pr_files}
    common = sorted(set(gt_by_id) & set(pr_by_id))
    if not common:
        raise FileNotFoundError(f"No gt/pr .mat pairs in {save_path}")

    if subject_ids is None:
        subject_ids = common[:5]
    else:
        subject_ids = [s for s in subject_ids if s in common]

    pairs_raw: List[dict] = []  # list of {"subject_id", "segment", "gt", "pred"}
    vis_root = os.path.join(save_path, "vis") if vis_run_name is not None else None

    # Progress bar over subjects
    for sid in tqdm(subject_ids, desc="Visualizing subjects"):
        gt_mat = scio.loadmat(os.path.join(save_path, gt_by_id[sid]))
        pr_mat = scio.loadmat(os.path.join(save_path, pr_by_id[sid]))
        Wave_gt = np.squeeze(gt_mat["Wave"])
        Wave_pr = np.squeeze(pr_mat["Wave"])
        if Wave_gt.ndim == 1:
            Wave_gt = Wave_gt[None, :]
        if Wave_pr.ndim == 1:
            Wave_pr = Wave_pr[None, :]
        num_seg = Wave_gt.shape[0]
        if segment_indices is None:
            segs = list(range(min(max_segments_per_subject, num_seg)))
        else:
            segs = [i for i in segment_indices if 0 <= i < num_seg]
        if not segs:
            continue
        # Ensure visualization directory for this subject exists if requested
        if vis_root is not None:
            subject_vis_dir = os.path.join(vis_root, vis_run_name, sid)
            os.makedirs(subject_vis_dir, exist_ok=True)
        n_plots = len(segs)
        fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        n_samples = Wave_gt.shape[1]
        t = np.arange(n_samples) / fs
        for ax, seg in zip(axes, segs):
            gt_arr = np.asarray(Wave_gt[seg, :], dtype=float).ravel()
            pred_arr = np.asarray(Wave_pr[seg, :], dtype=float).ravel()
            pairs_raw.append({
                "subject_id": sid,
                "segment": seg,
                "gt": gt_arr,
                "pred": pred_arr,
            })
            hr_gt = hr_from_fft(gt_arr, fs=FS_BVP)
            hr_pr = hr_from_fft(pred_arr, fs=FS_BVP)
            label_gt = f"GT ({hr_gt:.1f} BPM)" if np.isfinite(hr_gt) else "GT (— BPM)"
            label_pr = f"Pred ({hr_pr:.1f} BPM)" if np.isfinite(hr_pr) else "Pred (— BPM)"
            ax.plot(t, gt_arr, label=label_gt, color="C0", alpha=0.8)
            ax.plot(t, pred_arr, label=label_pr, color="C1", alpha=0.8)
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")
            ax.set_title(f"Subject {sid} — segment {seg}")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"BVP waves: subject {sid}", y=1.02)
        plt.tight_layout()
        # Save figure into vis/<vis_run_name>/<subject_id>/ if requested
        if vis_root is not None:
            subject_vis_dir = os.path.join(vis_root, vis_run_name, sid)
            os.makedirs(subject_vis_dir, exist_ok=True)
            seg_str = "_".join(str(s) for s in segs)
            fig_path = os.path.join(subject_vis_dir, f"segments_{seg_str}.png")
            fig.savefig(fig_path, dpi=150)
        plt.show()

    return pairs_raw





# %%
# Config: set path to Wave_sort directory (gt/pr .mat pairs)
# Use config parameters so this stays in sync with train.py.
run_name = config.build_run_name()  # e.g. rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1_lossCM
save_path = os.path.join(config.WAVE_SORT_ROOT, config.TGT_DOMAIN, run_name)
# Name for first-level visualization subfolder under save_path/vis/.
# Final structure: Wave_sort/<TGT_DOMAIN>/vis/<VIS_RUN_NAME>/<subject_id>/

# %%
# Optional: visualize waves before evaluation (uncomment and run)
pairs = visualize_mat_waves(
    save_path,
    segment_indices=[3, 5],
    vis_run_name=config.LOSS_TYPE,
)
# %%
import numpy as np
from scipy.signal import welch

# signal: 1D numpy array
# fs: sampling frequency (Hz)
signal = pairs[5]['pred']
f, psd = welch(
    signal,
    fs=30,
    window='hann',
    noverlap=None,
    detrend='constant'
)

plt.plot(signal)
plt.show()
plt.plot(f, psd)
plt.show()

# Heart-rate frequency band
band = (f >= 0.7) & (f <= 4.0)

f_hr = f[band]
psd_hr = psd[band]

# Peak frequency → HR estimate
peak_freq = f_hr[np.argmax(psd_hr)]
hr_bpm = peak_freq * 60

print(f"Estimated HR: {hr_bpm:.2f} bpm")
# %%
result = run_eval(save_path)

# %%
# Print table: ME, Std, MAE, RMSE, MER, Pearson r
print("Feature    \tME\t\tStd\t\tMAE\t\tRMSE\t\tMER\t\tr")
print("-" * 90)
for name, metrics in result.items():
    print(
        f"{name:10}\t{metrics['ME']:.6f}\t{metrics['Std']:.6f}\t{metrics['MAE']:.6f}\t"
        f"{metrics['RMSE']:.6f}\t{metrics['MER']:.6f}\t{metrics['r']:.6f}"
    )

# %%
