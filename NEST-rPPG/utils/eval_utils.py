#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Evaluation utilities for BVP models based on Wave_sort .mat files.

Functions moved from eval_from_bvp.py so they can be reused:
- bpfilter64
- hr_from_fft
- my_eval
- run_eval
- visualize_mat_waves
"""
import os
import csv
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import scipy.io as scio
from scipy import signal as scipy_signal
from tqdm import tqdm
import matplotlib.pyplot as plt


FS_BVP = 30  # BVP sampling rate [Hz] in .mat (raw segments)
HR_FREQ_LOW, HR_FREQ_HIGH = 0.7, 3.0  # HR band [Hz] for FFT peak


def bpfilter64(sig: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass filter [0.8, 3] Hz using FIR (Hamming), filtfilt."""
    sig = np.asarray(sig, dtype=float).ravel()
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


def run_eval(
    save_path: str,
    *,
    return_details: bool = False,
) -> Dict[str, Dict[str, Any]] | Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
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

    def subject_id(s: str) -> str:
        return s.replace("gt_Wave.mat", "").replace("pr_Wave.mat", "").strip()

    gt_by_id = {subject_id(f): f for f in gt_files}
    pr_by_id = {subject_id(f): f for f in pr_files}
    common_ids = sorted(set(gt_by_id) & set(pr_by_id))
    if not common_ids:
        raise FileNotFoundError(f"No gt/pr .mat pairs found in {save_path}")

    hr_pr_per_subject: List[float] = []
    hr_gt_per_subject: List[float] = []
    details: Dict[str, Any] = {
        "save_path": save_path,
        "subjects": [],  # list of per-subject dicts (with per-segment arrays)
        "rows": [],      # flat rows for CSV writing (subject_id, segment_idx, hr_gt, hr_pr, err)
    }
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
        hr_pr_list: List[float] = []
        hr_gt_list: List[float] = []
        for n in range(num):
            wave_pr_one = np.asarray(Wave_pr[n, :], dtype=float)
            wave_gt_one = np.asarray(Wave_gt[n, :], dtype=float)
            hr_pr_n = hr_from_fft(wave_pr_one, fs=FS_BVP)
            hr_gt_n = hr_from_fft(wave_gt_one, fs=FS_BVP)
            hr_pr_list.append(hr_pr_n)
            hr_gt_list.append(hr_gt_n)
            details["rows"].append({
                "subject_id": sid,
                "segment_idx": int(n),
                "hr_gt_bpm": float(hr_gt_n) if np.isfinite(hr_gt_n) else np.nan,
                "hr_pr_bpm": float(hr_pr_n) if np.isfinite(hr_pr_n) else np.nan,
                "err_bpm": float(hr_pr_n - hr_gt_n) if (np.isfinite(hr_pr_n) and np.isfinite(hr_gt_n)) else np.nan,
            })
        with np.errstate(invalid="ignore"):
            subj_hr_pr = float(np.nanmean(hr_pr_list))
            subj_hr_gt = float(np.nanmean(hr_gt_list))
            hr_pr_per_subject.append(subj_hr_pr)
            hr_gt_per_subject.append(subj_hr_gt)

        details["subjects"].append({
            "subject_id": sid,
            "hr_gt_segments_bpm": np.array(hr_gt_list, dtype=float),
            "hr_pr_segments_bpm": np.array(hr_pr_list, dtype=float),
            "err_segments_bpm": np.array(hr_pr_list, dtype=float) - np.array(hr_gt_list, dtype=float),
            "mean_err_bpm": float(np.nanmean(np.array(hr_pr_list) - np.array(hr_gt_list))),
            "median_err_bpm": float(np.nanmedian(np.array(hr_pr_list) - np.array(hr_gt_list))),
            "mean_hr_gt_bpm": subj_hr_gt,
            "mean_hr_pr_bpm": subj_hr_pr,
        })

    if len(hr_pr_per_subject) == 0:
        raise ValueError("No valid data found - all subjects have empty arrays")

    hr_pr = np.array(hr_pr_per_subject)
    hr_gt = np.array(hr_gt_per_subject)
    me, e_std, mae, rmse, mer, p = my_eval(hr_pr, hr_gt)
    result = {"HR": {"ME": me, "Std": e_std, "MAE": mae, "RMSE": rmse, "MER": mer, "r": p}}
    if return_details:
        details["hr_gt_subject_mean_bpm"] = hr_gt
        details["hr_pr_subject_mean_bpm"] = hr_pr
        details["err_subject_mean_bpm"] = hr_pr - hr_gt
        return result, details
    return result


def plot_subject_error_bars(
    details: Dict[str, Any],
    *,
    top_k: Optional[int] = None,
    sort_by: str = "mean_abs",  # mean_abs | mean | median
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
):
    """
    Bar plot per subject (gt/pr pair) error summaries based on segment-level errors.

    Plots two bars per subject:
    - mean error over segments
    - median error over segments
    """
    subjects = list(details.get("subjects", []))
    if not subjects:
        raise ValueError("details has no subjects. Call run_eval(..., return_details=True) first.")

    def key_fn(s):
        mean_e = float(s.get("mean_err_bpm", np.nan))
        med_e = float(s.get("median_err_bpm", np.nan))
        if sort_by == "mean":
            return abs(mean_e) if np.isfinite(mean_e) else float("inf")
        if sort_by == "median":
            return abs(med_e) if np.isfinite(med_e) else float("inf")
        # default: mean_abs
        return abs(mean_e) if np.isfinite(mean_e) else float("inf")

    subjects = sorted(subjects, key=key_fn, reverse=True)
    if top_k is not None:
        subjects = subjects[: int(top_k)]

    ids = [s["subject_id"] for s in subjects]
    mean_err = np.array([s.get("mean_err_bpm", np.nan) for s in subjects], dtype=float)
    med_err = np.array([s.get("median_err_bpm", np.nan) for s in subjects], dtype=float)

    x = np.arange(len(ids))
    w = 0.42
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(x - w / 2, mean_err, width=w, label="Mean error (BPM)")
    ax.bar(x + w / 2, med_err, width=w, label="Median error (BPM)")
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=60, ha="right")
    ax.set_ylabel("Error (Pred - GT) [BPM]")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title(title or "Per-subject error summary (segment HR)")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, ax


def write_segment_errors_csv(details: Dict[str, Any], csv_path: str) -> str:
    """
    Write per-segment HR and error to CSV.

    CSV columns:
    - subject_id
    - segment_idx
    - hr_gt_bpm
    - hr_pr_bpm
    - err_bpm
    """
    rows = list(details.get("rows", []))
    if not rows:
        raise ValueError("details has no rows. Call run_eval(..., return_details=True) first.")

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = ["subject_id", "segment_idx", "hr_gt_bpm", "hr_pr_bpm", "err_bpm"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return os.path.abspath(csv_path)


def plot_worst_subject_from_segment_csv(
    segment_csv_path: str,
    *,
    metric: str = "max_abs",  # max_abs | mean_abs | median_abs
    save_fig_path: Optional[str] = None,
    show: bool = True,
):
    """
    Select the subject(folder) with the highest absolute error and plot it.

    Uses segment-level errors from `segment_errors.csv` generated by `write_segment_errors_csv`.

    Args:
        segment_csv_path: Path to segment_errors.csv
        metric:
          - 'max_abs': max(|err_bpm|) across segments
          - 'mean_abs': mean(|err_bpm|) across segments
          - 'median_abs': median(|err_bpm|) across segments
        save_fig_path: optional PNG path to save the figure
        show: whether to plt.show()

    Returns:
        (best_subject_id, best_score)
    """
    rows = []
    with open(segment_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"Empty CSV: {segment_csv_path}")

    # Group rows by subject_id
    grouped: Dict[str, List[dict]] = {}
    for r in rows:
        sid = r["subject_id"]
        grouped.setdefault(sid, []).append(r)

    def _score(err_list: List[float]) -> float:
        arr = np.asarray(err_list, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return -np.inf
        abs_err = np.abs(arr)
        if metric == "max_abs":
            return float(np.max(abs_err))
        if metric == "mean_abs":
            return float(np.mean(abs_err))
        if metric == "median_abs":
            return float(np.median(abs_err))
        raise ValueError(f"Unknown metric: {metric}")

    best_subject = None
    best_score = -np.inf
    for sid, rs in grouped.items():
        err_vals = [float(x["err_bpm"]) for x in rs if x.get("err_bpm", "") != ""]
        sc = _score(err_vals)
        if sc > best_score:
            best_score = sc
            best_subject = sid

    if best_subject is None:
        raise ValueError("Could not determine worst subject (all scores were -inf).")

    best_rows = sorted(grouped[best_subject], key=lambda x: int(x["segment_idx"]))
    seg_idx = np.array([int(x["segment_idx"]) for x in best_rows], dtype=int)
    hr_gt = np.array([float(x["hr_gt_bpm"]) for x in best_rows], dtype=float)
    hr_pr = np.array([float(x["hr_pr_bpm"]) for x in best_rows], dtype=float)
    err = np.array([float(x["err_bpm"]) for x in best_rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"Worst subject by {metric}: {best_subject} (score={best_score:.4g} BPM)",
        y=0.98,
    )

    ax1.plot(seg_idx, hr_gt, "o-", label="GT HR (bpm)", alpha=0.9, markersize=4)
    ax1.plot(seg_idx, hr_pr, "o-", label="Pred HR (bpm)", alpha=0.9, markersize=4)
    ax1.set_ylabel("HR (bpm)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.plot(seg_idx, err, "s-", label="Error (Pred - GT) [bpm]", alpha=0.9, markersize=4)
    ax2.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("segment_idx")
    ax2.set_ylabel("err_bpm")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_fig_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_fig_path)), exist_ok=True)
        fig.savefig(save_fig_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return best_subject, best_score


def plot_worst_subject_signals(
    save_path: str,
    segment_csv_path: str,
    *,
    num_segments: int = 20,
    metric: str = "max_abs",  # max_abs | mean_abs | median_abs
    worst_subject_id: Optional[str] = None,
    save_fig_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Plot GT and Pred BVP signals for segments of the worst subject.

    Selection:
    - worst subject is chosen by abs error (max_abs by default) using segment_errors.csv
    - then the top `num_segments` segments with highest abs(err_bpm) for that subject
      are plotted (GT + Pred waveforms).

    Args:
      save_path: Wave_sort run directory that contains:
        <subject_id>gt_Wave.mat and <subject_id>pr_Wave.mat
      segment_csv_path: .../feature/segment_errors.csv
      num_segments: number of segments to plot
      worst_subject_id: if provided, skip searching and use this subject id
      save_fig_path: optional PNG path for saving
      show: whether to plt.show()
    """
    # Load CSV rows
    rows: List[dict] = []
    with open(segment_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError(f"Empty CSV: {segment_csv_path}")

    grouped: Dict[str, List[dict]] = {}
    for r in rows:
        sid = r["subject_id"]
        grouped.setdefault(sid, []).append(r)

    def _score_subject(rs: List[dict]) -> float:
        err_vals = [float(x["err_bpm"]) for x in rs if x.get("err_bpm", "") != "" and np.isfinite(float(x["err_bpm"]))]
        if not err_vals:
            return -np.inf
        abs_err = np.abs(np.asarray(err_vals, dtype=float))
        if metric == "max_abs":
            return float(np.max(abs_err))
        if metric == "mean_abs":
            return float(np.mean(abs_err))
        if metric == "median_abs":
            return float(np.median(abs_err))
        raise ValueError(f"Unknown metric: {metric}")

    if worst_subject_id is None:
        best_sid = None
        best_score = -np.inf
        for sid, rs in grouped.items():
            sc = _score_subject(rs)
            if sc > best_score:
                best_score = sc
                best_sid = sid
        worst_subject_id = best_sid
    if worst_subject_id is None or worst_subject_id not in grouped:
        raise ValueError("Could not determine worst_subject_id from CSV.")

    subj_rows = grouped[worst_subject_id]
    # Pick top segments by abs(err)
    seg_candidates = []  # list of (seg_idx, err_bpm, hr_gt_bpm, hr_pr_bpm)
    for r in subj_rows:
        try:
            seg_i = int(r["segment_idx"])
            err_i = float(r["err_bpm"])
            hr_gt_i = float(r.get("hr_gt_bpm", "nan"))
            hr_pr_i = float(r.get("hr_pr_bpm", "nan"))
        except Exception:
            continue
        if np.isfinite(err_i):
            seg_candidates.append((seg_i, err_i, hr_gt_i, hr_pr_i))
    if not seg_candidates:
        raise ValueError(f"No finite err_bpm values for subject: {worst_subject_id}")

    seg_candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    chosen = seg_candidates[: int(num_segments)]
    seg_idx = [s for s, _e, _gt, _pr in chosen]
    chosen_err = {s: _e for s, _e, _gt, _pr in chosen}
    chosen_hr_gt = {s: _gt for s, _e, _gt, _pr in chosen}
    chosen_hr_pr = {s: _pr for s, _e, _gt, _pr in chosen}

    # Load waveforms for this subject
    save_path = os.path.abspath(save_path)
    gt_path = os.path.join(save_path, f"{worst_subject_id}gt_Wave.mat")
    pr_path = os.path.join(save_path, f"{worst_subject_id}pr_Wave.mat")
    if not os.path.isfile(gt_path) or not os.path.isfile(pr_path):
        # Fallback: search by prefix + suffix
        all_mats = [f for f in os.listdir(save_path) if f.endswith(".mat")]
        gt_matches = [f for f in all_mats if f.startswith(worst_subject_id) and f.endswith("gt_Wave.mat")]
        pr_matches = [f for f in all_mats if f.startswith(worst_subject_id) and f.endswith("pr_Wave.mat")]
        if not gt_matches or not pr_matches:
            raise FileNotFoundError(
                f"Could not find gt/pr mat for subject_id={worst_subject_id} under {save_path}"
            )
        gt_path = os.path.join(save_path, gt_matches[0])
        pr_path = os.path.join(save_path, pr_matches[0])

    gt_mat = scio.loadmat(gt_path)
    pr_mat = scio.loadmat(pr_path)
    Wave_gt = np.squeeze(gt_mat["Wave"])
    Wave_pr = np.squeeze(pr_mat["Wave"])
    if Wave_gt.ndim == 1:
        Wave_gt = Wave_gt[None, :]
    if Wave_pr.ndim == 1:
        Wave_pr = Wave_pr[None, :]

    max_seg = min(Wave_gt.shape[0], Wave_pr.shape[0])
    # Keep only segments within range
    seg_idx = [i for i in seg_idx if 0 <= i < max_seg]
    if not seg_idx:
        raise ValueError(f"Chosen segment indices are out of range for subject {worst_subject_id}")

    n = len(seg_idx)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        # For a 2-column grid: keep width reasonable and height proportional to the number of rows.
        # Example: num_segments=10 -> nrows=5 -> figsize ~ (20, 15)
        figsize = (20.0, max(6.0, 2.0 * nrows))
        print(f"Using default figsize: {figsize}")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)
    axes = np.array(axes).reshape(-1)

    t = np.arange(Wave_gt.shape[1], dtype=float) / FS_BVP
    for k, seg_i in enumerate(seg_idx):
        ax = axes[k]
        ax.plot(t, Wave_gt[seg_i, :], label="GT", linewidth=1.2, alpha=0.9)
        ax.plot(t, Wave_pr[seg_i, :], label="Pred", linewidth=1.0, alpha=0.9)
        err_i = chosen_err.get(seg_i, np.nan)
        hr_gt_i = chosen_hr_gt.get(seg_i, np.nan)
        hr_pr_i = chosen_hr_pr.get(seg_i, np.nan)
        if np.isfinite(hr_gt_i) and np.isfinite(hr_pr_i):
            ax.set_title(
                f"seg={seg_i}\nGT={hr_gt_i:.2f} bpm | Pred={hr_pr_i:.2f} bpm\n|err|={abs(err_i):.3g} bpm",
                fontsize=10,
            )
        else:
            ax.set_title(f"seg={seg_i}, |err|={abs(err_i):.3g} bpm", fontsize=10)
        ax.grid(True, alpha=0.25)
        if k == 0:
            ax.legend()

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Worst subject signals: {worst_subject_id}", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_fig_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_fig_path)), exist_ok=True)
        fig.savefig(save_fig_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return worst_subject_id, seg_idx


def visualize_mat_waves(
    save_path: str,
    subject_ids: Optional[List[str]] = None,
    segment_indices: Optional[List[int]] = None,
    max_segments_per_subject: int = 3,
    figsize: tuple = (12, 4),
    fs: float = 30.0,
    vis_run_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Visualize BVP waves from Wave_sort .mat files (gt vs predicted).
    Returns a list of pairs of raw arrays: each item is {"subject_id", "segment", "gt", "pred"}
    with "gt" and "pred" as 1D numpy arrays.
    """
    save_path = os.path.abspath(save_path)
    all_files = [f for f in os.listdir(save_path) if f.endswith(".mat")]
    gt_files = [f for f in all_files if "gt_Wave" in f]
    pr_files = [f for f in all_files if "pr_Wave" in f]

    def _subject_id(s: str) -> str:
        return s.replace("gt_Wave.mat", "").replace("pr_Wave.mat", "").strip()

    gt_by_id = {_subject_id(f): f for f in gt_files}
    pr_by_id = {_subject_id(f): f for f in pr_files}
    common = sorted(set(gt_by_id) & set(pr_by_id))
    if not common:
        raise FileNotFoundError(f"No gt/pr .mat pairs in {save_path}")

    def _resolve_subject_ids(requested: List[str]) -> List[str]:
        """Map requested ids to gt/pr keys (exact match, then case-insensitive)."""
        resolved: List[str] = []
        for s in requested:
            if s in common:
                resolved.append(s)
                continue
            cf_s = s.casefold()
            same_ci = [c for c in common if c.casefold() == cf_s]
            if len(same_ci) == 1:
                print(
                    f"[visualize_mat_waves] subject id {s!r} -> {same_ci[0]!r} "
                    f"(case-insensitive match; Wave_sort filenames are exact)"
                )
                resolved.append(same_ci[0])
            elif len(same_ci) > 1:
                raise ValueError(
                    f"Ambiguous subject id {s!r} (case-insensitive): {same_ci!r}"
                )
            else:
                preview = ", ".join(repr(x) for x in common[:min(6, len(common))])
                raise ValueError(
                    f"subject id {s!r} not found under {save_path!r}. "
                    f"Use the exact prefix before 'gt_Wave.mat' (example ids: {preview} ...; "
                    f"total {len(common)} subjects)."
                )
        return resolved

    if subject_ids is None:
        subject_ids = common[:5]
    else:
        subject_ids = _resolve_subject_ids(list(subject_ids))

    pairs_raw: List[Dict[str, Any]] = []  # list of {"subject_id", "segment", "gt", "pred"}
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
            seg_str = "_join(str(s) for s in segs)"
            fig_path = os.path.join(subject_vis_dir, f"segments_{seg_str}.png")
            fig.savefig(fig_path, dpi=150)
        plt.show()

    return pairs_raw


def estimate_hr_from_psd(signal: np.ndarray, fs: float = FS_BVP,
                         f_low: float = 0.7, f_high: float = 4.0) -> float:
    """
    Estimate heart rate (BPM) from a single 1D BVP segment using Welch PSD.

    - Computes PSD of `signal` with Hann window.
    - Restricts to the band [f_low, f_high] Hz (default 0.7–4.0).
    - Finds the peak frequency in that band and converts to BPM.
    - Returns np.nan if no valid peak is found.
    """
    from scipy.signal import welch as _welch

    signal = np.asarray(signal, dtype=float).ravel()
    if signal.size < 16:
        return np.nan
    f, psd = _welch(
        signal,
        fs=fs,
        window="hann",
        noverlap=None,
        detrend="constant",
    )
    band = (f >= f_low) & (f <= f_high)
    if not np.any(band):
        return np.nan
    f_hr = f[band]
    psd_hr = psd[band]
    peak_freq = f_hr[np.argmax(psd_hr)]
    return float(peak_freq * 60.0)
