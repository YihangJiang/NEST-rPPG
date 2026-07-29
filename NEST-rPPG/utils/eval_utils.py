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
- append_regions_eval_summary_csv
"""
import os
import csv
from functools import lru_cache
from typing import List, Optional, Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy import signal as scipy_signal
from scipy.ndimage import convolve1d
from tqdm import tqdm
import matplotlib.pyplot as plt


FS_BVP = 30  # BVP sampling rate [Hz] in .mat (raw segments)
HR_FREQ_LOW, HR_FREQ_HIGH = 0.7, 3.0  # HR band [Hz] for FFT peak


@lru_cache(maxsize=32)
def _fir_bandpass_coefs(sig_len: int, fs: float) -> np.ndarray:
    n = max(round(sig_len / 10), 1)
    return scipy_signal.firwin(n + 1, [0.8, 3.0], pass_zero=False, fs=fs)


def bpfilter64(sig: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass filter [0.8, 3] Hz using FIR (Hamming), forward-backward convolve."""
    sig = np.asarray(sig, dtype=float)
    single = sig.ndim == 1
    if single:
        sig = sig[None, :]
    b = _fir_bandpass_coefs(sig.shape[-1], float(fs))
    # scipy.signal.filtfilt can hang with recent SciPy/NumPy; convolve1d is equivalent here.
    filtered = convolve1d(sig, b, axis=-1, mode="nearest")
    filtered = convolve1d(filtered[..., ::-1], b, axis=-1, mode="nearest")[..., ::-1]
    return filtered[0] if single else filtered


def hr_from_fft_batch(signals: np.ndarray, fs: float = FS_BVP) -> np.ndarray:
    """
    Heart rate from FFT for one or many 1D segments.
    signals: shape (T,) or (N, T) -> returns scalar BPM or (N,) array.
    CPU-only (NumPy/SciPy); no GPU used.
    """
    signals = np.asarray(signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[None, :]
    n_seg, n = signals.shape
    if n < 64:
        return np.full(n_seg, np.nan)

    filtered = bpfilter64(signals, fs)
    filtered = filtered - filtered.mean(axis=-1, keepdims=True)
    filtered = filtered / (filtered.std(axis=-1, keepdims=True) + 1e-12)
    fft_vals = np.fft.rfft(filtered, axis=-1)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    power = np.abs(fft_vals) ** 2
    mask = (freqs >= HR_FREQ_LOW) & (freqs <= HR_FREQ_HIGH)
    if not np.any(mask):
        return np.full(n_seg, np.nan)

    power_band = power[:, mask]
    freqs_band = freqs[mask]
    idx_max = np.argmax(power_band, axis=-1)
    return freqs_band[idx_max] * 60.0


def hr_from_fft(signal_1d: np.ndarray, fs: float = FS_BVP) -> float:
    """
    Heart rate from FFT: bandpass filter, then peak frequency in [HR_FREQ_LOW, HR_FREQ_HIGH] Hz.
    Returns HR in BPM, or np.nan if signal too short or no valid peak.
    """
    hr = hr_from_fft_batch(signal_1d, fs=fs)
    return float(hr[0]) if hr.size else np.nan


def _empty_hr_metrics() -> Dict[str, float]:
    nan = float(np.nan)
    return {
        "ME": nan,
        "Std": nan,
        "MAE": nan,
        "MAE_Std": nan,
        "MAE_SE": nan,
        "RMSE": nan,
        "RMSE_Std": nan,
        "RMSE_SE": nan,
    }


def my_eval(hr_pr: np.ndarray, hr_gt: np.ndarray) -> Dict[str, float]:
    """
    Subject-level HR error metrics (matches MyEval.m for ME, Std, MAE, RMSE).

    Also reports dispersion / standard error for MAE and RMSE:
    - MAE_Std / MAE_SE: std / SE of per-subject absolute errors |HR_pr - HR_gt|
    - RMSE_Std / RMSE_SE: std / SE of jackknife leave-one-out RMSE (BPM)
    """
    hr_pr = np.asarray(hr_pr, dtype=float).ravel()
    hr_gt = np.asarray(hr_gt, dtype=float).ravel()
    if len(hr_pr) != len(hr_gt) or len(hr_pr) == 0:
        return _empty_hr_metrics()

    err = hr_pr - hr_gt
    valid = np.isfinite(hr_pr) & np.isfinite(hr_gt)
    n = int(np.sum(valid))
    if n == 0:
        return _empty_hr_metrics()

    err = err[valid]
    abs_err = np.abs(err)
    sq_err = err * err

    me = float(np.mean(err))
    e_std = float(np.std(err, ddof=0)) if n > 1 else (0.0 if n == 1 else np.nan)
    mae = float(np.mean(abs_err))
    mae_std = float(np.std(abs_err, ddof=0)) if n > 1 else (0.0 if n == 1 else np.nan)
    mae_se = float(mae_std / np.sqrt(n))
    rmse = float(np.sqrt(np.mean(sq_err)))
    if n > 1:
        total_sq = float(np.sum(sq_err))
        loo_rmse = np.sqrt((total_sq - sq_err) / (n - 1))
        rmse_std = float(np.std(loo_rmse, ddof=0))
        rmse_se = float(rmse_std / np.sqrt(n))
    else:
        rmse_std = 0.0
        rmse_se = 0.0

    return {
        "ME": me,
        "Std": e_std,
        "MAE": mae,
        "MAE_Std": mae_std,
        "MAE_SE": mae_se,
        "RMSE": rmse,
        "RMSE_Std": rmse_std,
        "RMSE_SE": rmse_se,
    }


def _video_metrics_from_chunk_errors(
    hr_pr_list: List[float],
    hr_gt_list: List[float],
) -> Dict[str, float]:
    """Per-video metrics from paired chunk HR values (chunk error first)."""
    hr_pr = np.asarray(hr_pr_list, dtype=float)
    hr_gt = np.asarray(hr_gt_list, dtype=float)
    if hr_pr.shape != hr_gt.shape or hr_pr.size == 0:
        nan = float(np.nan)
        return {"video_me": nan, "video_mae": nan, "video_rmse": nan, "n_chunks_used": 0}

    err = hr_pr - hr_gt
    valid = np.isfinite(err)
    n = int(np.sum(valid))
    if n == 0:
        nan = float(np.nan)
        return {"video_me": nan, "video_mae": nan, "video_rmse": nan, "n_chunks_used": 0}

    err = err[valid]
    abs_err = np.abs(err)
    sq_err = err * err
    return {
        "video_me": float(np.mean(err)),
        "video_mae": float(np.mean(abs_err)),
        "video_rmse": float(np.sqrt(np.mean(sq_err))),
        "n_chunks_used": n,
    }


def aggregate_per_video_metrics(
    video_mes: np.ndarray,
    video_maes: np.ndarray,
    video_rmses: np.ndarray,
) -> Dict[str, float]:
    """
    Dataset-level HR metrics from per-video chunk-first statistics.

    Each video contributes one value (equal weight), regardless of chunk count.
    """
    video_mes = np.asarray(video_mes, dtype=float).ravel()
    video_maes = np.asarray(video_maes, dtype=float).ravel()
    video_rmses = np.asarray(video_rmses, dtype=float).ravel()
    if (
        len(video_mes) != len(video_maes)
        or len(video_mes) != len(video_rmses)
        or len(video_mes) == 0
    ):
        return _empty_hr_metrics()

    valid = np.isfinite(video_mes) & np.isfinite(video_maes) & np.isfinite(video_rmses)
    n = int(np.sum(valid))
    if n == 0:
        return _empty_hr_metrics()

    video_mes = video_mes[valid]
    video_maes = video_maes[valid]
    video_rmses = video_rmses[valid]

    me = float(np.mean(video_mes))
    e_std = float(np.std(video_mes, ddof=0)) if n > 1 else (0.0 if n == 1 else np.nan)
    mae = float(np.mean(video_maes))
    mae_std = float(np.std(video_maes, ddof=0)) if n > 1 else (0.0 if n == 1 else np.nan)
    mae_se = float(mae_std / np.sqrt(n))
    rmse = float(np.mean(video_rmses))
    if n > 1:
        rmse_std = float(np.std(video_rmses, ddof=0))
        rmse_se = float(rmse_std / np.sqrt(n))
    else:
        rmse_std = 0.0
        rmse_se = 0.0

    return {
        "ME": me,
        "Std": e_std,
        "MAE": mae,
        "MAE_Std": mae_std,
        "MAE_SE": mae_se,
        "RMSE": rmse,
        "RMSE_Std": rmse_std,
        "RMSE_SE": rmse_se,
    }


def print_hr_metrics(
    result: Dict[str, Dict[str, Any]],
    *,
    source_domain: Optional[str] = None,
    target_domain: Optional[str] = None,
) -> None:
    """Print HR metric table including MAE/RMSE std and standard error."""
    if source_domain is not None:
        print(f"Source domain: {source_domain}")
    if target_domain is not None:
        print(f"Target domain: {target_domain}")
    print(
        "Feature    \tME\t\tStd\t\tMAE\t\tMAE_Std\t\tMAE_SE\t\t"
        "RMSE\t\tRMSE_Std\t\tRMSE_SE"
    )
    print("-" * 120)
    for name, metrics in result.items():
        print(
            f"{name:10}\t{metrics['ME']:.6f}\t{metrics['Std']:.6f}\t"
            f"{metrics['MAE']:.6f}\t{metrics['MAE_Std']:.6f}\t{metrics['MAE_SE']:.6f}\t"
            f"{metrics['RMSE']:.6f}\t{metrics['RMSE_Std']:.6f}\t{metrics['RMSE_SE']:.6f}"
        )


@lru_cache(maxsize=256)
def _load_bvp_full(bvp_path: str) -> np.ndarray:
    """Load full BVP trace once per subject (cached for index scans)."""
    bvp = scio.loadmat(bvp_path)["BVP"]
    return np.array(bvp, dtype=np.float32).reshape(-1)


def _load_bvp_segment(now_path: str, step_index: int, frames_num: int, data_name: str) -> np.ndarray:
    """Load and min-max normalize one BVP clip (matches MyDataset.getLabel)."""
    if (
        data_name.startswith("PURE_my")
        or data_name.startswith("UBFC_my")
        or data_name.startswith("BUAA_my")
        or data_name in ("PURE", "UBFC", "BUAA")
    ):
        bvp_path = os.path.join(now_path, "Label", "BVP.mat")
        bvp = _load_bvp_full(bvp_path)
        bvp = bvp[step_index: step_index + frames_num]
        bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp) + 1e-8)
        return bvp.astype(np.float32)
    raise ValueError(f"Unsupported data_name for BVP load: {data_name!r}")


def infer_frames_num_from_wave_sort(save_path: str) -> int:
    """Infer segment length (frames_num) from the first gt_Wave.mat in a Wave_sort folder."""
    save_path = os.path.abspath(save_path)
    gt_files = sorted(f for f in os.listdir(save_path) if f.endswith(".mat") and "gt_Wave" in f)
    if not gt_files:
        raise FileNotFoundError(f"No gt_Wave.mat files in {save_path}")
    wave = np.squeeze(scio.loadmat(os.path.join(save_path, gt_files[0]))["Wave"])
    if wave.ndim == 1:
        return int(wave.shape[0])
    return int(wave.shape[1])


def mean_gt_hr_from_index(
    index_dir: str,
    data_name: str,
    frames_num: int,
) -> Tuple[float, int]:
    """
    Mean FFT HR (BPM) over all GT BVP clips in index_dir.
    Returns (mean_hr_bpm, n_valid_segments).
    """
    segment_hrs = gt_hr_segments_from_index(index_dir, data_name, frames_num)
    hr_values = [float(s["hr_bpm"]) for s in segment_hrs if np.isfinite(s["hr_bpm"])]
    if not hr_values:
        raise ValueError(f"No valid FFT HR values from index: {index_dir}")
    return float(np.mean(hr_values)), len(hr_values)


def gt_hr_segments_from_index(
    index_dir: str,
    data_name: str,
    frames_num: int,
) -> List[Dict[str, Any]]:
    """FFT HR (BPM) for every GT BVP clip listed in an index directory."""
    index_dir = os.path.abspath(index_dir)
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Index dir not found: {index_dir}")

    segments: List[Dict[str, Any]] = []
    index_files = sorted(f for f in os.listdir(index_dir) if f.endswith(".mat"))
    for fname in tqdm(index_files, desc=f"Index HR ({os.path.basename(index_dir)})", unit="clip"):
        temp = scio.loadmat(os.path.join(index_dir, fname))
        now_path = str(temp["Path"][0])
        step_index = int(np.asarray(temp["Step_Index"]).flat[0])
        bvp = _load_bvp_segment(now_path, step_index, frames_num, data_name)
        hr_bpm = hr_from_fft(bvp, fs=FS_BVP)
        segments.append({
            "subject_id": os.path.basename(now_path.rstrip(os.sep)),
            "hr_bpm": float(hr_bpm) if np.isfinite(hr_bpm) else np.nan,
        })
    if not segments:
        raise ValueError(f"No index clips found in: {index_dir}")
    return segments


def subject_mean_hr_from_segments(segments: List[Dict[str, Any]]) -> Tuple[np.ndarray, int, int]:
    """Average segment HR per subject. Returns (subject_mean_hr, n_segments, n_subjects)."""
    grouped: Dict[str, List[float]] = {}
    for seg in segments:
        hr_bpm = float(seg["hr_bpm"])
        if not np.isfinite(hr_bpm):
            continue
        grouped.setdefault(str(seg["subject_id"]), []).append(hr_bpm)

    if not grouped:
        raise ValueError("No valid segment HR values to aggregate by subject")

    subject_means = np.array(
        [float(np.mean(hrs)) for hrs in grouped.values()],
        dtype=float,
    )
    n_segments = sum(len(hrs) for hrs in grouped.values())
    return subject_means, n_segments, len(grouped)


def eval_mean_guessing_baseline(
    train_index_dir: str,
    source_domain: str,
    test_index_dir: str,
    target_domain: str,
    frames_num: int,
) -> Dict[str, float]:
    """
    Mean-guessing baseline (subject-level).

    1. train_mean = mean FFT HR over all GT BVP clips in the training index.
    2. test_gt = per-subject mean FFT HR over all GT BVP clips in the test index.
    3. Predict constant train_mean for every test subject.
    4. Compare predictions to test GT with my_eval.
    """
    train_mean_hr, n_train_segments = mean_gt_hr_from_index(
        train_index_dir, source_domain, frames_num
    )
    test_segments = gt_hr_segments_from_index(test_index_dir, target_domain, frames_num)
    test_hr_gt, n_test_segments, n_test_subjects = subject_mean_hr_from_segments(test_segments)
    hr_pr = np.full_like(test_hr_gt, train_mean_hr)
    metrics = my_eval(hr_pr, test_hr_gt)
    return {
        "train_domain": source_domain,
        "test_domain": target_domain,
        "train_mean_hr_bpm": train_mean_hr,
        "n_train_segments": float(n_train_segments),
        "n_test_segments": float(n_test_segments),
        "n_test_subjects": float(n_test_subjects),
        **metrics,
    }


def run_eval(
    save_path: str,
    *,
    return_details: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]] | Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Load gt/pr .mat pairs, compute FFT heart rate per chunk (fs=FS_BVP, no interpolation).

    Protocol:
    1. Per chunk: err = HR_pr - HR_gt
    2. Per video: MAE/ME/RMSE from chunk errors (mean |err|, mean err, sqrt(mean err^2))
    3. Dataset: mean of per-video metrics (each video weighted equally)

    When gt/pr chunk counts differ for a video, only the first min(n_gt, n_pr) chunks
    are paired by index; a warning is printed and the mismatch is recorded in details.
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

    video_mes: List[float] = []
    video_maes: List[float] = []
    video_rmses: List[float] = []
    details: Dict[str, Any] = {
        "save_path": save_path,
        "subjects": [],  # list of per-subject dicts (with per-segment arrays)
        "rows": [],      # flat rows for CSV writing (subject_id, segment_idx, hr_gt, hr_pr, err)
        "chunk_mismatches": [],  # videos where n_gt != n_pr
    }
    print(
        f"Found {len(common_ids)} subject pairs. "
        f"Chunk-error-first eval (fs={FS_BVP} Hz, no interpolation)..."
    )
    if verbose:
        print(f"Subjects: {common_ids}")
    for sid in tqdm(common_ids, desc="Subjects", unit="subject", disable=not verbose):
        gt_mat = scio.loadmat(os.path.join(save_path, gt_by_id[sid]))
        pr_mat = scio.loadmat(os.path.join(save_path, pr_by_id[sid]))
        Wave_gt = np.squeeze(gt_mat["Wave"])
        Wave_pr = np.squeeze(pr_mat["Wave"])
        if Wave_gt.ndim == 1:
            Wave_gt = Wave_gt[None, :]
        if Wave_pr.ndim == 1:
            Wave_pr = Wave_pr[None, :]
        n_gt = int(Wave_gt.shape[0])
        n_pr = int(Wave_pr.shape[0])
        chunk_mismatch = n_gt != n_pr
        if n_gt == 0 or n_pr == 0:
            print(f"Warning: Subject {sid} has no segments (gt={n_gt}, pr={n_pr}), skipping...")
            continue
        if chunk_mismatch:
            msg = (
                f"Chunk count mismatch for subject {sid}: "
                f"gt={n_gt}, pr={n_pr}; using first {min(n_gt, n_pr)} paired chunks"
            )
            print(f"Warning: {msg}")
            details["chunk_mismatches"].append({
                "subject_id": sid,
                "n_chunks_gt": n_gt,
                "n_chunks_pr": n_pr,
                "n_chunks_used": min(n_gt, n_pr),
            })

        n_use = min(n_gt, n_pr)
        hr_pr_list: List[float] = []
        hr_gt_list: List[float] = []
        hr_pr_arr = hr_from_fft_batch(np.asarray(Wave_pr[:n_use], dtype=float), fs=FS_BVP)
        hr_gt_arr = hr_from_fft_batch(np.asarray(Wave_gt[:n_use], dtype=float), fs=FS_BVP)
        for n in range(n_use):
            hr_pr_n = float(hr_pr_arr[n])
            hr_gt_n = float(hr_gt_arr[n])
            hr_pr_list.append(hr_pr_n)
            hr_gt_list.append(hr_gt_n)
            details["rows"].append({
                "subject_id": sid,
                "segment_idx": int(n),
                "hr_gt_bpm": float(hr_gt_n) if np.isfinite(hr_gt_n) else np.nan,
                "hr_pr_bpm": float(hr_pr_n) if np.isfinite(hr_pr_n) else np.nan,
                "err_bpm": float(hr_pr_n - hr_gt_n) if (np.isfinite(hr_pr_n) and np.isfinite(hr_gt_n)) else np.nan,
                "chunk_mismatch": chunk_mismatch,
                "n_chunks_gt": n_gt,
                "n_chunks_pr": n_pr,
            })

        chunk_err = np.array(hr_pr_list, dtype=float) - np.array(hr_gt_list, dtype=float)
        video_stats = _video_metrics_from_chunk_errors(hr_pr_list, hr_gt_list)
        video_mes.append(video_stats["video_me"])
        video_maes.append(video_stats["video_mae"])
        video_rmses.append(video_stats["video_rmse"])

        with np.errstate(invalid="ignore"):
            subj_hr_pr = float(np.nanmean(hr_pr_list))
            subj_hr_gt = float(np.nanmean(hr_gt_list))

        details["subjects"].append({
            "subject_id": sid,
            "n_chunks_gt": n_gt,
            "n_chunks_pr": n_pr,
            "n_chunks_used": n_use,
            "chunk_count_mismatch": chunk_mismatch,
            "hr_gt_segments_bpm": np.array(hr_gt_list, dtype=float),
            "hr_pr_segments_bpm": np.array(hr_pr_list, dtype=float),
            "err_segments_bpm": chunk_err,
            "video_me_bpm": video_stats["video_me"],
            "video_mae_bpm": video_stats["video_mae"],
            "video_rmse_bpm": video_stats["video_rmse"],
            "mean_err_bpm": video_stats["video_me"],
            "median_err_bpm": float(np.nanmedian(chunk_err)),
            "mean_hr_gt_bpm": subj_hr_gt,
            "mean_hr_pr_bpm": subj_hr_pr,
        })

    if len(video_maes) == 0:
        raise ValueError("No valid data found - all subjects have empty arrays")

    if details["chunk_mismatches"] and verbose:
        print(
            f"Chunk mismatches: {len(details['chunk_mismatches'])} / {len(common_ids)} subjects"
        )

    result = {
        "HR": aggregate_per_video_metrics(
            np.array(video_mes, dtype=float),
            np.array(video_maes, dtype=float),
            np.array(video_rmses, dtype=float),
        )
    }
    if return_details:
        details["video_me_bpm"] = np.array(video_mes, dtype=float)
        details["video_mae_bpm"] = np.array(video_maes, dtype=float)
        details["video_rmse_bpm"] = np.array(video_rmses, dtype=float)
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
    n_total = len(subjects)
    if top_k is None and n_total > 40:
        top_k = 40
    if top_k is not None:
        subjects = subjects[: int(top_k)]

    ids = [s["subject_id"] for s in subjects]
    mean_err = np.array([s.get("mean_err_bpm", np.nan) for s in subjects], dtype=float)
    med_err = np.array([s.get("median_err_bpm", np.nan) for s in subjects], dtype=float)

    n = len(ids)
    fig_w = max(figsize[0], min(48.0, 0.22 * n))
    fig_h = figsize[1]
    x = np.arange(n)
    w = 0.42
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.bar(x - w / 2, mean_err, width=w, label="Mean error (BPM)")
    ax.bar(x + w / 2, med_err, width=w, label="Median error (BPM)")
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax.set_xticks(x)
    label_fs = max(5, min(8, 200 // max(n, 1)))
    ax.set_xticklabels(ids, rotation=90, ha="center", fontsize=label_fs)
    ax.set_ylabel("Error (Pred - GT) [BPM]")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plot_title = title or "Per-subject error summary (segment HR)"
    if top_k is not None and n_total > n:
        plot_title += f" (top {n} of {n_total} subjects)"
    ax.set_title(plot_title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        save_dpi = 100 if n > 30 else 150
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
        plt.close(fig)
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
    - chunk_mismatch
    - n_chunks_gt
    - n_chunks_pr
    """
    rows = list(details.get("rows", []))
    if not rows:
        raise ValueError("details has no rows. Call run_eval(..., return_details=True) first.")

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "subject_id", "segment_idx", "hr_gt_bpm", "hr_pr_bpm", "err_bpm",
        "chunk_mismatch", "n_chunks_gt", "n_chunks_pr",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return os.path.abspath(csv_path)


def write_chunk_mismatch_csv(details: Dict[str, Any], csv_path: str) -> Optional[str]:
    """Write per-video gt/pr chunk-count mismatches (empty file skipped if none)."""
    mismatches = list(details.get("chunk_mismatches", []))
    if not mismatches:
        return None

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = ["subject_id", "n_chunks_gt", "n_chunks_pr", "n_chunks_used"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in mismatches:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return os.path.abspath(csv_path)


def append_regions_eval_summary_csv(
    csv_path: str,
    *,
    source_domain: str,
    target_domain: str,
    weight: float,
    regions: str,
    result: Dict[str, Dict[str, Any]],
    metric_key: str = "HR",
) -> str:
    """
    Append one row to a summary CSV (creates file with header if missing).

    Intended for `run_regions.sh` flows: after `train_regions.py` + `eval_from_bvp.py`,
    record source/target domains, InfoNCE weight, and HR metrics from `run_eval` output.

    Columns: Source Domain, Target domain, Weight, Regions, Std, MAE, MAE_Std, MAE_SE,
    RMSE, RMSE_Std, RMSE_SE
    """
    if metric_key not in result:
        raise KeyError(f"result has no key {metric_key!r}; keys: {list(result.keys())}")
    m = result[metric_key]
    row = {
        "Source Domain": source_domain,
        "Target domain": target_domain,
        "Weight": weight,
        "Regions": regions,
        "Std": m.get("Std", np.nan),
        "MAE": m.get("MAE", np.nan),
        "MAE_Std": m.get("MAE_Std", np.nan),
        "MAE_SE": m.get("MAE_SE", np.nan),
        "RMSE": m.get("RMSE", np.nan),
        "RMSE_Std": m.get("RMSE_Std", np.nan),
        "RMSE_SE": m.get("RMSE_SE", np.nan),
    }
    fieldnames = [
        "Source Domain", "Target domain", "Weight", "Regions",
        "Std", "MAE", "MAE_Std", "MAE_SE", "RMSE", "RMSE_Std", "RMSE_SE",
    ]
    training_cols = fieldnames[:4]
    inference_cols = fieldnames[4:]

    csv_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        # Append to existing CSV.
        df = pd.read_csv(csv_path)
        if "r" in df.columns:
            df = df.drop(columns=["r"])
        df = df.reindex(columns=fieldnames)
        new_row = pd.DataFrame([[
            source_domain, target_domain, weight, regions,
            m.get("Std", np.nan), m.get("MAE", np.nan),
            m.get("MAE_Std", np.nan), m.get("MAE_SE", np.nan),
            m.get("RMSE", np.nan), m.get("RMSE_Std", np.nan), m.get("RMSE_SE", np.nan),
        ]], columns=fieldnames)
        match = pd.Series(True, index=df.index)
        for c in training_cols:
            match = match & (df[c].astype(str) == str(new_row.iloc[0][c]))
        if bool(match.any()):
            df.loc[match, :] = new_row.iloc[0].values
        else:
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        # Create new DataFrame with single-level headers.
        new_row = [[
            source_domain, target_domain, weight, regions,
            m.get("Std", np.nan), m.get("MAE", np.nan),
            m.get("MAE_Std", np.nan), m.get("MAE_SE", np.nan),
            m.get("RMSE", np.nan), m.get("RMSE_Std", np.nan), m.get("RMSE_SE", np.nan),
        ]]
        df = pd.DataFrame(new_row, columns=fieldnames)

    df.to_csv(csv_path, index=False)
    return csv_path


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
    show: bool = False,
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
        if show:
            plt.show()
        else:
            plt.close(fig)

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
