"""
Evaluates BVP model performance from Wave_sort .mat files (gt/pr pairs):
heart rate only, from FFT on raw segments. Reports ME, Std, MAE, RMSE and
MAE/RMSE std and standard error.
When run this script, use interpreter: mpipe (or training)
"""
# %%

import os
import json

import config
from utils.eval_utils import (
    FS_BVP,
    hr_from_fft,
    run_eval,
    print_hr_metrics,
    infer_frames_num_from_wave_sort,
    eval_mean_guessing_baseline,
    visualize_mat_waves,
    estimate_hr_from_psd,
    plot_subject_error_bars,
    write_segment_errors_csv,
    write_chunk_mismatch_csv,
    append_regions_eval_summary_csv,
    plot_worst_subject_from_segment_csv,
    plot_worst_subject_signals,
)
import utils.mlflow_utils as mlflow_utils


save_path = config.EVAL_SAVE_PATH
# If train_regions.py was run recently, follow its Wave_sort output path automatically.
try:
    last_path_file = os.path.join(config.RESULT_LOG_DIR, "last_wave_sort_path.txt")
    if os.path.isfile(last_path_file):
        with open(last_path_file, "r") as f:
            candidate = f.read().strip()
        if candidate and os.path.isdir(candidate):
            save_path = candidate
except Exception:
    pass
print(f"Evaluating Wave_sort path: {save_path}")

# Batch runs (run_regions.sh): skip matplotlib plots unless NEST_EVAL_PLOTS=1.
GENERATE_PLOTS = os.environ.get("NEST_EVAL_PLOTS", "0") == "1"
if not GENERATE_PLOTS:
    print("Skipping eval plots (set NEST_EVAL_PLOTS=1 to enable).")


# %%
# Optional: visualize waves before evaluation (set True in Jupyter to inspect samples)
VISUALIZE_EXAMPLE = False
if VISUALIZE_EXAMPLE:
    pairs = visualize_mat_waves(
        save_path,
        segment_indices=[3, 5],
        vis_run_name=config.LOSS_TYPE,
        show=True,
    )
    signal = pairs[5]["pred"]
    hr_bpm = estimate_hr_from_psd(signal, fs=FS_BVP, f_low=0.7, f_high=4.0)
    print(f"Estimated HR (Welch PSD): {hr_bpm:.2f} bpm")

# %%
# Evaluate + collect per-segment details for analysis outputs
result, details = run_eval(save_path, return_details=True)

# Save analysis outputs under: <Wave_sort run folder>/feature/
feature_dir = os.path.join(save_path, "feature")
os.makedirs(feature_dir, exist_ok=True)

# 1) CSV: per-segment errors
csv_path = os.path.join(feature_dir, "segment_errors.csv")
write_segment_errors_csv(details, csv_path)
print(f"Saved segment errors CSV: {csv_path}")

mismatch_csv = os.path.join(feature_dir, "chunk_mismatches.csv")
mismatch_path = write_chunk_mismatch_csv(details, mismatch_csv)
if mismatch_path:
    print(f"Saved chunk mismatch CSV: {mismatch_path}")
else:
    print("No gt/pr chunk-count mismatches across subjects.")

# 2) Bar plot: per-subject mean/median error
bar_path = os.path.join(feature_dir, "subject_error_bars.png")
if GENERATE_PLOTS:
    print("Saving subject error bar plot...")
    plot_subject_error_bars(
        details,
        title=f"{config.SRC_DOMAIN} -> {config.TGT_DOMAIN}",
        save_path=bar_path,
    )
    print(f"Saved subject error bar plot: {bar_path}")

# 2.5) Plot the worst subject by absolute error
worst_plot_path = os.path.join(feature_dir, "worst_subject_plot.png")
worst_sid = None
worst_score = float("nan")
if GENERATE_PLOTS:
    print("Saving worst-subject plot...")
    worst_sid, worst_score = plot_worst_subject_from_segment_csv(
        csv_path,
        metric="max_abs",
        save_fig_path=worst_plot_path,
        show=False,
    )
    print(f"Saved worst subject plot: {worst_plot_path}")
    print(f"Worst subject: {worst_sid} (max_abs_err={worst_score:.4g} BPM)")

# 2.6) Plot worst subject GT/Pred signals for 20 segments
worst_signals_path = os.path.join(feature_dir, "worst_subject_signals.png")
if GENERATE_PLOTS and worst_sid is not None:
    print("Saving worst-subject signal plot...")
    plot_worst_subject_signals(
        save_path,
        csv_path,
        num_segments=10,
        worst_subject_id=worst_sid,
        save_fig_path=worst_signals_path,
        show=False,
    )
    print(f"Saved worst subject signals plot: {worst_signals_path}")

# 3) JSON: summary metrics + context
json_path = os.path.join(feature_dir, "eval_result.json")
payload = {
    "save_path": save_path,
    "source_domain": config.SRC_DOMAIN,
    "target_domain": config.TGT_DOMAIN,
    "loss_type": config.LOSS_TYPE,
    "eval_protocol": "chunk_error_first_then_per_video_mae",
    "chunk_mismatches": details.get("chunk_mismatches", []),
    "result": result,
}

# Append one row to cumulative regions summary (source/target/weight/regions from train meta).
# These fields must come from last_train_regions_meta.json written by train_regions.py.
meta_path = os.path.join(config.RESULT_LOG_DIR, "last_train_regions_meta.json")
if not os.path.isfile(meta_path):
    raise FileNotFoundError(f"Missing required train meta JSON: {meta_path}")
with open(meta_path, "r") as f:
    meta = json.load(f)
src_for_csv = str(meta["source_domain"])
tgt_for_csv = str(meta["target_domain"])
weight_for_csv = float(meta["weight_info"])
regions_for_csv = str(meta["regions"])
payload["regions"] = regions_for_csv

frames_num = infer_frames_num_from_wave_sort(save_path)
train_index_dir = os.path.join(config.STMAP_INDEX_BASE, src_for_csv)
test_index_dir = os.path.join(config.STMAP_INDEX_BASE, tgt_for_csv)
print(f"Computing mean-guessing baseline ({src_for_csv} -> {tgt_for_csv})...")
mean_guess = eval_mean_guessing_baseline(
    train_index_dir,
    config.canonical_data_name(src_for_csv),
    test_index_dir,
    config.canonical_data_name(tgt_for_csv),
    frames_num,
)
payload["mean_guessing_baseline"] = {
    "train_domain": src_for_csv,
    "test_domain": tgt_for_csv,
    "train_mean_hr_bpm": mean_guess["train_mean_hr_bpm"],
    "n_train_segments": int(mean_guess["n_train_segments"]),
    "n_test_segments": int(mean_guess["n_test_segments"]),
    "n_test_subjects": int(mean_guess["n_test_subjects"]),
    "ME": mean_guess["ME"],
    "Std": mean_guess["Std"],
    "MAE": mean_guess["MAE"],
    "MAE_Std": mean_guess["MAE_Std"],
    "MAE_SE": mean_guess["MAE_SE"],
    "RMSE": mean_guess["RMSE"],
    "RMSE_Std": mean_guess["RMSE_Std"],
    "RMSE_SE": mean_guess["RMSE_SE"],
}

with open(json_path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Updated eval summary JSON with regions: {json_path}")
summary_csv = os.path.join(config.RESULT_LOG_DIR, "regions_eval_summary.csv")
append_regions_eval_summary_csv(
    summary_csv,
    source_domain=src_for_csv,
    target_domain=tgt_for_csv,
    weight=weight_for_csv,
    regions=regions_for_csv,
    result=result,
)
print(f"Appended regions eval summary row: {summary_csv}")

if mlflow_utils.resume_run():
    hr_metrics = result.get("HR", {})
    mlflow_utils.log_metrics({
        'eval_ME': float(hr_metrics.get('ME', 0.0)),
        'eval_Std': float(hr_metrics.get('Std', 0.0)),
        'eval_MAE': float(hr_metrics.get('MAE', 0.0)),
        'eval_MAE_Std': float(hr_metrics.get('MAE_Std', 0.0)),
        'eval_MAE_SE': float(hr_metrics.get('MAE_SE', 0.0)),
        'eval_RMSE': float(hr_metrics.get('RMSE', 0.0)),
        'eval_RMSE_Std': float(hr_metrics.get('RMSE_Std', 0.0)),
        'eval_RMSE_SE': float(hr_metrics.get('RMSE_SE', 0.0)),
        'eval_mean_guess_MAE': float(mean_guess['MAE']),
        'eval_mean_guess_ME': float(mean_guess['ME']),
        'eval_mean_guess_RMSE': float(mean_guess['RMSE']),
        'eval_mean_guess_train_hr': float(mean_guess['train_mean_hr_bpm']),
    })
    mlflow_utils.log_params({
        'eval_source_domain': src_for_csv,
        'eval_target_domain': tgt_for_csv,
        'eval_weight_info': weight_for_csv,
        'eval_regions': regions_for_csv,
        'eval_mean_guess_train_domain': src_for_csv,
        'eval_mean_guess_test_domain': tgt_for_csv,
        'eval_mean_guess_n_train_segments': int(mean_guess['n_train_segments']),
        'eval_mean_guess_n_test_segments': int(mean_guess['n_test_segments']),
        'eval_mean_guess_n_test_subjects': int(mean_guess['n_test_subjects']),
    })
    artifact_paths = [json_path, csv_path]
    if GENERATE_PLOTS:
        artifact_paths.extend([bar_path, worst_plot_path, worst_signals_path])
    mlflow_utils.log_artifacts(artifact_paths)
    mlflow_utils.end_run()
    print("Logged eval metrics to MLflow run.")

print_hr_metrics(result, source_domain=src_for_csv, target_domain=tgt_for_csv)

print()
print("Mean-guessing baseline (training GT mean vs test GT)")
print(f"  Training domain:     {src_for_csv}")
print(f"  Test domain:         {tgt_for_csv}")
print(f"  Training index:      {train_index_dir}")
print(f"  Test index:          {test_index_dir}")
print(f"  Training mean HR:    {mean_guess['train_mean_hr_bpm']:.6f} bpm  "
      f"({int(mean_guess['n_train_segments'])} segments)")
print(f"  Test segments:       {int(mean_guess['n_test_segments'])}")
print(f"  Test subjects:       {int(mean_guess['n_test_subjects'])}")
print(f"  MAE (|train_mean - test_gt|): {mean_guess['MAE']:.6f}")

# %%
# Optional debug example (disabled by default):
# _pairs = visualize_mat_waves(
#     save_path,
#     subject_ids=["Sub_06lux 10.0"],
#     segment_indices=[145],
#     vis_run_name="check",
# )
# %%
