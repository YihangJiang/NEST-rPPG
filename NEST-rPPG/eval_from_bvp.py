"""
Evaluates BVP model performance from Wave_sort .mat files (gt/pr pairs):
heart rate only, from FFT on raw segments. Reports ME, Std, MAE, RMSE, MER, Pearson r.
When run this script, use interpreter: mprppg
"""
# %%

import os
import json

import config
from utils.eval_utils import (
    FS_BVP,
    hr_from_fft,
    my_eval,
    run_eval,
    visualize_mat_waves,
    estimate_hr_from_psd,
    plot_subject_error_bars,
    write_segment_errors_csv,
    plot_worst_subject_from_segment_csv,
    plot_worst_subject_signals,
)

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


# %%
# Optional: visualize waves before evaluation (example usage)
pairs = visualize_mat_waves(
    save_path,
    segment_indices=[3, 5],
    vis_run_name=config.LOSS_TYPE,
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

# 2) Bar plot: per-subject mean/median error
bar_path = os.path.join(feature_dir, "subject_error_bars.png")
plot_subject_error_bars(
    details,
    title=f"{config.SRC_DOMAIN} -> {config.TGT_DOMAIN}",
    save_path=bar_path,
)
print(f"Saved subject error bar plot: {bar_path}")

# 2.5) Plot the worst subject by absolute error
worst_plot_path = os.path.join(feature_dir, "worst_subject_plot.png")
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
    "result": result,
}
with open(json_path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Saved eval summary JSON: {json_path}")

# Print table: ME, Std, MAE, RMSE, MER, Pearson r
print(f"Source domain: {config.SRC_DOMAIN}")
print(f"Target domain: {config.TGT_DOMAIN}")
print("Feature    \tME\t\tStd\t\tMAE\t\tRMSE\t\tMER\t\tr")
print("-" * 90)
for name, metrics in result.items():
    print(
        f"{name:10}\t{metrics['ME']:.6f}\t{metrics['Std']:.6f}\t{metrics['MAE']:.6f}\t"
        f"{metrics['RMSE']:.6f}\t{metrics['MER']:.6f}\t{metrics['r']:.6f}"
    )

# %%
# Optional debug example (disabled by default):
# _pairs = visualize_mat_waves(
#     save_path,
#     subject_ids=["Sub_06lux 10.0"],
#     segment_indices=[145],
#     vis_run_name="check",
# )
# %%
