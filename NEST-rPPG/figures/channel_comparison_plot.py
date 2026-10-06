# %%
# Bar plot comparing MAE across different color channels (RGB, Red, Green, Blue).
#
# Follows the exact figure format and visual style of test.png:
#   - Six groups (cross-dataset transfers ordered by test dataset: BUAA, PURE, UBFC)
#   - Four bars per group: RGB, Red (R), Green (G), Blue (B)
#   - Two-row annotated x-axis (Train / Test)
#   - Dashed separators between test dataset blocks
#   - Standard Error (SE) error bars with numerical values on top of bars
#
# Run cell-by-cell in Jupyter / VS Code interactive window, or:
#   python figures/channel_comparison_plot.py

from __future__ import annotations

import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# 6 cross-dataset transfer pairs ordered by test dataset: BUAA, PURE, UBFC
TRANSFER_PAIRS = [
    ("UBFC_my_in", "BUAA_my_in"),
    ("PURE_my_in", "BUAA_my_in"),
    ("UBFC_my_in", "PURE_my_in"),
    ("BUAA_my_in", "PURE_my_in"),
    ("PURE_my_in", "UBFC_my_in"),
    ("BUAA_my_in", "UBFC_my_in"),
]

# Bar configurations for channels with true RGB colors
CHANNEL_CONFIGS = [
    {
        "channel": "rgb",
        "label": "RGB",
        "weight": 0.01,
        "regions": "all",
        "color": "#37474F",  # Dark Slate / Charcoal for full 3-channel baseline
    },
    {
        "channel": "r",
        "label": "Red (R)",
        "weight": 0.01,
        "regions": "all",
        "color": "#D32F2F",  # Real Red
    },
    {
        "channel": "g",
        "label": "Green (G)",
        "weight": 0.01,
        "regions": "all",
        "color": "#2E7D32",  # Real Green
    },
    {
        "channel": "b",
        "label": "Blue (B)",
        "weight": 0.01,
        "regions": "all",
        "color": "#1976D2",  # Real Blue
    },
]

METRIC_LABELS = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "Std": "Std",
}


def dataset_name(domain: str) -> str:
    """PURE_my_in / PURE_my -> PURE; UBFC_my -> UBFC."""
    for suffix in ("_my_in", "_my_rm", "_my_eye", "_my"):
        if domain.endswith(suffix):
            return domain[: -len(suffix)].upper()
    return domain.upper()


def pair_label(train: str, test: str) -> str:
    return f"{dataset_name(train)}->{dataset_name(test)}"


def get_test_group_spans(transfer_pairs=TRANSFER_PAIRS):
    spans = []
    for i, (_, tgt_in) in enumerate(transfer_pairs):
        test_name = dataset_name(tgt_in)
        if spans and spans[-1][0] == test_name:
            spans[-1] = (test_name, spans[-1][1], i)
        else:
            spans.append((test_name, i, i))
    return spans


def annotate_train_test_axis(ax, x_positions, transfer_pairs=TRANSFER_PAIRS):
    """Two-row x-axis: Train/Test labels on the left, dataset names per group."""
    spans = get_test_group_spans(transfer_pairs)
    train_row_y = -0.10
    test_row_y = -0.20
    left_x = -0.55

    for i, (_, start, _) in enumerate(spans):
        if i > 0:
            boundary = (x_positions[start] + x_positions[start - 1]) / 2.0
            ax.axvline(boundary, color="gray", linestyle="--", alpha=0.35, zorder=0)

    axis_transform = ax.get_xaxis_transform()
    ax.text(
        left_x,
        train_row_y,
        "Train:",
        transform=axis_transform,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        left_x,
        test_row_y,
        "Test:",
        transform=axis_transform,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    for xi, (src_in, _) in enumerate(transfer_pairs):
        ax.text(
            x_positions[xi],
            train_row_y,
            dataset_name(src_in),
            transform=axis_transform,
            ha="center",
            va="center",
            fontsize=9,
        )

    for test_name, start, end in spans:
        center = float(np.mean(x_positions[start : end + 1]))
        ax.text(
            center,
            test_row_y,
            test_name,
            transform=axis_transform,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )


def load_summary_csv(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "r") as f:
        first_line = f.readline()
    if first_line.startswith("Training"):
        df = pd.read_csv(csv_path, skiprows=1)
    else:
        df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    for col in ("Std", "MAE", "RMSE", "MAE_SE", "RMSE_SE"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Regions"] = df["Regions"].astype(str).str.strip()
    df["Source Domain"] = df["Source Domain"].astype(str).str.strip()
    df["Target domain"] = df["Target domain"].astype(str).str.strip()
    if "Channel" in df.columns:
        df["Channel"] = df["Channel"].astype(str).str.strip().str.lower().replace({"all": "rgb"})
    else:
        df["Channel"] = "rgb"
    return df


def lookup_row(
    df: pd.DataFrame,
    *,
    source: str,
    target: str,
    weight: float,
    regions: str,
    channel: str,
) -> pd.Series | None:
    mask = (
        (df["Source Domain"] == source)
        & (df["Target domain"] == target)
        & (df["Regions"] == regions)
        & (df["Channel"] == channel.lower())
        & (np.isclose(df["Weight"], weight, rtol=0.0, atol=1e-9))
    )
    rows = df.loc[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_channel_plot_data(df: pd.DataFrame, metric: str = "MAE", use_se: bool = True):
    se_col = f"{metric}_SE"
    group_labels = []
    values = {cfg["channel"]: [] for cfg in CHANNEL_CONFIGS}
    se_values = {cfg["channel"]: [] for cfg in CHANNEL_CONFIGS}
    missing = []

    for src_in, tgt_in in TRANSFER_PAIRS:
        group_labels.append(pair_label(src_in, tgt_in))
        for cfg in CHANNEL_CONFIGS:
            row = lookup_row(
                df,
                source=src_in,
                target=tgt_in,
                weight=cfg["weight"],
                regions=cfg["regions"],
                channel=cfg["channel"],
            )
            if row is None:
                values[cfg["channel"]].append(np.nan)
                se_values[cfg["channel"]].append(np.nan)
                missing.append(
                    f"{group_labels[-1]} | {cfg['label']} "
                    f"(src={src_in}, tgt={tgt_in}, w={cfg['weight']}, regions={cfg['regions']}, ch={cfg['channel']})"
                )
                continue
            val = float(row[metric])
            values[cfg["channel"]].append(val)
            if use_se and se_col in row.index and pd.notna(row[se_col]):
                se_values[cfg["channel"]].append(float(row[se_col]))
            else:
                se_values[cfg["channel"]].append(np.nan)

    return group_labels, values, se_values, missing


def draw_channel_bars(
    ax,
    group_labels,
    values,
    metric: str = "MAE",
    yerr_values=None,
):
    n_groups = len(group_labels)
    n_bars = len(CHANNEL_CONFIGS)
    x = np.arange(n_groups)
    width = 0.18
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width

    for i, cfg in enumerate(CHANNEL_CONFIGS):
        ys = np.array(values[cfg["channel"]], dtype=float)
        yerr = None
        if yerr_values is not None:
            yerr = np.array(yerr_values[cfg["channel"]], dtype=float)

        bars = ax.bar(
            x + offsets[i],
            ys,
            width=width,
            yerr=yerr,
            capsize=3,
            label=cfg["label"],
            color=cfg["color"],
            edgecolor="white",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"},
        )
        for j, (bar, y) in enumerate(zip(bars, ys)):
            if not np.isfinite(y):
                continue
            err = 0.0
            if yerr is not None and np.isfinite(yerr[j]):
                err = float(yerr[j])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + err + 0.6,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
    annotate_train_test_axis(ax, x)

    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Dynamic upper limit with headroom for error bars and labels
    tops = []
    for key, ys in values.items():
        errs = yerr_values[key] if yerr_values is not None else [0] * len(ys)
        for y, err in zip(ys, errs):
            if np.isfinite(y):
                tops.append(y + (err if np.isfinite(err) else 0.0))
    ymax = max(tops) if tops else 50.0
    ax.set_ylim(0, max(50, ymax * 1.15))


def generate_channel_comparison_figure(
    df: pd.DataFrame,
    out_path: str,
    metric: str = "MAE",
    use_se: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    group_labels, values, se_values, missing = build_channel_plot_data(df, metric=metric, use_se=use_se)
    if missing:
        print(f"Warning: missing CSV rows:")
        for line in missing:
            print(f"  - {line}")

    summary_table = pd.DataFrame(
        {cfg["label"]: values[cfg["channel"]] for cfg in CHANNEL_CONFIGS},
        index=group_labels,
    )

    fig, ax = plt.subplots(figsize=(15, 6))
    draw_channel_bars(ax, group_labels, values, metric=metric, yerr_values=se_values if use_se else None)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.24, left=0.07, right=0.98, top=0.96)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved channel comparison figure to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return summary_table


def main():
    csv_path = os.path.join(_BASE_DIR, "Training_Log", "regions_eval_summary.csv")
    out_path = os.path.join(_SCRIPT_DIR, "channel_comparison_mae.png")

    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    df = load_summary_csv(csv_path)
    print("Loaded summary CSV. Rows:", len(df))
    table = generate_channel_comparison_figure(df, out_path, metric="MAE", use_se=True, show=False)

    print("\nChannel Comparison Table (MAE):")
    print("=" * 60)
    print(table.to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()

# %%
