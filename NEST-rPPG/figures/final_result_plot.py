# %%
# Bar plot summarizing regions_eval_summary.csv.
#
# Six groups (cross-dataset transfers), three bars each:
#   1. Train whole face (*_my)  -> test infraorbital (*_my_in)
#   2. Train infraorbital       -> test infraorbital, regions=all, weight=0
#   3. Train infraorbital       -> test infraorbital, regions=all, weight=0.01
#
# Run cell-by-cell in Jupyter / VS Code interactive window, or:
#   python figures/final_result_plot.py

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# (infraorbital source, infraorbital target, whole-face source for bar 1)
# Ordered by test dataset: BUAA, then UBFC, then PURE.
TRANSFER_PAIRS = [
    ("UBFC_my_in", "BUAA_my_in", "UBFC_my"),
    ("PURE_my_in", "BUAA_my_in", "PURE_my"),
    ("PURE_my_in", "UBFC_my_in", "PURE_my"),
    ("BUAA_my_in", "UBFC_my_in", "BUAA_my"),
    ("UBFC_my_in", "PURE_my_in", "UBFC_my"),
    ("BUAA_my_in", "PURE_my_in", "BUAA_my"),
]

BAR_CONFIGS = [
    {
        "key": "whole",
        "label": "Whole face → Infraorbital",
        "regions": "whole",
        "weight": 0.0,
        "use_whole_src": True,
        "color": "#4C72B0",
    },
    {
        "key": "infra",
        "label": "Infraorbital → Infraorbital (without contrastive learning)",
        "regions": "all",
        "weight": 0.0,
        "use_whole_src": False,
        "color": "#55A868",
    },
    {
        "key": "infra_w001",
        "label": "Infraorbital → Infraorbital (with contrastive learning)",
        "regions": "all",
        "weight": 0.01,
        "use_whole_src": False,
        "color": "#C44E52",
    },
]

METRIC_LABELS = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "Std": "Std",
}

METRIC_UNITS = {
    "MAE": "BPM",
    "RMSE": "BPM",
    "Std": "BPM",
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
    for i, (_, tgt_in, _) in enumerate(transfer_pairs):
        test_name = dataset_name(tgt_in)
        if spans and spans[-1][0] == test_name:
            spans[-1] = (test_name, spans[-1][1], i)
        else:
            spans.append((test_name, i, i))
    return spans


def annotate_test_groups(ax, x_positions):
    spans = get_test_group_spans()
    for i, (test_name, start, end) in enumerate(spans):
        if i > 0:
            boundary = (x_positions[start] + x_positions[start - 1]) / 2.0
            ax.axvline(boundary, color="gray", linestyle="--", alpha=0.35, zorder=0)
        center = float(np.mean(x_positions[start : end + 1]))
        ax.text(
            center,
            -0.14,
            f"Test: {test_name}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )


def load_summary_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    for col in ("Std", "MAE", "RMSE"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Regions"] = df["Regions"].astype(str).str.strip()
    df["Source Domain"] = df["Source Domain"].astype(str).str.strip()
    df["Target domain"] = df["Target domain"].astype(str).str.strip()
    return df


def lookup_row(
    df: pd.DataFrame,
    *,
    source: str,
    target: str,
    weight: float,
    regions: str,
) -> pd.Series | None:
    mask = (
        (df["Source Domain"] == source)
        & (df["Target domain"] == target)
        & (df["Regions"] == regions)
        & (np.isclose(df["Weight"], weight, rtol=0.0, atol=1e-9))
    )
    rows = df.loc[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_plot_data(df: pd.DataFrame, metric: str):
    group_labels = []
    values = {cfg["key"]: [] for cfg in BAR_CONFIGS}
    missing = []

    for src_in, tgt_in, src_whole in TRANSFER_PAIRS:
        group_labels.append(pair_label(src_in, tgt_in))
        for cfg in BAR_CONFIGS:
            source = src_whole if cfg["use_whole_src"] else src_in
            row = lookup_row(
                df,
                source=source,
                target=tgt_in,
                weight=cfg["weight"],
                regions=cfg["regions"],
            )
            if row is None:
                values[cfg["key"]].append(np.nan)
                missing.append(
                    f"{group_labels[-1]} | {cfg['label']} "
                    f"(src={source}, tgt={tgt_in}, w={cfg['weight']}, regions={cfg['regions']})"
                )
                continue
            values[cfg["key"]].append(float(row[metric]))

    return group_labels, values, missing


def build_plot_data_mae_std(df: pd.DataFrame):
    group_labels = []
    mae_values = {cfg["key"]: [] for cfg in BAR_CONFIGS}
    std_values = {cfg["key"]: [] for cfg in BAR_CONFIGS}
    missing = []

    for src_in, tgt_in, src_whole in TRANSFER_PAIRS:
        group_labels.append(pair_label(src_in, tgt_in))
        for cfg in BAR_CONFIGS:
            source = src_whole if cfg["use_whole_src"] else src_in
            row = lookup_row(
                df,
                source=source,
                target=tgt_in,
                weight=cfg["weight"],
                regions=cfg["regions"],
            )
            if row is None:
                mae_values[cfg["key"]].append(np.nan)
                std_values[cfg["key"]].append(np.nan)
                missing.append(
                    f"{group_labels[-1]} | {cfg['label']} "
                    f"(src={source}, tgt={tgt_in}, w={cfg['weight']}, regions={cfg['regions']})"
                )
                continue
            mae_values[cfg["key"]].append(float(row["MAE"]))
            std_values[cfg["key"]].append(float(row["Std"]))

    return group_labels, mae_values, std_values, missing


def draw_grouped_bars(
    ax,
    group_labels,
    values,
    metric: str,
    *,
    yerr_values=None,
    title: str | None = None,
):
    n_groups = len(group_labels)
    n_bars = len(BAR_CONFIGS)
    x = np.arange(n_groups)
    width = 0.24
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width

    for i, cfg in enumerate(BAR_CONFIGS):
        ys = np.array(values[cfg["key"]], dtype=float)
        yerr = None
        if yerr_values is not None:
            yerr = np.array(yerr_values[cfg["key"]], dtype=float)
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
                y + err,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=0)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlabel("Training → Testing")
    annotate_test_groups(ax, x)

    unit = METRIC_UNITS.get(metric, "")
    default_title = (
        f"Cross-dataset HR error summary — {metric} ({unit})"
        if unit
        else f"Cross-dataset HR error summary — {metric}"
    )
    ax.set_title(title or default_title)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)

    ymax = np.nanmax([v for series in values.values() for v in series if np.isfinite(v)])
    if yerr_values is not None:
        tops = []
        for key, ys in values.items():
            errs = yerr_values[key]
            for y, err in zip(ys, errs):
                if np.isfinite(y):
                    tops.append(y + (err if np.isfinite(err) else 0.0))
        if tops:
            ymax = max(ymax, np.nanmax(tops))
    if np.isfinite(ymax):
        ax.set_ylim(0, ymax * 1.15)


def save_grouped_bar_figure(
    df: pd.DataFrame,
    metric: str,
    out_path: str,
    *,
    title: str | None = None,
    show: bool = False,
) -> pd.DataFrame:
    group_labels, values, missing = build_plot_data(df, metric)
    if missing:
        print(f"Warning: missing CSV rows for {metric}:")
        for line in missing:
            print(f"  - {line}")

    summary_table = pd.DataFrame(
        {cfg["label"]: values[cfg["key"]] for cfg in BAR_CONFIGS},
        index=group_labels,
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    draw_grouped_bars(ax, group_labels, values, metric, title=title)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return summary_table


# %%
# ============ Config ============
CSV_PATH = os.path.join(_BASE_DIR, "regions_eval_summary (1).csv")
OUT_PATH_RMSE = os.path.join(_SCRIPT_DIR, "final_result_summary.png")
OUT_PATH_MAE_STD = os.path.join(_SCRIPT_DIR, "final_result_summary_mae_std.png")
SHOW_PLOT = True          # False when saving only in headless runs

# %%
# ============ Load summary CSV ============
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"Summary CSV not found: {CSV_PATH}")

df = load_summary_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")
df.head()

# %%
# ============ RMSE plot ============
summary_table_rmse = save_grouped_bar_figure(
    df,
    "RMSE",
    OUT_PATH_RMSE,
    show=SHOW_PLOT,
)
summary_table_rmse

# %%
# ============ MAE plot (Std as error bars) ============
group_labels, mae_values, std_values, missing = build_plot_data_mae_std(df)
if missing:
    print("Warning: missing CSV rows for MAE/Std:")
    for line in missing:
        print(f"  - {line}")

summary_table_mae = pd.DataFrame(
    {cfg["label"]: mae_values[cfg["key"]] for cfg in BAR_CONFIGS},
    index=group_labels,
)
summary_table_std = pd.DataFrame(
    {cfg["label"]: std_values[cfg["key"]] for cfg in BAR_CONFIGS},
    index=group_labels,
)

fig, ax = plt.subplots(figsize=(14, 6))
draw_grouped_bars(
    ax,
    group_labels,
    mae_values,
    "MAE",
    yerr_values=std_values,
    title="Cross-dataset HR error summary — MAE ± Std (BPM)",
)
plt.tight_layout()
fig.subplots_adjust(bottom=0.18)
os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH_MAE_STD)), exist_ok=True)
fig.savefig(OUT_PATH_MAE_STD, dpi=150, bbox_inches="tight")
print(f"Saved figure: {OUT_PATH_MAE_STD}")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)

summary_table_mae_std = pd.concat({"MAE": summary_table_mae, "Std": summary_table_std}, axis=1)
summary_table_mae_std

# %%
