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

import config

# (infraorbital source, infraorbital target, whole-face source for bar 1)
TRANSFER_PAIRS = [
    ("PURE_my_in", "UBFC_my_in", "PURE_my"),
    ("UBFC_my_in", "PURE_my_in", "UBFC_my"),
    ("BUAA_my_in", "UBFC_my_in", "BUAA_my"),
    ("UBFC_my_in", "BUAA_my_in", "UBFC_my"),
    ("BUAA_my_in", "PURE_my_in", "BUAA_my"),
    ("PURE_my_in", "BUAA_my_in", "PURE_my"),
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
    "r": "Pearson r",
}

METRIC_UNITS = {
    "MAE": "BPM",
    "RMSE": "BPM",
    "Std": "BPM",
    "r": "",
}


def dataset_name(domain: str) -> str:
    """PURE_my_in / PURE_my -> PURE; UBFC_my -> UBFC."""
    for suffix in ("_my_in", "_my_rm", "_my_eye", "_my"):
        if domain.endswith(suffix):
            return domain[: -len(suffix)].upper()
    return domain.upper()


def pair_label(train: str, test: str) -> str:
    return f"{dataset_name(train)}->{dataset_name(test)}"


def load_summary_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    for col in ("Std", "MAE", "RMSE", "r"):
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


# %%
# ============ Config ============
CSV_PATH = os.path.join(config.RESULT_LOG_DIR, "regions_eval_summary.csv")
OUT_PATH = os.path.join(_SCRIPT_DIR, "final_result_summary.png")
METRIC = "RMSE"         # MAE | RMSE | Std | r
TITLE = None              # e.g. "Cross-dataset HR error"
SHOW_PLOT = True          # False when saving only in headless runs

# %%
# ============ Load summary CSV ============
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"Summary CSV not found: {CSV_PATH}")

df = load_summary_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")
df.head()

# %%
# ============ Build grouped bar data ============
group_labels, values, missing = build_plot_data(df, METRIC)

if missing:
    print("Warning: missing CSV rows for:")
    for line in missing:
        print(f"  - {line}")

summary_table = pd.DataFrame(
    {cfg["label"]: values[cfg["key"]] for cfg in BAR_CONFIGS},
    index=group_labels,
)
summary_table

# %%
# ============ Plot ============
n_groups = len(group_labels)
n_bars = len(BAR_CONFIGS)
x = np.arange(n_groups)
width = 0.24
offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width

fig, ax = plt.subplots(figsize=(14, 6))
for i, cfg in enumerate(BAR_CONFIGS):
    ys = np.array(values[cfg["key"]], dtype=float)
    bars = ax.bar(
        x + offsets[i],
        ys,
        width=width,
        label=cfg["label"],
        color=cfg["color"],
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, y in zip(bars, ys):
        if np.isfinite(y):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

ax.set_xticks(x)
ax.set_xticklabels(group_labels, rotation=0)
ax.set_ylabel(METRIC_LABELS.get(METRIC, METRIC))
ax.set_xlabel("Training → Testing")

unit = METRIC_UNITS.get(METRIC, "")
default_title = (
    f"Cross-dataset HR error summary — {METRIC} ({unit})"
    if unit
    else f"Cross-dataset HR error summary — {METRIC}"
)
ax.set_title(TITLE or default_title)
ax.legend(loc="upper left", frameon=True)
ax.grid(True, axis="y", alpha=0.3)

ymax = np.nanmax([v for series in values.values() for v in series if np.isfinite(v)])
if np.isfinite(ymax):
    ax.set_ylim(0, ymax * 1.15)

plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved figure: {OUT_PATH}")

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)

# %%
