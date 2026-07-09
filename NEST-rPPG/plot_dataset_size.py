# %%
# Histogram of video frame counts for PURE, UBFC, and BUAA (infraorbital / *_my_in).
#
# Frame count per subject is read from STMap width (same convention as MyDataset.getIndex).
#
# Run cell-by-cell in Jupyter / VS Code interactive window, or:
#   python plot_dataset_size.py

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import config

# Infraorbital region datasets used in train_regions (STMap_my/*_my_in).
DATASETS = ("PURE_my_in", "UBFC_my_in", "BUAA_my_in")

DATASET_COLORS = {
    "PURE": "#4C72B0",
    "UBFC": "#55A868",
    "BUAA": "#C44E52",
}


def dataset_label(domain: str) -> str:
    """PURE_my_in / PURE_my -> PURE; UBFC_my_in -> UBFC."""
    for suffix in ("_my_in", "_my_rm", "_my_eye", "_my"):
        if domain.endswith(suffix):
            return domain[: -len(suffix)].upper()
    return domain.upper()


def read_frame_count(img_path: str) -> int | None:
    """Return STMap temporal length (width) without decoding full image pixels."""
    try:
        with Image.open(img_path) as img:
            return int(img.size[0])
    except OSError:
        return None


def collect_frame_counts(domain: str, stmap_name: str | None = None) -> list[int]:
    rel_root, _, _ = config.FILEA_NAME[domain]
    stmap_name = stmap_name or config.STMAP_NAME
    data_root = os.path.join(config.STMAP_DATA_ROOT, rel_root.split("/", 1)[1])
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    frame_counts: list[int] = []
    skipped: list[str] = []
    for subject in sorted(os.listdir(data_root)):
        if subject.startswith(".") or subject.endswith(".DS_Store"):
            continue
        subject_dir = os.path.join(data_root, subject)
        if not os.path.isdir(subject_dir):
            continue
        img_path = os.path.join(subject_dir, "STMap", stmap_name)
        frame_count = read_frame_count(img_path)
        if frame_count is None:
            skipped.append(img_path)
            continue
        frame_counts.append(frame_count)

    if skipped:
        print(f"  {dataset_label(domain)}: skipped {len(skipped)} subject(s) with missing/invalid STMap")
    return frame_counts


def collect_all_frame_counts(domains: tuple[str, ...] = DATASETS) -> dict[str, np.ndarray]:
    counts_by_dataset: dict[str, np.ndarray] = {}
    for domain in domains:
        print(f"Scanning {domain} ...")
        counts = collect_frame_counts(domain)
        label = dataset_label(domain)
        if not counts:
            raise RuntimeError(f"No valid STMap images found for {domain}")
        counts_by_dataset[label] = np.asarray(counts, dtype=np.int32)
        print(
            f"  {label}: {len(counts)} subjects, frames min/median/max = "
            f"{counts_by_dataset[label].min()}/{np.median(counts_by_dataset[label]):.0f}/"
            f"{counts_by_dataset[label].max()}"
        )
    return counts_by_dataset


def plot_frame_histogram(
    counts_by_dataset: dict[str, np.ndarray],
    out_path: str,
    *,
    bins: int = 30,
    title: str = "Frame count distribution per subject",
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    all_counts = np.concatenate(list(counts_by_dataset.values()))
    bin_edges = np.histogram_bin_edges(all_counts, bins=bins)

    for label, counts in counts_by_dataset.items():
        ax.hist(
            counts,
            bins=bin_edges,
            alpha=0.55,
            label=f"{label} (n={len(counts)})",
            color=DATASET_COLORS.get(label, None),
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xlabel("Number of frames per subject")
    ax.set_ylabel("Number of subjects")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# %%
# ============ Config ============
OUT_DIR = os.path.join(_SCRIPT_DIR, "figures")
OUT_PATH = os.path.join(OUT_DIR, "dataset_frame_histogram.png")
BINS = 30
SHOW_PLOT = True          # False when saving only in headless runs

# %%
# ============ Collect frame counts ============
counts_by_dataset = collect_all_frame_counts()

summary_table = {
    label: {
        "n_subjects": len(counts),
        "min": int(counts.min()),
        "median": float(np.median(counts)),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
    }
    for label, counts in counts_by_dataset.items()
}
summary_table

# %%
# ============ Frame-count histogram ============
plot_frame_histogram(
    counts_by_dataset,
    OUT_PATH,
    bins=BINS,
    title="Frame count distribution per subject",
    show=SHOW_PLOT,
)
counts_by_dataset

# %%
