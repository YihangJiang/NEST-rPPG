# %%
#!/usr/bin/env python3
"""
Jupyter-friendly script: two figures in order.

1) One align frame per ROI (rm / in / eye) and the 5×5-pooled one-column STMap
   vector from each frame (paths derived from BASE_FRAME_PATH under *_in).
2) Three separate figures: full STMap per ROI (rm, in, eye) for the same session.
Run as script: python plot_frame_to_stmap_change.py
Run in Jupyter: run cells top to bottom.
"""

# %%
import os
from typing import Dict, Optional, Tuple

import matplotlib as mpl

_mpl_backend = os.environ.get("MPLBACKEND")
if _mpl_backend:
    mpl.use(_mpl_backend, force=True)

import cv2
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _show_figure(fig=None) -> None:
    """Avoid blocking when running headless (e.g. MPLBACKEND=Agg)."""
    if fig is None:
        fig = plt.gcf()
    backend = (mpl.get_backend() or "").lower()
    if "agg" in backend:
        plt.close(fig)
    else:
        plt.show()


# %%
# ============ Cell: paths & options (edit here) ============
BASE_FRAME_PATH = "/home/yj167/Desktop/NEST-rPPG/STMap_my/BUAA_my_in/Sub_07lux 100.0/Align/10000.png"
STMAP_FILENAME = "STMap_RGB.png"  # under <session>/STMap/
OUTPUT_DIR = None  # None -> default per-figure dirs next to data
SAVE_FIGURE = False


# %%
def draw_5x5_grid(img_bgr: np.ndarray) -> np.ndarray:
    """Return a copy of image with a 5x5 grid overlay."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    w_step = int(w / 5)
    h_step = int(h / 5)
    color = (0, 255, 255)  # yellow in BGR
    thickness = 1

    for i in range(1, 5):
        x = i * w_step
        cv2.line(out, (x, 0), (x, h - 1), color, thickness)
    for j in range(1, 5):
        y = j * h_step
        cv2.line(out, (0, y), (w - 1, y), color, thickness)
    return out


def frame_to_stmap_vector(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Build one STMap time-column vector from a single frame using 5x5 pooling.
    Returns shape (25, 3) in BGR.
    """
    h, w = frame_bgr.shape[:2]
    w_step = int(w / 5)
    h_step = int(h / 5)
    vec = []
    for w_idx in range(5):
        for h_idx in range(5):
            cell = frame_bgr[
                h_idx * h_step: (h_idx + 1) * h_step,
                w_idx * w_step: (w_idx + 1) * w_step,
                :,
            ]
            pooled = np.nanmean(np.nanmean(cell, axis=0), axis=0)
            vec.append(pooled)
    return np.array(vec, dtype=np.float32)


def vector_to_rgb_strip(vec_bgr: np.ndarray, width: int = 40) -> np.ndarray:
    """Convert (25,3) BGR vector to displayable RGB strip image."""
    vec_rgb = vec_bgr[:, ::-1]  # BGR -> RGB
    vec_rgb = np.clip(vec_rgb, 0, 255).astype(np.uint8)  # (25, 3)
    vec_rgb_img = vec_rgb.reshape(25, 1, 3)
    vec_rgb_img = np.repeat(vec_rgb_img, width, axis=1)
    return vec_rgb_img


def resolve_region_frame_paths(base_frame_path: str) -> Dict[str, str]:
    """
    Build frame paths for rm/in/eye from an `_in` base path.
    Example:
      .../BUAA_my_in/.../10000.png -> BUAA_my_rm and BUAA_my_eye counterparts.
    """
    base = os.path.abspath(base_frame_path)
    domain_in = None
    for tok in ("BUAA_my_in", "PURE_my_in", "UBFC_my_in"):
        if tok in base:
            domain_in = tok
            break
    if domain_in is None:
        raise ValueError(
            "BASE_FRAME_PATH must contain BUAA_my_in, PURE_my_in, or UBFC_my_in."
        )
    return {
        "rm": base.replace(domain_in, domain_in.replace("_in", "_rm")),
        "in": base,
        "eye": base.replace(domain_in, domain_in.replace("_in", "_eye")),
    }


def resolve_three_stmap_paths_from_in_align_frame(
    base_frame_path: str,
    stmap_filename: str = STMAP_FILENAME,
) -> Dict[str, str]:
    """
    STMap PNG paths for the three ROI videos that match the `_in` align frame.

    Expects layout: .../<dataset>_in/<session>/Align/<frame>.png
    STMaps live at: .../<dataset>_rm|_in|_eye/<session>/STMap/<filename>
    """
    base = os.path.abspath(base_frame_path)
    session_dir = os.path.dirname(os.path.dirname(base))

    domain_in = None
    for tok in ("BUAA_my_in", "PURE_my_in", "UBFC_my_in"):
        if tok in session_dir:
            domain_in = tok
            break
    if domain_in is None:
        raise ValueError(
            "Session path must contain BUAA_my_in, PURE_my_in, or UBFC_my_in "
            f"(got {session_dir!r})."
        )

    rel_stmap = os.path.join("STMap", stmap_filename)
    return {
        "rm": os.path.join(session_dir.replace(domain_in, domain_in.replace("_in", "_rm")), rel_stmap),
        "in": os.path.join(session_dir.replace(domain_in, domain_in.replace("_in", "_in")), rel_stmap),
        "eye": os.path.join(session_dir.replace(domain_in, domain_in.replace("_in", "_eye")), rel_stmap),
    }


def show_three_region_stmaps_separate(
    region_images_rgb: Dict[str, np.ndarray],
    region_titles: Dict[str, str],
    stmap_basename: str,
    save_dir: Optional[str],
) -> None:
    """One matplotlib figure per ROI (rm, in, eye)."""
    order: Tuple[str, ...] = ("rm", "in", "eye")
    for key in order:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.imshow(region_images_rgb[key], aspect="auto", interpolation="nearest")
        ax.set_title(region_titles[key])
        ax.axis("off")
        plt.tight_layout()
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"stmap_{key}_{stmap_basename}.png")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print("Saved figure:", save_path)
        _show_figure(fig)


def show_frame_and_three_region_vectors(
    region_frames_bgr: Dict[str, np.ndarray],
    region_vectors_bgr: Dict[str, np.ndarray],
    frame_name: str,
    save_path: Optional[str],
) -> None:
    """Show 3 region frames-with-grid on left and 3 vectors on right."""
    region_order = ["eye", "in", "rm"]
    region_frame_titles = {
        "eye": "eye region frame with 5x5 grid",
        "in": "infraorbital region frame with 5x5 grid",
        "rm": "malar region frame with 5x5 grid",
    }
    region_vector_titles = {
        "eye": "eye region Spatial Temporal Map vector",
        "in": "infraorbital region Spatial Temporal Map vector",
        "rm": "malar region Spatial Temporal Map vector",
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    for r, region in enumerate(region_order):
        frame_grid_bgr = draw_5x5_grid(region_frames_bgr[region])
        frame_grid_rgb = cv2.cvtColor(frame_grid_bgr, cv2.COLOR_BGR2RGB)
        axes[r, 0].imshow(frame_grid_rgb)
        axes[r, 0].set_title(f"{region_frame_titles[region]} ({frame_name})")
        axes[r, 0].axis("off")

        strip = vector_to_rgb_strip(region_vectors_bgr[region], width=40)
        axes[r, 1].imshow(strip, aspect="auto")
        axes[r, 1].set_title(region_vector_titles[region])
        axes[r, 1].set_xlabel("RGB value")
        axes[r, 1].set_ylabel("Grid cell (0..24)")
        axes[r, 1].set_xticks([])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved figure:", save_path)
    _show_figure(fig)


def run_figure_frames_and_vectors() -> None:
    """Figure 1: aligned frames + 5×5 pooled vectors (rm / in / eye)."""
    base_frame_path = os.path.abspath(BASE_FRAME_PATH)
    region_paths = resolve_region_frame_paths(base_frame_path)

    for region, path in region_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {region} frame: {path}")

    region_frames_bgr: Dict[str, np.ndarray] = {}
    region_vectors_bgr: Dict[str, np.ndarray] = {}
    for region, path in region_paths.items():
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read {region} frame: {path}")
        region_frames_bgr[region] = img
        region_vectors_bgr[region] = frame_to_stmap_vector(img)
        print(f"{region} frame: {path}")

    frame_name = os.path.basename(base_frame_path)
    save_path = None
    if SAVE_FIGURE:
        if OUTPUT_DIR is None:
            output_dir = os.path.join(os.path.dirname(base_frame_path), "frame_to_stmap_plots")
        else:
            output_dir = os.path.abspath(OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "frame_and_three_region_vectors.png")

    show_frame_and_three_region_vectors(
        region_frames_bgr=region_frames_bgr,
        region_vectors_bgr=region_vectors_bgr,
        frame_name=frame_name,
        save_path=save_path,
    )


def run_figure_three_stmaps() -> None:
    """Figures 2–4: one full STMap figure per ROI (rm / in / eye), same session as BASE_FRAME_PATH."""
    base_frame_path = os.path.abspath(BASE_FRAME_PATH)
    paths = resolve_three_stmap_paths_from_in_align_frame(base_frame_path, STMAP_FILENAME)

    for region, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {region} STMap: {path}")

    region_rgb: Dict[str, np.ndarray] = {}
    for region, path in paths.items():
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read {region}: {path}")
        region_rgb[region] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"{region} STMap: {path} shape={region_rgb[region].shape}")

    session_dir = os.path.dirname(os.path.dirname(base_frame_path))
    session_name = os.path.basename(session_dir)

    titles = {
        "rm": f"STMap — malar (rm) — {session_name}\n{STMAP_FILENAME}",
        "in": f"STMap — infraorbital (in) — {session_name}\n{STMAP_FILENAME}",
        "eye": f"STMap — eye (periorbital) — {session_name}\n{STMAP_FILENAME}",
    }

    save_dir = None
    if SAVE_FIGURE:
        if OUTPUT_DIR is None:
            save_dir = os.path.join(session_dir, "STMap", "three_region_stmap_plots")
        else:
            save_dir = os.path.abspath(OUTPUT_DIR)
        os.makedirs(save_dir, exist_ok=True)

    stmap_base, _ = os.path.splitext(os.path.basename(STMAP_FILENAME))
    show_three_region_stmaps_separate(region_rgb, titles, stmap_base, save_dir)


# %%
# Figure 1 — frames + pooled vectors
run_figure_frames_and_vectors()

# %%
# Figures 2–4 — one STMap figure per ROI (rm, then in, then eye)
run_figure_three_stmaps()

# %%
