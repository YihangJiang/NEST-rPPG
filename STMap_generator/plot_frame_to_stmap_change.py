# %%
#!/usr/bin/env python3
"""
Pick one BUAA frame and plot STMap vectors for three regions (rm/in/eye).
"""

import os
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ------------------ Hard-coded settings ------------------
# Edit these paths/values directly before running.
BASE_FRAME_PATH = "/home/yj167/Desktop/NEST-rPPG/STMap_my/BUAA_my_in/Sub_07lux 100.0/Align/10000.png"
OUTPUT_DIR = None  # None -> <subject>/frame_to_stmap_plots
SAVE_FIGURE = False
# ---------------------------------------------------------


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
    if "BUAA_my_in" not in base_frame_path:
        raise ValueError("BASE_FRAME_PATH must contain 'BUAA_my_in'.")
    return {
        "rm": base_frame_path.replace("BUAA_my_in", "BUAA_my_rm"),
        "in": base_frame_path,
        "eye": base_frame_path.replace("BUAA_my_in", "BUAA_my_eye"),
    }


def show_frame_and_three_region_vectors(
    region_frames_bgr: Dict[str, np.ndarray],
    region_vectors_bgr: Dict[str, np.ndarray],
    frame_name: str,
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
    if SAVE_FIGURE:
        if OUTPUT_DIR is None:
            output_dir = os.path.join(os.path.dirname(BASE_FRAME_PATH), "frame_to_stmap_plots")
        else:
            output_dir = os.path.abspath(OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "frame_and_three_region_vectors.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print("Saved figure:", out_path)
    plt.show()


def main() -> None:
    base_frame_path = os.path.abspath(BASE_FRAME_PATH)
    region_paths = resolve_region_frame_paths(base_frame_path)

    for region, path in region_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {region} frame: {path}")

    region_frames_bgr = {}
    region_vectors_bgr = {}
    for region, path in region_paths.items():
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read {region} frame: {path}")
        region_frames_bgr[region] = img
        region_vectors_bgr[region] = frame_to_stmap_vector(img)
        print(f"{region} frame: {path}")

    frame_name = os.path.basename(base_frame_path)
    show_frame_and_three_region_vectors(
        region_frames_bgr=region_frames_bgr,
        region_vectors_bgr=region_vectors_bgr,
        frame_name=frame_name,
    )


if __name__ == "__main__":
    main()

# %%
