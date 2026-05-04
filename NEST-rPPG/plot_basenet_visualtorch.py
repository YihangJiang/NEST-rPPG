# %%
#!/usr/bin/env python3
"""
Plot BaseNet architecture using visualtorch.

This produces a clean layer diagram (not the full autograd graph).

Requirements:
  pip install visualtorch pillow

Notes:
  - visualtorch uses forward hooks and an example input shape to infer tensor sizes.
  - If a layer cannot be inferred, try reducing FRAMES_NUM or switching to CPU.
"""

import os
# %%
import torch

import model

# %%
# ----- Settings (edit these) -----
H = 25
FRAMES_NUM = 512
DEVICE = "cpu"  # "cuda:0" also works if available

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphviz_out")
OUT_PNG = os.path.join(OUT_DIR, "basenet_visualtorch.png")

# IMPORTANT: BaseNet expects a 4D tensor for BatchNorm2d: (B, C, H, W).
# visualtorch will call `model(dummy_input)` where dummy_input has shape INPUT_SHAPE,
# so include the batch dimension here.
INPUT_SHAPE = (1, 3, H, FRAMES_NUM)
# ---------------------------------

# %%
os.makedirs(OUT_DIR, exist_ok=True)
net = model.BaseNet().to(DEVICE).eval()

# %%
try:
    from visualtorch import lenet_view
except ModuleNotFoundError as e:
    raise RuntimeError(
        "visualtorch is not installed. Install with: pip install visualtorch pillow"
    ) from e

# %%
img = lenet_view(
    net,
    input_shape=INPUT_SHAPE,
)

# %%
img.save(OUT_PNG)
print("Saved visualtorch diagram:")
print(" -", OUT_PNG)


if __name__ == "__main__":
    pass

# %%
