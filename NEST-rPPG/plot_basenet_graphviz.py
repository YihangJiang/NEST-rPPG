# %%
#!/usr/bin/env python3
"""
Render BaseNet with Graphviz.

Two modes:
1) STRUCTURE GRAPH (recommended): clean module-level architecture diagram.
   - Requires: pip install graphviz
   - Requires: system graphviz binary (`dot`) available (e.g. apt install graphviz)

2) COMPUTATION GRAPH (torchviz): full autograd graph (often huge/ugly).
   - Requires: pip install torchviz graphviz
   - Requires: system graphviz binary (`dot`)
"""

import os

import torch

import model


def main() -> None:
    # ---- Hard-coded settings (edit if needed) ----
    MAKE_STRUCTURE_GRAPH = True
    MAKE_COMPUTATION_GRAPH = False

    FRAMES_NUM = 512          # STMap temporal length (W)
    H = 25                   # STMap spatial rows (H)
    BATCH = 1
    DEVICE = "cpu"           # change to "cuda:0" if you want
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphviz_out")
    OUT_NAME_STRUCTURE = "basenet_structure"
    OUT_NAME_COMPUTATION = "basenet_computation"
    # ---------------------------------------------

    os.makedirs(OUT_DIR, exist_ok=True)

    net = model.BaseNet().to(DEVICE).eval()

    if MAKE_STRUCTURE_GRAPH:
        try:
            from graphviz import Digraph
        except Exception as e:
            raise RuntimeError(
                "graphviz python package is required. Install with: pip install graphviz"
            ) from e

        g = Digraph("BaseNet", format="png")
        g.attr(rankdir="LR", fontsize="16")

        def add(name: str, label: str):
            g.node(name, label=label, shape="box", style="rounded,filled", fillcolor="#eef6ff")

        def link(a: str, b: str, label: str | None = None):
            if label:
                g.edge(a, b, label=label)
            else:
                g.edge(a, b)

        # Encoder (ResNet-18 backbone)
        add("input", f"Input STMap\\n(B, 3, {H}, {FRAMES_NUM})")
        add("stem", "conv1 + bn1 + relu")
        add("l1", "layer1")
        add("l2", "layer2")
        add("l3", "layer3")
        add("l4", "layer4 (em)")

        # Feature descriptor path (av)
        add("av1", "get_av(layer1)")
        add("av2", "get_av(layer2)")
        add("av3", "get_av(layer3)")
        add("av4", "get_av(layer4)")
        add("avcat", "av = cat([av1,av2,av3,av4])")

        # HR head
        add("avgpool", "AdaptiveAvgPool2d(1,1)")
        add("fc", "fc(512→1)\\nHR")

        # BVP/Sig decoder
        add("up1", "up1\\nConvT + BasicBlock")
        add("up2", "up2\\nConvT + BasicBlock")
        add("up3", "up3\\nConvT + BasicBlock")
        add("up4", "up4\\nConvT + BasicBlock")
        add("sig", "Sig (B, H?, T?)\\n.squeeze(1)")

        # Main flow
        link("input", "stem")
        link("stem", "l1")
        link("l1", "l2")
        link("l2", "l3")
        link("l3", "l4")

        # av taps
        link("l1", "av1")
        link("l2", "av2")
        link("l3", "av3")
        link("l4", "av4")
        link("av1", "avcat")
        link("av2", "avcat")
        link("av3", "avcat")
        link("av4", "avcat")

        # HR head
        link("l4", "avgpool")
        link("avgpool", "fc")

        # Sig decoder
        link("l4", "up1")
        link("up1", "up2")
        link("up2", "up3")
        link("up3", "up4")
        link("up4", "sig")

        out_path = os.path.join(OUT_DIR, OUT_NAME_STRUCTURE)
        g.render(out_path, cleanup=False)
        print("Saved structure graph:")
        print(" -", out_path + ".png")

    if MAKE_COMPUTATION_GRAPH:
        # STMap tensor shape used in training is typically (B, 3, 25, T)
        x = torch.zeros((BATCH, 3, H, FRAMES_NUM), dtype=torch.float32, device=DEVICE, requires_grad=True)

        # IMPORTANT: do NOT use no_grad(), we want an autograd graph.
        sig, hr, av = net(x)
        y = sig.mean() + hr.mean() + av.mean()

        try:
            from torchviz import make_dot
        except Exception as e:
            raise RuntimeError(
                "torchviz is required. Install with: pip install torchviz graphviz"
            ) from e

        dot = make_dot(y)  # omit params to avoid massive parameter subgraphs
        dot.format = "png"
        out_path = os.path.join(OUT_DIR, OUT_NAME_COMPUTATION)
        dot.render(out_path, cleanup=False)
        print("Saved computation graph:")
        print(" -", out_path + ".png")

    print("Note: if rendering fails, install system Graphviz: `sudo apt install graphviz`")


if __name__ == "__main__":
    main()


# %%
