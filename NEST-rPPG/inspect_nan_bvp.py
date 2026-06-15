# %%
import os

import scipy.io as scio
# %%
import config
import MyDataset

import tqdm
import numpy as np

print(f"config.STMAP_INDEX_BASE: {config.STMAP_INDEX_BASE}")

def find_nan_segments(domain: str, frames_num: int = 256):
    """
    Scan index .mat files for a domain and report any BVP segments containing NaNs.

    You can call this directly from a Jupyter cell, e.g.:

        from inspect_nan_bvp import find_nan_segments
        bad = find_nan_segments("PURE_my_in", frames_num=256)
    """
    index_dir = os.path.join(config.STMAP_INDEX_BASE, domain)
    if not os.path.isdir(index_dir):
        print(f"Index directory not found: {index_dir}")
        return []

    files = sorted(f for f in os.listdir(index_dir) if f.endswith(".mat"))
    print(f"[inspect_nan_bvp] Scanning {len(files)} index files under {index_dir} for NaNs in BVP...")

    # Minimal args object for Data_DG (no augmentation)
    dummy_args = type("Args", (), {
        "spatial_aug_rate": 0.0,
        "temporal_aug_rate": 0.0,
    })()

    bad = []

    for fname in tqdm.tqdm(files, desc=f"Scanning {domain}", unit="file"):
        fpath = os.path.join(index_dir, fname)
        try:
            idx = scio.loadmat(fpath)
            now_path = str(idx["Path"][0])         # subject folder, e.g. STMap_my/PURE_my_in/01-01
            step_idx = int(np.asarray(idx["Step_Index"]).flat[0])
        except Exception as e:
            print(f"[inspect_nan_bvp] Warning: failed to read index file {fpath}: {repr(e)}")
            continue

        # Use Data_DG.getLabel to load the same segment as in training
        ds = MyDataset.Data_DG(
            root_dir=index_dir,
            dataName=domain,
            STMap=config.STMAP_NAME,
            frames_num=frames_num,
            args=dummy_args,
        )
        try:
            gt, bvp = ds.getLabel(now_path, step_idx)
        except Exception as e:
            print(f"[inspect_nan_bvp] Warning: getLabel failed for {now_path} Step_Index={step_idx}: {repr(e)}")
            continue

        bvp_arr = np.asarray(bvp, dtype=np.float32).reshape(-1)
        if np.isnan(bvp_arr).any():
            bad.append((fname, now_path, step_idx))
            print(f"NaN BVP in {fname}: subject={now_path}, Step_Index={step_idx}")

    if not bad:
        print(f"[inspect_nan_bvp] No NaNs found for domain {domain}.")
    else:
        print(f"\n[inspect_nan_bvp] Total NaN segments for {domain}: {len(bad)}")
    return bad


# Convenience defaults so you can run this file directly in a Jupyter cell
# by simply executing it (without argparse).
#
# Example in a cell:
#   %run inspect_nan_bvp.py
#
# It will scan PURE_my_in with frames_num=256. Edit these two lines if you
# want different defaults.
DEFAULT_DOMAIN = "PURE_my_in"
DEFAULT_FRAMES_NUM = 256


if __name__ == "__main__":
    find_nan_segments(DEFAULT_DOMAIN, frames_num=DEFAULT_FRAMES_NUM)


# %%
