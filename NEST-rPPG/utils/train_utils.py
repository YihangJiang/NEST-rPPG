# -*- coding: UTF-8 -*-
""" Training helpers: example-figure saving, etc. """
import os
import numpy as np
import scipy.io as scio
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _extract_subject_id(index_file_path):
    """Extract subject ID from index .mat file path."""
    temp = scio.loadmat(index_file_path)
    path_str = str(temp['Path'][0])
    # Extract subject folder name (e.g., "01-01" from "./STMap/PURE/01-01")
    subject_id = os.path.basename(os.path.normpath(path_str))
    return subject_id


def _get_stmap_path_and_size(index_file_path, stmap_name):
    """Get absolute path and original dimensions of STMap from index file."""
    temp = scio.loadmat(index_file_path)
    path_str = str(temp['Path'][0])
    stmap_dir = os.path.join(path_str, 'STMap')
    stmap_file = os.path.join(stmap_dir, stmap_name)
    abs_path = os.path.abspath(stmap_file)
    if os.path.exists(abs_path):
        img = cv2.imread(abs_path)
        if img is not None:
            h, w, c = img.shape
            return abs_path, (h, w)
    return abs_path, None


def save_example_figure(data, bvp, gt, domain_name, save_dir, filename, subject_id=None, original_size=None):
    """
    Save one figure: STMap + BVP for the first sample in a batch.

    Args:
        data: (1, C, H, W) tensor
        bvp: (1, T) tensor
        gt: (1,) or scalar tensor (HR)
        domain_name: str for title
        save_dir: directory to save the image
        filename: output filename (e.g. 'tgt_loaded.png')
        subject_id: optional subject ID string to include in title
        original_size: optional tuple (H, W) of original STMap dimensions
    """
    d = data[0].detach().cpu().numpy()
    b = np.asarray(bvp[0].detach().cpu().numpy()).ravel()
    g = float(np.asarray(gt[0].detach().cpu().numpy()).flat[0])
    if d.shape[0] == 3:
        img = np.transpose(d, (1, 2, 0))
        img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
    else:
        img = d[0]
        img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(img, aspect='auto')
    # img shape: (H, W) where H=spatial regions, W=time frames (after transform)
    spatial_h, spatial_w = img.shape[0], img.shape[1]
    if original_size:
        # Use original dimensions for display
        title_stmap = '%s — STMap [%d×%d]' % (domain_name, original_size[0], original_size[1])
    else:
        title_stmap = '%s — STMap [%d×%d]' % (domain_name, spatial_h, spatial_w)
    if subject_id:
        title_stmap += ' (Subject: %s)' % subject_id
    axes[0].set_title(title_stmap)
    axes[0].set_xlabel('Time (frames)')
    axes[0].set_ylabel('Spatial region')
    axes[1].plot(b, color='C0')
    title_bvp = 'BVP (GT HR %.1f)' % g
    if subject_id:
        title_bvp += ' — Subject: %s' % subject_id
    axes[1].set_title(title_bvp)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=120)
    plt.close(fig)


def save_example_figures(root_file, tgt_loader, src_loaders, source_configs, target_name, tgt_db=None, source_dbs=None, target_map=None, example_fig_dir=None):
    """
    Save example figures for target and source domains to STMap_Index/example_fig/.

    Creates:
        - tgt_loaded.png: one sample from target domain (STMap + BVP)
        - src_loaded.png: one sample per source domain in a single figure

    Args:
        root_file: e.g. './STMap/'
        tgt_loader: DataLoader for target
        src_loaders: list of DataLoaders for sources
        source_configs: list of dicts with 'name' key
        target_name: str, target domain name
        tgt_db: optional target dataset (to get subject ID)
        source_dbs: optional list of source datasets (to get subject IDs)
        target_map: optional target STMap filename (e.g. 'STMap.png')
        example_fig_dir: optional; if set, save figures here instead of root_file/STMap_Index/example_fig
    """
    if example_fig_dir is None:
        example_fig_dir = os.path.join(root_file, 'STMap_Index', 'example_fig')
    os.makedirs(example_fig_dir, exist_ok=True)

    # Target: one example
    tgt_batch = next(iter(tgt_loader))
    tgt_subject_id = None
    tgt_stmap_path = None
    tgt_stmap_size = None
    if tgt_db is not None:
        # Get first sample's index file path to extract subject ID and STMap info
        tgt_idx_file = os.path.join(tgt_db.root_dir, tgt_db.datalist[0])
        tgt_subject_id = _extract_subject_id(tgt_idx_file)
        # Get STMap path and original size
        stmap_name = target_map if target_map else (source_configs[0]['map'] if source_configs else 'STMap.png')
        tgt_stmap_path, tgt_stmap_size = _get_stmap_path_and_size(tgt_idx_file, stmap_name)
        if tgt_stmap_size:
            print("Target STMap: %s | Original size: %d×%d (spatial×time)" % (tgt_stmap_path, tgt_stmap_size[0], tgt_stmap_size[1]))
    save_example_figure(
        tgt_batch[0], tgt_batch[1], tgt_batch[2],
        target_name, example_fig_dir, 'tgt_loaded.png', subject_id=tgt_subject_id, 
        original_size=tgt_stmap_size
    )

    # Sources: one combined figure (src_loaded.png)
    num_src = len(source_configs)
    fig_src, axes_src = plt.subplots(num_src, 2, figsize=(10, 3 * num_src))
    if num_src == 1:
        axes_src = axes_src[None, :]
    for i, cfg in enumerate(source_configs):
        src_batch = next(iter(src_loaders[i]))
        d = src_batch[0][0].detach().cpu().numpy()
        b = src_batch[1][0].detach().cpu().numpy().ravel()
        g = float(np.asarray(src_batch[2][0].detach().cpu().numpy()).flat[0])
        if d.shape[0] == 3:
            img = np.transpose(d, (1, 2, 0))
            img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
        else:
            img = d[0]
            img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
        # Get subject ID and STMap info if dataset available
        src_subject_id = None
        src_stmap_size = None
        if source_dbs is not None and i < len(source_dbs):
            src_idx_file = os.path.join(source_dbs[i].root_dir, source_dbs[i].datalist[0])
            src_subject_id = _extract_subject_id(src_idx_file)
            src_stmap_path, src_stmap_size = _get_stmap_path_and_size(src_idx_file, cfg['map'])
            if src_stmap_size:
                print("Source STMap [%s]: %s | Original size: %d×%d (spatial×time)" % 
                      (cfg['name'], src_stmap_path, src_stmap_size[0], src_stmap_size[1]))
        axes_src[i, 0].imshow(img, aspect='auto')
        # Use original dimensions if available, otherwise use transformed
        if src_stmap_size:
            title_stmap = '%s — STMap [%d×%d]' % (cfg['name'], src_stmap_size[0], src_stmap_size[1])
        else:
            spatial_h, spatial_w = img.shape[0], img.shape[1]
            title_stmap = '%s — STMap [%d×%d]' % (cfg['name'], spatial_h, spatial_w)
        if src_subject_id:
            title_stmap += ' (Subject: %s)' % src_subject_id
        axes_src[i, 0].set_title(title_stmap)
        axes_src[i, 0].set_xlabel('Time (frames)')
        axes_src[i, 0].set_ylabel('Spatial region')
        axes_src[i, 1].plot(b, color='C0')
        title_bvp = '%s — BVP (HR %.1f)' % (cfg['name'], g)
        if src_subject_id:
            title_bvp += ' — Subject: %s' % src_subject_id
        axes_src[i, 1].set_title(title_bvp)
        axes_src[i, 1].set_xlabel('Sample')
        axes_src[i, 1].set_ylabel('Amplitude')
    fig_src.tight_layout()
    fig_src.savefig(os.path.join(example_fig_dir, 'src_loaded.png'), dpi=120)
    plt.close(fig_src)

    print('Example figures saved to %s (tgt_loaded.png, src_loaded.png)' % example_fig_dir)
    return example_fig_dir
