#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# %%
"""
Subject difficulty analysis for PURE_my_in.

Analogous to the UBFC per-video analysis, but adapted to our STMap structure:
  - Features come from STMap images and GT BVP signals (no raw video needed)
  - HR errors come from the wave_sort fold results (wl0.0 run)
  - Spearman correlation heatmap shows which signal/image properties
    correlate with model error.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.eval_utils import hr_from_fft, FS_BVP

# %%
# ---- Paths ----
DATA_ROOT      = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME['PURE_my_in'][0])
WAVE_SORT_ROOT = os.path.join(config.WAVE_SORT_ROOT, 'cv_PURE_my_in')
WEIGHT_CL      = '0.0'   # which run to use for HR errors
STMAP_NAME     = config.STMAP_NAME
FRAMES_NUM     = 512
FS             = FS_BVP   # 30 Hz

print(f"Data root:      {DATA_ROOT}")
print(f"Wave_sort root: {WAVE_SORT_ROOT}")

# %%
# ---- Step 1: Collect per-subject HR error from wave_sort ----
# Each subject appears in exactly one fold's test set.
# Scan all wl{WEIGHT_CL} fold directories and collect mean HR error per subject.

def collect_hr_errors(wave_sort_root, weight_cl):
    """Return dict: subject_id -> {hr_gt, hr_pr, mae, me} averaged over segments."""
    errors = {}
    fold_dirs = sorted([
        d for d in os.listdir(wave_sort_root)
        if d.startswith(f'cv_PURE_my_in_wl{weight_cl}_fold')
    ])
    print(f"Found {len(fold_dirs)} fold directories for wl={weight_cl}")
    for fold_dir in fold_dirs:
        fold_path = os.path.join(wave_sort_root, fold_dir)
        gt_files = sorted([f for f in os.listdir(fold_path) if f.endswith('gt_Wave.mat')])
        for gt_file in gt_files:
            sid = gt_file.replace('gt_Wave.mat', '')
            pr_file = sid + 'pr_Wave.mat'
            if not os.path.isfile(os.path.join(fold_path, pr_file)):
                continue
            gt_mat = scio.loadmat(os.path.join(fold_path, gt_file))
            pr_mat = scio.loadmat(os.path.join(fold_path, pr_file))
            wave_gt = np.squeeze(gt_mat['Wave'])
            wave_pr = np.squeeze(pr_mat['Wave'])
            if wave_gt.ndim == 1:
                wave_gt = wave_gt[None, :]
                wave_pr = wave_pr[None, :]
            hr_gts, hr_prs = [], []
            for n in range(wave_gt.shape[0]):
                hg = hr_from_fft(wave_gt[n], fs=FS)
                hp = hr_from_fft(wave_pr[n], fs=FS)
                if np.isfinite(hg) and np.isfinite(hp):
                    hr_gts.append(hg)
                    hr_prs.append(hp)
            if hr_gts:
                errors[sid] = {
                    'hr_gt':  float(np.mean(hr_gts)),
                    'hr_pr':  float(np.mean(hr_prs)),
                    'mae':    float(np.mean(np.abs(np.array(hr_prs) - np.array(hr_gts)))),
                    'me':     float(np.mean(np.array(hr_prs) - np.array(hr_gts))),
                    'n_segs': len(hr_gts),
                    'fold':   fold_dir,
                }
    return errors

hr_errors = collect_hr_errors(WAVE_SORT_ROOT, WEIGHT_CL)
print(f"Collected HR errors for {len(hr_errors)} subjects")

# %%
# ---- Step 2: Extract features from STMap + GT BVP for each subject ----

def stmap_features(stmap_path):
    """Luminance, sharpness, and temporal/spatial variance from the STMap image.

    STMap layout: H = spatial (skin rows), W = time frames, C = RGB.
    - Temporal variance: how much pixel values change over time (motion proxy)
    - Spatial variance:  how much rows differ from each other (skin heterogeneity)
    - Sharpness:         Laplacian variance (blur proxy)
    """
    img = cv2.imread(stmap_path)
    if img is None:
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    lum_mean = gray.mean()
    lum_std  = gray.std()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    temporal_var = float(np.var(gray, axis=1).mean())  # variance along time per row
    spatial_var  = float(np.var(gray, axis=0).mean())  # variance across rows per frame
    return {
        'lum_mean':     lum_mean,
        'lum_std':      lum_std,
        'sharpness':    sharpness,
        'temporal_var': temporal_var,
        'spatial_var':  spatial_var,
    }


def bvp_features(bvp_path, frames_num=FRAMES_NUM, fs=FS):
    """SNR and HR variability of the GT BVP signal (averaged over all windows)."""
    try:
        bvp_full = scio.loadmat(bvp_path)['BVP'].reshape(-1).astype(np.float64)
    except Exception:
        return {}
    n = len(bvp_full)
    step = 10
    snrs, hr_vals = [], []
    i = 0
    while i + frames_num <= n:
        seg = bvp_full[i:i + frames_num]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        freqs = np.fft.rfftfreq(frames_num, 1.0 / fs)
        power = np.abs(np.fft.rfft(seg)) ** 2
        sig   = (freqs >= 0.7) & (freqs <= 3.0)
        noise = ~sig & (freqs > 0)
        if sig.any() and noise.any():
            snr = 10 * np.log10(power[sig].sum() / (power[noise].sum() + 1e-8))
            snrs.append(float(snr))
        hr = hr_from_fft(seg, fs=fs)
        if np.isfinite(hr):
            hr_vals.append(hr)
        i += frames_num  # non-overlapping windows (~3 per PURE recording)
    return {
        'bvp_snr':        float(np.mean(snrs))    if snrs    else np.nan,
        'gt_hr_mean':     float(np.mean(hr_vals)) if hr_vals else np.nan,
        'gt_hr_std':      float(np.std(hr_vals))  if len(hr_vals) > 1 else np.nan,
    }


subjects = sorted([s for s in os.listdir(DATA_ROOT) if not s.startswith('.')])
records = []

for sid in tqdm(subjects, desc='Extracting features'):
    if sid not in hr_errors:
        continue
    subj_path = os.path.join(DATA_ROOT, sid)
    stmap_path = os.path.join(subj_path, 'STMap', STMAP_NAME)
    bvp_path   = os.path.join(subj_path, 'Label', 'BVP.mat')

    row = {'subject_id': sid}
    row.update(hr_errors[sid])
    row.update(stmap_features(stmap_path))
    row.update(bvp_features(bvp_path))
    row['participant'] = sid.split('-')[0]   # e.g. '01' from '01-03'
    row['scenario']    = sid.split('-')[1]   # e.g. '03'
    records.append(row)

df = pd.DataFrame(records)
out_csv = os.path.join(config.RESULT_LOG_DIR, 'pure_difficulty_features.csv')
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}  ({len(df)} subjects)")
print(df[['subject_id', 'mae', 'bvp_snr', 'lum_mean', 'sharpness', 'temporal_var']].to_string())

# %%
# ---- Step 3: Spearman correlation heatmap ----

feature_cols = ['lum_mean', 'lum_std', 'sharpness', 'temporal_var', 'spatial_var',
                'bvp_snr', 'gt_hr_mean', 'gt_hr_std']
target_cols  = ['mae', 'me']

numeric_df = df[feature_cols + target_cols].dropna()
corr = numeric_df.corr(method='spearman')
corr_target = corr[target_cols].loc[feature_cols].sort_values('mae', ascending=False)

fig, ax = plt.subplots(figsize=(5, 7))
sns.heatmap(corr_target, annot=True, cmap='coolwarm', center=0,
            linewidths=0.5, cbar=True, fmt='.2f', ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Spearman Correlation with HR Error\n(PURE_my_in, intratest wl=0.0)')
ax.set_xlabel('')
ax.set_ylabel('Feature')
plt.tight_layout()
plot_path = os.path.join(config.RESULT_LOG_DIR, 'pure_difficulty_corr.png')
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"Saved: {plot_path}")

# %%
# ---- Step 4: Per-participant breakdown ----
# Since PURE has only 10 participants, show MAE per participant
# to see if difficulty is participant-specific or scenario-specific.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

part_df = df.groupby('participant')['mae'].agg(['mean', 'std']).reset_index()
part_df = part_df.sort_values('mean', ascending=False)
axes[0].bar(part_df['participant'], part_df['mean'], yerr=part_df['std'],
            capsize=4, color='steelblue', alpha=0.8)
axes[0].set_xlabel('Participant ID')
axes[0].set_ylabel('MAE (BPM)')
axes[0].set_title('Mean MAE per Participant\n(error bars = std across scenarios)')
axes[0].grid(True, axis='y', alpha=0.3)

scen_df = df.groupby('scenario')['mae'].agg(['mean', 'std']).reset_index()
scen_df = scen_df.sort_values('mean', ascending=False)
axes[1].bar(scen_df['scenario'], scen_df['mean'], yerr=scen_df['std'],
            capsize=4, color='coral', alpha=0.8)
axes[1].set_xlabel('Scenario')
axes[1].set_ylabel('MAE (BPM)')
axes[1].set_title('Mean MAE per Scenario\n(error bars = std across participants)')
axes[1].grid(True, axis='y', alpha=0.3)

plt.suptitle('PURE_my_in — Is difficulty participant- or scenario-driven?', y=1.02)
plt.tight_layout()
breakdown_path = os.path.join(config.RESULT_LOG_DIR, 'pure_difficulty_breakdown.png')
fig.savefig(breakdown_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {breakdown_path}")

# %%
# ---- Step 5: PCA and t-SNE clustering ----
# Project subjects into 2-D using PCA and t-SNE on the feature matrix.
# Color by MAE to see whether hard subjects cluster together.

cluster_df = df[feature_cols + ['mae', 'subject_id']].dropna().reset_index(drop=True)
X = cluster_df[feature_cols].values
mae_vals = cluster_df['mae'].values
labels   = cluster_df['subject_id'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=mae_vals, cmap='RdYlGn_r',
                s=80, edgecolors='k', linewidths=0.4)
plt.colorbar(sc, ax=ax, label='MAE (BPM)')
for i, lbl in enumerate(labels):
    ax.annotate(lbl, (X_pca[i, 0], X_pca[i, 1]),
                fontsize=6, ha='center', va='bottom',
                xytext=(0, 4), textcoords='offset points')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax.set_title('PCA of Subject Features — colored by MAE\n(PURE_my_in, intratest wl=0.0)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
pca_path = os.path.join(config.RESULT_LOG_DIR, 'pure_difficulty_pca.png')
fig.savefig(pca_path, dpi=150)
plt.show()
print(f"Saved: {pca_path}")

# t-SNE (perplexity capped at n_samples - 1 for small datasets)
perplexity = min(5, len(cluster_df) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=mae_vals, cmap='RdYlGn_r',
                s=80, edgecolors='k', linewidths=0.4)
plt.colorbar(sc, ax=ax, label='MAE (BPM)')
for i, lbl in enumerate(labels):
    ax.annotate(lbl, (X_tsne[i, 0], X_tsne[i, 1]),
                fontsize=6, ha='center', va='bottom',
                xytext=(0, 4), textcoords='offset points')
ax.set_xlabel('t-SNE dim 1')
ax.set_ylabel('t-SNE dim 2')
ax.set_title(f't-SNE of Subject Features — colored by MAE\n(PURE_my_in, intratest wl=0.0, perplexity={perplexity})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
tsne_path = os.path.join(config.RESULT_LOG_DIR, 'pure_difficulty_tsne.png')
fig.savefig(tsne_path, dpi=150)
plt.show()
print(f"Saved: {tsne_path}")
