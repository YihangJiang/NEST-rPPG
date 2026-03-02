# -*- coding: UTF-8 -*-
"""
Visualize feature space (av) with and without contrastive loss (NEST_DM).
Run in Jupyter: execute cells in order. Edit Cell 1 (Config) then run all.

Compares two trained models on the same data:
  - With contrastive loss: orig vs aug pairs should be closer in feature space.
  - Without: pairs can be farther apart.
"""
# %%
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from types import SimpleNamespace

import MyDataset
import model

# %%
# ============ Cell 1: Config – edit and run ============
_USE_JUPYTER_CONFIG = True

if _USE_JUPYTER_CONFIG:
    # Jupyter: set these in the notebook
    index_dir = './STMap/STMap_Index/PURE'
    data_name = 'PURE'
    stmap_name = 'STMap'
    frames_num = 256
    batch_size = 50
    max_samples = 500
    out_dir = './feature_vis'
    device_name = 'cuda:0'
    seed = 42
    reducer = 'tsne'  # 'tsne' or 'pca'

    # Compare four models (different loss types), e.g. One / TA / CM / DM
    model_one = '/home/yj167/Desktop/NEST-rPPG/NEST-rPPG/model/rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1_lossTA'   # e.g. './model/rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1_lossOne'
    model_ta  = '/home/yj167/Desktop/NEST-rPPG/NEST-rPPG/model/rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1_lossTA' # e.g. './model/rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1_lossTA'
    model_cm  = '/home/yj167/Desktop/NEST-rPPG/NEST-rPPG/model/rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1_lossCM' # e.g. './model/rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1_lossCM'
    model_dm  = '/home/yj167/Desktop/NEST-rPPG/NEST-rPPG/model/rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1_lossDM' # e.g. './model/rPPGNet_PURE_srcUBFCSpatial0.5Temporal0.1_lossDM'
else:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--index-dir', default='./STMap/STMap_Index/PURE')
    p.add_argument('--data-name', default='PURE')
    p.add_argument('--stmap-name', default='STMap')
    p.add_argument('--frames-num', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=50)
    p.add_argument('--max-samples', type=int, default=500)
    p.add_argument('--out-dir', default='./feature_vis')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--reducer', default='tsne', choices=['tsne', 'pca'])
    p.add_argument('--model-one', default=None)
    p.add_argument('--model-ta', default=None)
    p.add_argument('--model-cm', default=None)
    p.add_argument('--model-dm', default=None)
    args = p.parse_args()
    index_dir = args.index_dir
    data_name = args.data_name
    stmap_name = args.stmap_name
    frames_num = args.frames_num
    batch_size = args.batch_size
    max_samples = args.max_samples
    out_dir = args.out_dir
    device_name = getattr(args, 'device', 'cuda:0')
    seed = args.seed
    reducer = args.reducer
    model_one = getattr(args, 'model_one', None)
    model_ta = getattr(args, 'model_ta', None)
    model_cm = getattr(args, 'model_cm', None)
    model_dm = getattr(args, 'model_dm', None)

# %%
def make_dummy_args():
    return SimpleNamespace(
        spatial_aug_rate=0.5,
        temporal_aug_rate=0.1,
        frames_num=frames_num,
    )


def collect_features(net, loader, device, max_samples):
    """Collect av and av_aug for each batch. Returns (av_all, av_aug_all) each (N, D)."""
    net.eval()
    av_list, av_aug_list = [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            map_ori, _, _, map_aug, _, _ = batch
            map_ori = map_ori.float().to(device)
            map_aug = map_aug.float().to(device)
            _, _, av = net(map_ori)
            _, _, av_aug = net(map_aug)
            av_list.append(av.cpu().numpy())
            av_aug_list.append(av_aug.cpu().numpy())
            n += av.shape[0]
            if n >= max_samples:
                break
    av_all = np.concatenate(av_list, axis=0)[:max_samples]
    av_aug_all = np.concatenate(av_aug_list, axis=0)[:max_samples]
    return av_all, av_aug_all


def reduce_2d(X, method='tsne', perplexity=30):
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X)
    from sklearn.manifold import TSNE
    # Some sklearn versions don't expose n_iter in __init__, so rely on defaults.
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(2, X.shape[0] - 1)),
        random_state=42,
    )
    return tsne.fit_transform(X)


def plot_pairs(ax, orig_2d, aug_2d, title, show_lines=True, alpha=0.6):
    ax.scatter(orig_2d[:, 0], orig_2d[:, 1], c='C0', label='Original', alpha=alpha, s=15)
    ax.scatter(aug_2d[:, 0], aug_2d[:, 1], c='C1', label='Augmented', alpha=alpha, s=15)
    if show_lines:
        for i in range(min(orig_2d.shape[0], 80)):
            ax.plot([orig_2d[i, 0], aug_2d[i, 0]], [orig_2d[i, 1], aug_2d[i, 1]],
                    'gray', alpha=0.25, linewidth=0.5)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal')

# %%
# Dataset and DataLoader
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

dummy_args = make_dummy_args()
dataset = MyDataset.Data_DG(
    root_dir=index_dir,
    dataName=data_name,
    STMap=stmap_name + '.png',
    frames_num=frames_num,
    args=dummy_args,
)
# Important: use shuffle=False so repeated passes over the loader see the same samples,
# and comparisons between models are on exactly the same data.
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print('Dataset size:', len(dataset), '| max_samples:', max_samples)

# %%
def load_and_collect(model_path, max_n):
    net = torch.load(model_path, map_location=device)
    net = net.to(device)
    return collect_features(net, loader, device, max_n)


def load_and_collect_four(model_paths, max_n):
    """
    Collect features for up to four models on the SAME batches (no shuffling).
    model_paths: list/tuple of up to 4 model paths [one, ta, cm, dm].
    Returns: list of (av_all, av_aug_all) per model, all aligned on the same samples.
    """
    nets = [torch.load(p, map_location=device).to(device) for p in model_paths]
    for net in nets:
        net.eval()
    av_lists = [[] for _ in nets]
    av_aug_lists = [[] for _ in nets]
    n = 0
    with torch.no_grad():
        for batch in loader:
            map_ori, _, _, map_aug, _, _ = batch
            map_ori = map_ori.float().to(device)
            map_aug = map_aug.float().to(device)

            for i, net in enumerate(nets):
                _, _, av_i = net(map_ori)
                _, _, av_aug_i = net(map_aug)
                av_lists[i].append(av_i.cpu().numpy())
                av_aug_lists[i].append(av_aug_i.cpu().numpy())

            n += map_ori.shape[0]
            if n >= max_n:
                break
    results = []
    for av_list, av_aug_list in zip(av_lists, av_aug_lists):
        av_all = np.concatenate(av_list, axis=0)[:max_n]
        av_aug_all = np.concatenate(av_aug_list, axis=0)[:max_n]
        results.append((av_all, av_aug_all))
    return results

# Compare four models (e.g., loss types One / TA / CM / DM)
if all(m is not None for m in (model_one, model_ta, model_cm, model_dm)):
    results = load_and_collect_four(
        [model_one, model_ta, model_cm, model_dm],
        max_samples,
    )
    (av_one, av_aug_one), (av_ta, av_aug_ta), (av_cm, av_aug_cm), (av_dm, av_aug_dm) = results

    n = av_one.shape[0]
    orig_one = np.concatenate([av_one, av_aug_one], axis=0)
    orig_ta = np.concatenate([av_ta, av_aug_ta], axis=0)
    orig_cm = np.concatenate([av_cm, av_aug_cm], axis=0)
    orig_dm = np.concatenate([av_dm, av_aug_dm], axis=0)

    # Shared 2D embedding across all four models
    all_feats = np.concatenate([orig_one, orig_ta, orig_cm, orig_dm], axis=0)
    all_2d = reduce_2d(all_feats, method=reducer)

    two_n = 2 * n
    one_2d_all = all_2d[0:two_n]
    ta_2d_all = all_2d[two_n:2 * two_n]
    cm_2d_all = all_2d[2 * two_n:3 * two_n]
    dm_2d_all = all_2d[3 * two_n:4 * two_n]

    orig_one_2d, aug_one_2d = one_2d_all[:n], one_2d_all[n:]
    orig_ta_2d, aug_ta_2d = ta_2d_all[:n], ta_2d_all[n:]
    orig_cm_2d, aug_cm_2d = cm_2d_all[:n], cm_2d_all[n:]
    orig_dm_2d, aug_dm_2d = dm_2d_all[:n], dm_2d_all[n:]

    dist_one = np.linalg.norm(orig_one_2d - aug_one_2d, axis=1)
    dist_ta = np.linalg.norm(orig_ta_2d - aug_ta_2d, axis=1)
    dist_cm = np.linalg.norm(orig_cm_2d - aug_cm_2d, axis=1)
    dist_dm = np.linalg.norm(orig_dm_2d - aug_dm_2d, axis=1)
    print(
        "Mean pair distance (orig–aug): "
        "One = %.4f, TA = %.4f, CM = %.4f, DM = %.4f"
        % (dist_one.mean(), dist_ta.mean(), dist_cm.mean(), dist_dm.mean())
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_pairs(axes[0, 0], orig_one_2d, aug_one_2d, 'Loss: One', show_lines=True)
    plot_pairs(axes[0, 1], orig_ta_2d, aug_ta_2d, 'Loss: TA', show_lines=True)
    plot_pairs(axes[1, 0], orig_cm_2d, aug_cm_2d, 'Loss: CM', show_lines=True)
    plot_pairs(axes[1, 1], orig_dm_2d, aug_dm_2d, 'Loss: DM', show_lines=True)
    fig.suptitle('Feature space: original vs augmented pairs (same sample)', y=1.02)
    out_path = os.path.join(out_dir, 'features_four_losses.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved', out_path)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    ax2.hist(dist_one, bins=30, alpha=0.5, label='One', color='C0')
    ax2.hist(dist_ta, bins=30, alpha=0.5, label='TA', color='C1')
    ax2.hist(dist_cm, bins=30, alpha=0.5, label='CM', color='C2')
    ax2.hist(dist_dm, bins=30, alpha=0.5, label='DM', color='C3')
    ax2.set_xlabel('Pair distance (orig–aug) in 2D')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.set_title('Distribution of same-sample pair distances by loss type')
    out_path2 = os.path.join(out_dir, 'pair_distance_hist_four_losses.png')
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved', out_path2)

# %%
# If no model path was set, prompt in notebook
if not all(m is not None for m in (model_one, model_ta, model_cm, model_dm)):
    print('Edit Cell 1 (Config): set all of (model_one, model_ta, model_cm, model_dm), then re-run.')

# %%
