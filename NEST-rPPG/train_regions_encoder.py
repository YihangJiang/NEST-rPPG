#!/usr/bin/env python
# %%
# -*- coding: UTF-8 -*-
# Region-aware (ROI) training: train on multiple region domains (e.g. cheek, target, eye),
# test on a different dataset/region. Source regions and test domain come from config.
# Run as script: python train_regions.py
# Run in Jupyter: execute cells in order (Cell 1 = config, Cell 2 = rest or run all)
# %reload_ext autoreload
# %autoreload 2

import os
import random
from types import SimpleNamespace

from datetime import datetime
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import scipy.io as io

from torch.utils.data import DataLoader
from torch.autograd import Variable

import MyDataset
import MyLoss
import model
from utils.core import Logger, time_to_str, get_args
import utils.train_utils as train_utils

import config

# %%
# ============ Cell 1: Config (constants) ============
# True = use constants below (Jupyter). False = use command-line args (python train_my.py ...).
_USE_JUPYTER_CONFIG = False

if _USE_JUPYTER_CONFIG:
    args = SimpleNamespace(
        GPU='0',
        num_workers=2,
        epochs=50,
        batchsize=100,
        lr=0.001,
        max_iter=1000,          # total training iterations (like train.py)
        seed=config.SEED,
        k1=1.0, k2=0.1, k3=1.0, k4=0.1, k5=1.0, k6=0.1, k7=0.1, k8=0.1,
        temporal_aug_rate=config.TEMPORAL_AUG_RATE,
        spatial_aug_rate=config.SPATIAL_AUG_RATE,
        loss_type=config.LOSS_TYPE,  # One / TA / CM / DM / All
        frames_num=512,
        tgt=config.TGT_DOMAIN,
        # Baseline: single source ROI domain (default: config.TARGET_DOMAIN[config.TGT_DOMAIN][1], i.e., *_in)
        src=config.SRC_DOMAIN,
        stmap_name=config.STMAP_NAME,
        index_root=config.STMAP_INDEX_BASE,  # index root (subfolders are domain names)
        # Save per-subject feature representations (av) during training
        save_features=True,
        # Weight and temperature for InfoNCE alignment between src and pos/neg domains
        weight_info=0.01,
        tau_info=0.07,
        # When False: disable InfoNCE (no pos/neg forwards, align_pos_loss=0).
        # When True: use pos/neg domains for contrastive alignment.
        use_infonce=False,
        grad_clip=5.0,
    )
else:
    args = get_args()
    if not hasattr(args, 'frames_num'):
        args.frames_num = 512
    if not hasattr(args, 'tgt'):
        args.tgt = config.TGT_DOMAIN
    if not hasattr(args, 'src') or args.src is None:
        raise ValueError("Please provide --src <base_source_domain> (e.g. --src PURE_my)")
    if not hasattr(args, 'stmap_name'):
        args.stmap_name = config.STMAP_NAME
    if not hasattr(args, 'index_root'):
        args.index_root = config.STMAP_INDEX_BASE
    if not hasattr(args, 'max_iter'):
        args.max_iter = 1000
    if not hasattr(args, 'loss_type'):
        args.loss_type = config.LOSS_TYPE
    if not hasattr(args, 'save_features'):
        args.save_features = False
    if not hasattr(args, 'tau_info'):
        args.tau_info = 0.07
    if not hasattr(args, 'seed'):
        args.seed = 0
    if not hasattr(args, 'grad_clip'):
        args.grad_clip = 5.0
    if not hasattr(args, 'weight_info'):
        # Used later as: loss = loss + args.weight_info * align_pos_loss
        args.weight_info = 0.01

# ============ End Cell 1 ============

# %%
# ============ Cell 2: Dataset & index ============
print("=" * 60)
print(args.use_infonce)

# Reproducibility: set all RNG seeds before any randomness (datasets, model, dataloader shuffle)
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _worker_init_fn(worker_id):
    """Deterministic DataLoader worker: seed per worker so augmentation order is reproducible."""
    base_seed = int(getattr(args, 'seed', 0))
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)

_set_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("  Random seed:", args.seed, "(deterministic training)")
print("  weight_info:", getattr(args, "weight_info", 0.0))

# Test/target domain can be passed from CLI via -t/--tgt.
tgt_domain = args.tgt
if tgt_domain not in config.FILEA_NAME:
    raise ValueError(f"Target domain has no FILEA_NAME entry: {tgt_domain}")

# Base source (prefix): must come from --src in utils/core.py.
# You may pass --src as either a base like "PURE_my" or a full region-domain like "PURE_my_in".
# If a suffix is included (last underscore token), we strip it so we can derive:
#   pos  = <base>_rm
#   neg  = <base>_eye
#   src  = <base>_<tgt_suffix>
if args.src is None:
    raise ValueError("Please provide --src <base_source_domain> (e.g. --src PURE_my or --src PURE_my_in)")
if "_" not in args.src:
    raise ValueError(f"--src must contain at least one '_' suffix token, got {args.src!r}")
base_src = args.src.rsplit("_", 1)[0]

# Region suffixes (requested behavior):
# - pos domain uses rm
# - neg domain uses eye
# - source domain uses args.src directly
pos_domain = f"{base_src}_rm"
neg_domain = f"{base_src}_eye"
source_domain = args.src

region_domains = [pos_domain, source_domain, neg_domain]
for d in region_domains:
    if d not in config.FILEA_NAME:
        raise ValueError(
            f"Derived region domain {d!r} has no FILEA_NAME entry. "
            f"Check that base_src={base_src!r} and tgt_domain={tgt_domain!r} are compatible."
        )

# Roots for each region domain and target domain
region_roots = {
    d: os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[d][0])
    for d in region_domains
}
index_root = args.index_root

# Index dirs for each region and for the target domain
region_index_dirs = {d: os.path.join(index_root, d) for d in region_domains}
source_index_dir = region_index_dirs[source_domain]
target_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[tgt_domain][0])
target_index_dir = os.path.join(index_root, tgt_domain)

frames_num = args.frames_num
batch_size = args.batchsize
num_workers = args.num_workers
GPU = args.GPU

print("Training domains (suffix rule, independent of config.TARGET_DOMAIN):")
print("  base_src      :", base_src)
print("  tgt_domain    :", tgt_domain)
print("  source_domain :", source_domain)
print("  use_infonce   :", args.use_infonce)
print("  pos_domain    :", pos_domain)
print("  neg_domain    :", neg_domain)
print("Test STMap root:", target_root)
print("Index root:", index_root)

os.makedirs(index_root, exist_ok=True)
def _build_index_always(root_dir, index_dir, stmap_name, label):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"{label} root not found: {root_dir}")
    os.makedirs(index_dir, exist_ok=True)

    # Match train.py behavior: always clear and rebuild index files.
    removed = 0
    for fname in os.listdir(index_dir):
        fpath = os.path.join(index_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            removed += 1
    if removed:
        print(f"  Cleared {removed} index files in {index_dir}")

    print(f"Building index for {label}:")
    files_list = sorted([f for f in os.listdir(root_dir) if not f.startswith('.')])
    MyDataset.getIndex(root_dir, files_list, index_dir, stmap_name, 10, frames_num)

# Build indexes for all three region domains (cheek, target, eye)
for d in region_domains:
    _build_index_always(region_roots[d], region_index_dirs[d], args.stmap_name, d)

# And for the test domain (target of evaluation)
_build_index_always(target_root, target_index_dir, args.stmap_name, tgt_domain)

print("Loading datasets...")

# Region datasets (cheek/target/eye), kept for future region-specific training logic
region_db_list = []
for d in region_domains:
    db = MyDataset.Data_DG(
        root_dir=region_index_dirs[d],
        dataName=config.canonical_data_name(d),
        STMap=args.stmap_name,
        frames_num=frames_num,
        args=args
    )
    region_db_list.append(db)
    print(f"  region dataset {d}: num_samples = {len(db)}")

# Explicit references for positive/negative region datasets
pos_db = MyDataset.Data_DG(
    root_dir=region_index_dirs[pos_domain],
    dataName=config.canonical_data_name(pos_domain),
    STMap=args.stmap_name,
    frames_num=frames_num,
    args=args
)
neg_db = MyDataset.Data_DG(
    root_dir=region_index_dirs[neg_domain],
    dataName=config.canonical_data_name(neg_domain),
    STMap=args.stmap_name,
    frames_num=frames_num,
    args=args
)
print("  pos_domain dataset:", pos_domain, "num_samples =", len(pos_db))
print("  neg_domain dataset:", neg_domain, "num_samples =", len(neg_db))

# Baseline: single source + single target (test) domains, used in training loop
source_db = MyDataset.Data_DG(
    root_dir=source_index_dir,
    dataName=config.canonical_data_name(source_domain),
    STMap=args.stmap_name,
    frames_num=frames_num,
    args=args
)
target_db = MyDataset.Data_DG(
    root_dir=target_index_dir,
    dataName=config.canonical_data_name(tgt_domain),
    STMap=args.stmap_name,
    frames_num=frames_num,
    args=args
)

print("  baseline source dataset:", source_domain, "num_samples =", len(source_db))
print("  baseline test dataset  :", tgt_domain, "num_samples =", len(target_db))

print("Creating DataLoaders (baseline like train.py)...")
# Fixed generator so shuffle order is reproducible across runs
_generator = torch.Generator().manual_seed(args.seed)
src_loader = DataLoader(
    source_db, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    worker_init_fn=_worker_init_fn, generator=_generator
)
tgt_loader = DataLoader(
    target_db, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    worker_init_fn=_worker_init_fn
)

# Dataloaders for pos/neg domains
pos_loader = DataLoader(
    pos_db, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    worker_init_fn=_worker_init_fn
)
neg_loader = DataLoader(
    neg_db, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    worker_init_fn=_worker_init_fn
)

steps_per_epoch = len(src_loader)
print(f"steps_per_epoch (source) = {steps_per_epoch}")
print("=" * 60)

# %%
# ============ Cell 3: Model, losses, optimizer ============
if torch.cuda.is_available():
    device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
else:
    device = torch.device('cpu')
    print('Using CPU')

# Encoder/decoder experiment: skip-connected decoder (train_regions.py keeps model.BaseNet).
BaseNet = model.BaseNetSkip().to(device=device)

optimizer = torch.optim.Adam(BaseNet.parameters(), lr=args.lr)
loss_func_NP = MyLoss.P_loss3().to(device)
loss_func_L1 = nn.L1Loss().to(device)
loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)
loss_func_NEST_CM = MyLoss.NEST_CM().to(device)
loss_func_NEST_DM = MyLoss.NEST_DM().to(device)
loss_func_NEST_TA = MyLoss.NEST_TA(device, Num_ref=8).to(device)

# Logging & model name (use config paths)
os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)

# Naming: rPPGNet_<tgt_domain>_src<target_region> (e.g. rPPGNet_UBFC_my_in_srcPURE_my_in)
Target_name = tgt_domain
rPPGNet_name = config.build_run_name(
    tgt=Target_name,
    src=source_domain,
    spatial=getattr(args, 'spatial_aug_rate', config.SPATIAL_AUG_RATE),
    temporal=getattr(args, 'temporal_aug_rate', config.TEMPORAL_AUG_RATE),
    ui=getattr(args, 'use_infonce', False)
)

log = Logger()
log_path = os.path.join(config.RESULT_LOG_DIR, rPPGNet_name + '_log.txt')
log.open(log_path, mode='a')
log.write("\n----------------------------------------------- [START %s] %s\n\n" %
          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

# Print a train.py-style training summary for this baseline region run
Source_domain_Names = [source_domain]
total_src_samples = len(source_db)

log.write("TRAINING CONFIG (regions baseline)\n")
log.write("  Target domain:     %s\n" % Target_name)
log.write("  Target index:      %s\n" % target_index_dir)
log.write("  Target STMap:      %s\n" % args.stmap_name)
log.write("  Source domains:    %s\n" % Source_domain_Names)
for i, d in enumerate(Source_domain_Names):
    log.write("    [%d] %s -> %s\n" % (i, d, source_index_dir))
log.write("  Batch size:        %s\n" % batch_size)
log.write("  Max iterations:    %s\n" % args.max_iter)
log.write("  Frames per clip:   %s\n" % frames_num)
log.write("  Loss type:         %s\n" % getattr(args, 'loss_type', config.LOSS_TYPE))
log.write("  Device:            %s\n" % device)
log.write("  Model:             %s\n" % BaseNet.__class__.__name__)
log.write("  Source samples:    %s  (target: %d)\n" % (total_src_samples, len(target_db)))
log.write("  Log file:          %s\n" % log_path)
log.write("=" * 60 + "\n\n")

# %%
# ============ Cell 4: Training loop (train.py-style iter/max_iter) ============
BaseNet.train()
start = timer()
max_iter = args.max_iter

src_iter = iter(src_loader)
src_iter_per_epoch = len(src_iter)
pos_iter = iter(pos_loader)
neg_iter = iter(neg_loader)

_printed_nan_debug = False

for iter_num in range(max_iter + 1):
    # Reset iterators at epoch boundaries (same as train.py)
    if iter_num > 0 and (iter_num % src_iter_per_epoch == 0):
        src_iter = src_loader.__iter__()
        pos_iter = pos_loader.__iter__()
        neg_iter = neg_loader.__iter__()

    # Source batch (includes augmented view) + subject paths (from Data_DG with return_path=True)
    data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug, subj_paths = src_iter.__next__()
    data = Variable(data).float().to(device=device)
    bvp = Variable(bvp).float().to(device=device).unsqueeze(dim=1)
    HR_rel = Variable(torch.Tensor(HR_rel)).float().to(device=device)
    data_aug = Variable(data_aug).float().to(device=device)
    bvp_aug = Variable(bvp_aug).float().to(device=device).unsqueeze(dim=1)
    HR_rel_aug = Variable(torch.Tensor(HR_rel_aug)).float().to(device=device)
    # Optional InfoNCE contrastive alignment (pos/neg domains).
    # When disabled, we don't run pos/neg forwards and align_pos_loss stays 0.
    use_infonce = bool(getattr(args, "use_infonce", False))
    align_pos_loss = torch.tensor(0.0, device=device)
    av_pos = None
    av_neg = None

    if use_infonce:
        # Positive/negative-domain batches for InfoNCE.
        # Important: run these auxiliary forwards WITHOUT gradients.
        try:
            pos_data, _, _, _, _, _, _ = pos_iter.__next__()
        except StopIteration:
            pos_iter = pos_loader.__iter__()
            pos_data, _, _, _, _, _, _ = pos_iter.__next__()
        try:
            neg_data, _, _, _, _, _, _ = neg_iter.__next__()
        except StopIteration:
            neg_iter = neg_loader.__iter__()
            neg_data, _, _, _, _, _, _ = neg_iter.__next__()

        pos_data = Variable(pos_data).float().to(device=device)
        neg_data = Variable(neg_data).float().to(device=device)

        BaseNet.eval()
        with torch.no_grad():
            _, _, av_pos = BaseNet(pos_data)
            _, _, av_neg = BaseNet(neg_data)
        BaseNet.train()


    optimizer.zero_grad()
    bvp_pre, HR_pr, av = BaseNet(data)
    bvp_pre_aug, HR_pr_aug, av_aug = BaseNet(data_aug)

    if use_infonce:
        # InfoNCE-style alignment: pull src av toward paired pos av; repel neg keys only (not other k_pos[j]).
        m = av.shape[0]
        tau = float(getattr(args, 'tau_info', 0.07))
        q = F.normalize(av, dim=1)
        k_pos = F.normalize(av_pos[:m], dim=1)
        k_neg = F.normalize(av_neg[:m], dim=1)
        # penalize 
        pos_scores = (q * k_pos).sum(dim=1, keepdim=True) / tau  # (m, 1)
        neg_scores = torch.mm(q, k_neg.t()) / tau  # (m, m)
        logits = torch.cat([pos_scores, neg_scores], dim=1)  # (m, m+1)
        labels = torch.zeros(m, dtype=torch.long, device=logits.device)
        align_pos_loss = F.cross_entropy(logits, labels)
        # Previous variant (batch keys): softmax over all k_pos and k_neg — also penalizes high q·k_pos[j], j!=i.
        # keys = torch.cat([k_pos, k_neg], dim=0)          # (2m, d)
        # logits = torch.mm(q, keys.t()) / tau            # (m, 2m)
        # labels = torch.arange(m, device=logits.device)  # positives at indices 0..m-1
        # align_pos_loss = F.cross_entropy(logits, labels)

    # One-time NaN diagnostics (helps locate first NaN source)
    if not _printed_nan_debug:
        any_nan = (
            torch.isnan(data).any() or torch.isnan(bvp_pre).any() or torch.isnan(av).any() or
            torch.isnan(data_aug).any() or torch.isnan(bvp_pre_aug).any() or torch.isnan(av_aug).any()
        )
        if any_nan:
            print("\n[NaN DEBUG] First NaN detected at iter_num =", iter_num)
            print("  data nan:", bool(torch.isnan(data).any()))
            print("  bvp_pre nan:", bool(torch.isnan(bvp_pre).any()))
            print("  av nan:", bool(torch.isnan(av).any()))
            print("  data_aug nan:", bool(torch.isnan(data_aug).any()))
            print("  bvp_pre_aug nan:", bool(torch.isnan(bvp_pre_aug).any()))
            print("  av_aug nan:", bool(torch.isnan(av_aug).any()))
            _printed_nan_debug = True



    # Optionally save per-subject feature representations (av) for this batch
    if getattr(args, 'save_features', False):
        av_cpu = av.detach().cpu().numpy()
        for path_str, feat in zip(subj_paths, av_cpu):
            # subj_paths elements are subject root paths like STMap_my/PURE_my_rm/01-01
            if not isinstance(path_str, str):
                path_str = str(path_str)
            feat_name = f"feat_iter_{iter_num:06d}.npy"
            feat_path = os.path.join(path_str, feat_name)
            try:
                os.makedirs(path_str, exist_ok=True)
                np.save(feat_path, feat)
            except Exception as e:
                print(f"Warning: failed to save feature to {feat_path}: {repr(e)}")

    src_loss = MyLoss.get_loss(
        bvp_pre, HR_pr, bvp, HR_rel, source_domain,
        loss_func_NP, loss_func_L1, args, iter_num
    )
    # src_loss_aug = MyLoss.get_loss(
    #     bvp_pre_aug, HR_pr_aug, bvp_aug, HR_rel_aug, source_domain,
    #     loss_func_NP, loss_func_L1, args, iter_num
    # )

    loss_CM = -loss_func_NEST_CM(torch.cat((av, av_aug), dim=0))
    loss_DM = loss_func_NEST_DM(av, av_aug)
    loss_TA = loss_func_NEST_TA(
        torch.cat((av, av_aug), dim=0),
        torch.cat((HR_rel, HR_rel_aug), dim=0)
    )

    # (InfoNCE alignment handled above; align_pos_loss is 0 if disabled.)

    loss_type = getattr(args, 'loss_type', config.LOSS_TYPE)
    if loss_type == 'One':
        loss = src_loss
    elif loss_type == 'TA':
        loss = src_loss + loss_TA
    elif loss_type == 'CM':
        loss = src_loss + loss_CM
    elif loss_type == 'DM':
        loss = src_loss + loss_DM
    elif loss_type == 'All':
        loss = src_loss + loss_TA + loss_CM + loss_DM
    else:
        loss = src_loss + loss_TA

    # Add alignment regularizer between src av and pos_domain av
    loss = loss + args.weight_info * align_pos_loss

    if torch.sum(torch.isnan(loss)) > 0:
        print('Nan')
        break
    loss.backward()
    # Prevent exploding gradients -> NaNs in decoder output (bvp_pre)
    torch.nn.utils.clip_grad_norm_(BaseNet.parameters(), max_norm=float(args.grad_clip))
    optimizer.step()

    if iter_num % 100 == 0:
        log_line = (
            'Train Inter:' + str(iter_num) +
            ' | loss:  ' + str(loss.data.cpu().numpy()) +
            ' |' + source_domain + ' : ' + str(src_loss.data.cpu().numpy()) +
            ' |' + 'CM' + ' : ' + str(loss_CM.data.cpu().numpy()) +
            ' |' + 'DM' + ' : ' + str(loss_DM.data.cpu().numpy()) +
            ' |' + 'TA' + ' : ' + str(loss_TA.data.cpu().numpy()) +
            ' |' + 'AlignPos' + ' : ' + str(align_pos_loss.data.cpu().numpy()) +
            ' |' + time_to_str(timer() - start, 'min')
        )
        log.write(log_line)
        log.write('\n')

print("Training finished.")

# %%
# ============ Cell 5: Inferencing on UBFC_my (target domain) ============
BaseNet.eval()

HR_pr_list = []
HR_rel_list = []
BVP_ALL = []
BVP_PR_ALL = []
nan_debug_saved = False

with torch.no_grad():
    for step, (data, bvp, HR_rel, _, _, _, paths) in enumerate(tgt_loader):
        data = Variable(data).float().to(device=device)
        bvp = Variable(bvp).float().to(device=device)
        HR_rel = Variable(HR_rel).float().to(device=device)
        bvp = bvp.unsqueeze(dim=1)

        Wave = bvp
        Wave_pr, HR_pr, av = BaseNet(data)

        # Debug: check decoder signal output stability during inference.
        # This explains NaNs later in `Wave_sort` / `eval_from_bvp.py`.
        has_nan = torch.isnan(Wave_pr).any().item()
        has_inf = torch.isinf(Wave_pr).any().item()
        if (has_nan or has_inf) and not nan_debug_saved:
            nan_cnt = int(torch.isnan(Wave_pr).sum().item())
            inf_cnt = int(torch.isinf(Wave_pr).sum().item())
            print(
                f"[WARN] NaN/Inf in Wave_pr during inference: "
                f"step={step} nan_count={nan_cnt} inf_count={inf_cnt} "
                f"Wave_pr_shape={tuple(Wave_pr.shape)}"
            )
            nan_debug_saved = True

        HR_rel_list.extend(HR_rel.data.cpu().numpy())
        HR_pr_list.extend(HR_pr.data.cpu().numpy())
        BVP_ALL.extend(Wave.data.cpu().numpy())
        BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())

os.makedirs(config.RESULT_DIR, exist_ok=True)
io.savemat(os.path.join(config.RESULT_DIR, rPPGNet_name + '_HR_pr.mat'), {'HR_pr': HR_pr_list})
io.savemat(os.path.join(config.RESULT_DIR, rPPGNet_name + '_HR_rel.mat'), {'HR_rel': HR_rel_list})
io.savemat(os.path.join(config.RESULT_DIR, rPPGNet_name + '_WAVE_ALL.mat'), {'Wave': BVP_ALL})
io.savemat(os.path.join(config.RESULT_DIR, rPPGNet_name + '_WAVE_PR_ALL.mat'), {'Wave': BVP_PR_ALL})

os.makedirs(config.MODEL_DIR, exist_ok=True)
model_path = os.path.join(config.MODEL_DIR, rPPGNet_name)
torch.save(BaseNet, model_path)
print('Saved model:', os.path.abspath(model_path))

wave_sort_out = os.path.join(config.WAVE_SORT_ROOT, tgt_domain, rPPGNet_name)
train_utils.wave_sort_from_index(target_index_dir, np.array(BVP_ALL), np.array(BVP_PR_ALL), wave_sort_out)

# Write last Wave_sort path so eval_from_bvp can follow the latest train_regions run.
try:
    last_path_file = os.path.join(config.RESULT_LOG_DIR, "last_wave_sort_path.txt")
    os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)
    with open(last_path_file, "w") as f:
        f.write(os.path.abspath(wave_sort_out) + "\n")
    print("Saved last Wave_sort path to:", last_path_file)
except Exception as e:
    print("Warning: failed to write last_wave_sort_path.txt:", repr(e))

# %%