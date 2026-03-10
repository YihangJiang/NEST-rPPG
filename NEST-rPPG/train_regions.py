#!/usr/bin/env python
# %%
# -*- coding: UTF-8 -*-
# Region-aware (ROI) training: train on multiple region domains (e.g. cheek, target, eye),
# test on a different dataset/region. Source regions and test domain come from config.
# Run as script: python train_regions.py
# Run in Jupyter: execute cells in order (Cell 1 = config, Cell 2 = rest or run all)
%reload_ext autoreload
%autoreload 2

import os
from types import SimpleNamespace

from datetime import datetime
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %%

import scipy.io as io

from torch.utils.data import DataLoader
from torch.autograd import Variable

import MyDataset
import MyLoss
import model
import utils
from utils import Logger, time_to_str

import config

# %%
# ============ Cell 1: Config (constants) ============
# True = use constants below (Jupyter). False = use command-line args (python train_my.py ...).
_USE_JUPYTER_CONFIG = True

if _USE_JUPYTER_CONFIG:
    args = SimpleNamespace(
        GPU='0',
        num_workers=2,
        epochs=50,
        batchsize=100,
        lr=0.001,
        max_iter=3000,          # total training iterations (like train.py)
        seed=0,
        k1=1.0, k2=0.1, k3=1.0, k4=0.1, k5=1.0, k6=0.1, k7=0.1, k8=0.1,
        temporal_aug_rate=config.TEMPORAL_AUG_RATE,
        spatial_aug_rate=config.SPATIAL_AUG_RATE,
        loss_type=config.LOSS_TYPE,  # One / TA / CM / DM / All
        frames_num=256,
        # Domains (paths resolved via config.FILEA_NAME)
        test_domain='UBFC_my_in',
        # Baseline: single source ROI domain (default: config.TARGET_DOMAIN[test_domain][1], i.e., *_in)
        source_domain=None,
        stmap_name=config.STMAP_NAME,
        index_root=config.STMAP_INDEX_BASE,  # index root (subfolders are domain names)
        # Logging / model name suffix
        exp_name=config.EXP_NAME,
        # Wave_sort root (for per-subject BVP files)
        wave_sort_root=config.WAVE_SORT_ROOT,
    )
else:
    base_args = utils.get_args()
    args = base_args
    if not hasattr(args, 'frames_num'):
        args.frames_num = 256
    if not hasattr(args, 'test_domain'):
        args.test_domain = 'UBFC_my_in'
    if not hasattr(args, 'source_domain'):
        args.source_domain = None
    if not hasattr(args, 'stmap_name'):
        args.stmap_name = config.STMAP_NAME
    if not hasattr(args, 'index_root'):
        args.index_root = config.STMAP_INDEX_BASE
    if not hasattr(args, 'max_iter'):
        args.max_iter = 3000
    if not hasattr(args, 'exp_name'):
        args.exp_name = config.EXP_NAME
    if not hasattr(args, 'wave_sort_root'):
        args.wave_sort_root = config.WAVE_SORT_ROOT
    if not hasattr(args, 'loss_type'):
        args.loss_type = config.LOSS_TYPE

# ============ End Cell 1 ============

# %%
# ============ Cell 2: Dataset & index ============
print("=" * 60)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Baseline: only `_in` ROI is used for training,
# but we still keep all three region domains (cheek, target, eye) loaded
# so it is easy to plug in region-specific logic later.
# - test domain: args.test_domain (e.g. UBFC_my_in)
# - 3 region domains for this test domain: cheek, target, eye
src_domains = config.TARGET_DOMAIN[args.test_domain]
cheek_domain, target_region_domain, eye_domain = src_domains[0], src_domains[1], src_domains[2]

# For later region-aware methods, treat:
# - pos_domain: typically the cheek region (e.g. PURE_my_rm)
# - neg_domain: typically the eye region  (e.g. PURE_my_eye)
pos_domain = cheek_domain
neg_domain = eye_domain

# Baseline source domain (ROI "_in"): allow override via args.source_domain
default_source_domain = target_region_domain
source_domain = args.source_domain or default_source_domain

# Roots for each region domain and test domain
region_domains = [cheek_domain, target_region_domain, eye_domain]
region_roots = {
    d: os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[d][0])
    for d in region_domains
}
target_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[args.test_domain][0])
index_root = args.index_root

# Index dirs for each region and for the test domain
region_index_dirs = {d: os.path.join(index_root, d) for d in region_domains}
source_index_dir = region_index_dirs[source_domain]
target_index_dir = os.path.join(index_root, args.test_domain)

frames_num = args.frames_num
batch_size = args.batchsize
num_workers = args.num_workers
GPU = args.GPU

print("Source region domains (config.TARGET_DOMAIN[test]):", src_domains)
print("Test domain:", args.test_domain)
print("Baseline training domains (train.py-style):")
print("  source_domain:", source_domain)
print("  test_domain  :", args.test_domain)
print("Region STMap roots:")
print("  cheek (pos_domain):", region_roots[cheek_domain])
print("  target          :", region_roots[target_region_domain])
print("  eye   (neg_domain):", region_roots[eye_domain])
print("Test STMap root:", target_root)
print("Index root:", index_root)

os.makedirs(index_root, exist_ok=True)
# %%
def _build_index_if_needed(root_dir, index_dir, stmap_name, label):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"{label} root not found: {root_dir}")
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print(f"Building index for {label}:")
        files_list = sorted([f for f in os.listdir(root_dir) if not f.startswith('.')])
        MyDataset.getIndex(root_dir, files_list, index_dir, stmap_name, 10, frames_num)
    else:
        print(f"Using existing index for {label}: {index_dir}")

# Build / reuse indexes for all three region domains (cheek, target, eye)
for d in region_domains:
    _build_index_if_needed(region_roots[d], region_index_dirs[d], args.stmap_name, d)

# And for the test domain (target of evaluation)
_build_index_if_needed(target_root, target_index_dir, args.stmap_name, args.test_domain)

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
    dataName=config.canonical_data_name(args.test_domain),
    STMap=args.stmap_name,
    frames_num=frames_num,
    args=args
)

print("  baseline source dataset:", source_domain, "num_samples =", len(source_db))
print("  baseline test dataset  :", args.test_domain, "num_samples =", len(target_db))

print("Creating DataLoaders (baseline like train.py)...")
src_loader = DataLoader(source_db, batch_size=batch_size, shuffle=True, num_workers=num_workers)
tgt_loader = DataLoader(target_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Dataloaders for pos/neg domains (not used yet in baseline, but available)
pos_loader = DataLoader(pos_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)
neg_loader = DataLoader(neg_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

BaseNet = model.BaseNet().to(device=device)

optimizer = torch.optim.Adam(BaseNet.parameters(), lr=args.lr)
loss_func_NP = MyLoss.P_loss3().to(device)
loss_func_L1 = nn.L1Loss().to(device)
loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)
loss_func_NEST_CM = MyLoss.NEST_CM().to(device)
loss_func_NEST_DM = MyLoss.NEST_DM().to(device)
loss_func_NEST_TA = MyLoss.NEST_TA(device, Num_ref=8).to(device)

# Logging & model name (use config paths)
os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)

# Naming: rPPGNet_<test_domain>_src<target_region> (e.g. rPPGNet_UBFC_my_in_srcPURE_my_in)
Target_name = args.test_domain
rPPGNet_name = config.build_run_name(
    tgt=Target_name,
    src=source_domain,
    spatial=getattr(args, 'spatial_aug_rate', config.SPATIAL_AUG_RATE),
    temporal=getattr(args, 'temporal_aug_rate', config.TEMPORAL_AUG_RATE),
    loss_type=getattr(args, 'loss_type', config.LOSS_TYPE),
)

log = Logger()
log_path = os.path.join(config.RESULT_LOG_DIR, rPPGNet_name + '_log.txt')
log.open(log_path, mode='a')
log.write("\n----------------------------------------------- [START %s] %s\n\n" %
          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

# Print a train.py-style training summary for this baseline region run
Source_domain_Names = [source_domain]
total_src_samples = len(source_db)

print("TRAINING CONFIG (regions baseline)")
print("  Target domain:    ", Target_name)
print("  Target index:     ", target_index_dir)
print("  Target STMap:     ", args.stmap_name)
print("  Source domains:   ", Source_domain_Names)
for i, d in enumerate(Source_domain_Names):
    print("    [%d] %s -> %s" % (i, d, source_index_dir))
print("  Batch size:       ", batch_size)
print("  Max iterations:   ", args.max_iter)
print("  Frames per clip:  ", frames_num)
print("  Loss type:        ", getattr(args, 'loss_type', config.LOSS_TYPE))
print("  Device:           ", device)
print("  Model:            ", BaseNet.__class__.__name__)
print("  Source samples:   ", total_src_samples, " (target: %d)" % len(target_db))
print("  Log file:         ", log_path)
print("=" * 60 + "\n")

# %%
# ============ Cell 4: Training loop (train.py-style iter/max_iter) ============
BaseNet.train()
start = timer()
max_iter = args.max_iter

src_iter = iter(src_loader)
src_iter_per_epoch = len(src_iter)

for iter_num in range(max_iter + 1):
    # Reset iterator at epoch boundaries (same as train.py)
    if iter_num > 0 and (iter_num % src_iter_per_epoch == 0):
        src_iter = src_loader.__iter__()

    # Source batch (includes augmented view)
    data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug = src_iter.__next__()
    data = Variable(data).float().to(device=device)
    bvp = Variable(bvp).float().to(device=device).unsqueeze(dim=1)
    HR_rel = Variable(torch.Tensor(HR_rel)).float().to(device=device)
    data_aug = Variable(data_aug).float().to(device=device)
    bvp_aug = Variable(bvp_aug).float().to(device=device).unsqueeze(dim=1)
    HR_rel_aug = Variable(torch.Tensor(HR_rel_aug)).float().to(device=device)

    optimizer.zero_grad()
    bvp_pre, HR_pr, av = BaseNet(data)
    bvp_pre_aug, HR_pr_aug, av_aug = BaseNet(data_aug)

    src_loss = MyLoss.get_loss(
        bvp_pre, HR_pr, bvp, HR_rel, source_domain,
        loss_func_NP, loss_func_L1, args, iter_num
    )
    src_loss_aug = MyLoss.get_loss(
        bvp_pre_aug, HR_pr_aug, bvp_aug, HR_rel_aug, source_domain,
        loss_func_NP, loss_func_L1, args, iter_num
    )

    loss_CM = -loss_func_NEST_CM(torch.cat((av, av_aug), dim=0))
    loss_DM = loss_func_NEST_DM(av, av_aug)
    loss_TA = loss_func_NEST_TA(
        torch.cat((av, av_aug), dim=0),
        torch.cat((HR_rel, HR_rel_aug), dim=0)
    )

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

    if torch.sum(torch.isnan(loss)) > 0:
        print('Nan')
        break
    loss.backward()
    optimizer.step()

    if iter_num % 100 == 0:
        log_line = (
            'Train Inter:' + str(iter_num) +
            ' | loss:  ' + str(loss.data.cpu().numpy()) +
            ' |' + source_domain + ' : ' + str(src_loss.data.cpu().numpy()) +
            ' |' + 'CM' + ' : ' + str(loss_CM.data.cpu().numpy()) +
            ' |' + 'DM' + ' : ' + str(loss_DM.data.cpu().numpy()) +
            ' |' + 'TA' + ' : ' + str(loss_TA.data.cpu().numpy()) +
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

with torch.no_grad():
    for step, (data, bvp, HR_rel, _, _, _) in enumerate(tgt_loader):
        data = Variable(data).float().to(device=device)
        bvp = Variable(bvp).float().to(device=device)
        HR_rel = Variable(HR_rel).float().to(device=device)
        bvp = bvp.unsqueeze(dim=1)

        Wave = bvp
        Wave_pr, HR_pr, av = BaseNet(data)

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

# %%
# Wave_sort: regroup per-window BVP into per-subject files (use config path)
try:
    wave_sort_root = getattr(args, 'wave_sort_root', config.WAVE_SORT_ROOT)
    wave_sort_out = os.path.join(wave_sort_root, args.test_domain, rPPGNet_name)
    utils.train_utils.wave_sort_from_index(target_index_dir, np.array(BVP_ALL), np.array(BVP_PR_ALL), wave_sort_out)
except Exception as e:
    print('Warning: Wave_sort failed:', repr(e))

# %%