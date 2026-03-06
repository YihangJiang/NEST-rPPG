#!/usr/bin/env python
# %%
# -*- coding: UTF-8 -*-
# Region-aware training on PURE_my_rm (cheek), PURE_my_in (target), PURE_my_eye (eye)
# Train on PURE_my regions, test on UBFC_my.
# Run as script: python train_my.py
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
        batchsize=64,
        lr=0.001,
        max_iter=400,          # total training iterations (like train.py)
        seed=0,
        k1=1.0, k2=0.1, k3=1.0, k4=0.1, k5=1.0, k6=0.1, k7=0.1, k8=0.1,
        temporal_aug_rate=config.TEMPORAL_AUG_RATE,
        spatial_aug_rate=config.SPATIAL_AUG_RATE,
        frames_num=256,
        # Domains (paths resolved via config.FILEA_NAME)
        test_domain='UBFC_my_in',
        stmap_name=config.STMAP_NAME,
        index_root=config.STMAP_INDEX_BASE,  # index root (subfolders are domain names)
        # Region alignment loss weights
        lambda_align=0.5,
        lambda_sep=0.5,
        triplet_margin=0.5,
        # Logging / model name suffix
        exp_name=config.EXP_NAME,
        # Wave_sort root (for per-subject BVP files)
        wave_sort_root='./Wave_sort',
    )
else:
    base_args = utils.get_args()
    # Map generic args into what we need; keep defaults for new ones
    args = base_args
    if not hasattr(args, 'frames_num'):
        args.frames_num = 256
    if not hasattr(args, 'test_domain'):
        args.test_domain = 'UBFC_my_in'
    if not hasattr(args, 'stmap_name'):
        args.stmap_name = config.STMAP_NAME
    if not hasattr(args, 'index_root'):
        args.index_root = config.STMAP_INDEX_BASE
    if not hasattr(args, 'max_iter'):
        args.max_iter = 3000
    if not hasattr(args, 'lambda_align'):
        args.lambda_align = 0.5
    if not hasattr(args, 'lambda_sep'):
        args.lambda_sep = 0.5
    if not hasattr(args, 'triplet_margin'):
        args.triplet_margin = 0.5
    if not hasattr(args, 'exp_name'):
        args.exp_name = config.EXP_NAME
    if not hasattr(args, 'wave_sort_root'):
        args.wave_sort_root = './Wave_sort'

# ============ End Cell 1 ============

# %%
# ============ Cell 2: Dataset & index ============
print("=" * 60)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 3 source-region domains (order matters: cheek, target, eye) + 1 test domain
src_domains = config.TARGET_DOMAIN[args.test_domain]
cheek_domain, target_domain, eye_domain = src_domains[0], src_domains[1], src_domains[2]

cheek_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[cheek_domain][0])
target_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[target_domain][0])
eye_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[eye_domain][0])
ubfc_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[args.test_domain][0])
index_root = args.index_root

cheek_index_dir = os.path.join(index_root, cheek_domain)
target_index_dir = os.path.join(index_root, target_domain)
eye_index_dir = os.path.join(index_root, eye_domain)
ubfc_index_dir = os.path.join(index_root, args.test_domain)

frames_num = args.frames_num
batch_size = args.batchsize
num_workers = args.num_workers
GPU = args.GPU

print("Source region domains (train):", src_domains)
print("Test domain:", args.test_domain)
print("Region STMap roots (train):")
print("  cheek :", cheek_root)
print("  target:", target_root)
print("  eye   :", eye_root)
print("Test STMap root:", ubfc_root)
print("Index root:", index_root)

os.makedirs(index_root, exist_ok=True)

def _build_index_if_needed(root_dir, index_dir, stmap_name, label):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"{label} root not found: {root_dir}")
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print(f"Building index for {label}:")
        files_list = sorted([f for f in os.listdir(root_dir) if not f.startswith('.')])
        MyDataset.getIndex(root_dir, files_list, index_dir, stmap_name, 10, frames_num)
    else:
        print(f"Using existing index for {label}: {index_dir}")

_build_index_if_needed(cheek_root, cheek_index_dir, args.stmap_name, cheek_domain)
_build_index_if_needed(target_root, target_index_dir, args.stmap_name, target_domain)
_build_index_if_needed(eye_root, eye_index_dir, args.stmap_name, eye_domain)
_build_index_if_needed(ubfc_root, ubfc_index_dir, args.stmap_name, args.test_domain)

print("Loading datasets...")

cheek_db = MyDataset.Data_DG(root_dir=cheek_index_dir, dataName=config.canonical_data_name(cheek_domain),
                             STMap=args.stmap_name, frames_num=frames_num, args=args)
target_db = MyDataset.Data_DG(root_dir=target_index_dir, dataName=config.canonical_data_name(target_domain),
                              STMap=args.stmap_name, frames_num=frames_num, args=args)
eye_db = MyDataset.Data_DG(root_dir=eye_index_dir, dataName=config.canonical_data_name(eye_domain),
                           STMap=args.stmap_name, frames_num=frames_num, args=args)
ubfc_db = MyDataset.Data_DG(root_dir=ubfc_index_dir, dataName=config.canonical_data_name(args.test_domain),
                            STMap=args.stmap_name, frames_num=frames_num, args=args)

print("  cheek dataset (PURE_my_rm) :", len(cheek_db))
print("  target dataset (PURE_my_in):", len(target_db))
print("  eye dataset (PURE_my_eye)  :", len(eye_db))
print("  test dataset (UBFC_my)     :", len(ubfc_db))

print("Creating DataLoaders (shuffle=False to keep PURE triplets aligned)...")
cheek_loader = DataLoader(cheek_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)
target_loader = DataLoader(target_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)
eye_loader = DataLoader(eye_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)
ubfc_loader = DataLoader(ubfc_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)

steps_per_epoch = min(len(cheek_loader), len(target_loader), len(eye_loader))
print(f"steps_per_epoch = {steps_per_epoch}")
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

lambda_align = args.lambda_align
lambda_sep = args.lambda_sep
margin = args.triplet_margin

# Logging & model name
if not os.path.exists('./Result_log'):
    os.makedirs('./Result_log')

# Naming style similar to train.py:
# rPPGNet_<Target_name>_src<PURE_my_in>
Target_name = args.test_domain
src_suffix = '_src' + src_domains[1]
rPPGNet_name = 'rPPGNet_' + Target_name + src_suffix

log = Logger()
log_path = f'./Result_log/{rPPGNet_name}_log.txt'
log.open(log_path, mode='a')
log.write("\n----------------------------------------------- [START %s] %s\n\n" %
          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

print("Training config:")
print("  Model name    :", rPPGNet_name)
print("  epochs        :", args.epochs)
print("  batch_size    :", batch_size)
print("  lambda_align  :", lambda_align)
print("  lambda_sep    :", lambda_sep)
print("  margin        :", margin)
print("  log file      :", log_path)
print("=" * 60)

# %%
# ============ Cell 4: Training loop (train.py-style iter/max_iter) ============
BaseNet.train()
start = timer()
global_step = 0
max_iter = args.max_iter

# Initial iterators
cheek_iter = iter(cheek_loader)
target_iter = iter(target_loader)
eye_iter = iter(eye_loader)

for iter_num in range(max_iter + 1):
    # Reset iterators at (approximate) epoch boundaries
    if iter_num > 0 and (iter_num % steps_per_epoch == 0):
        cheek_iter = iter(cheek_loader)
        target_iter = iter(target_loader)
        eye_iter = iter(eye_loader)

    try:
        c_data, c_bvp, c_HR, _, _, _ = next(cheek_iter)
        t_data, t_bvp, t_HR, _, _, _ = next(target_iter)
        e_data, e_bvp, e_HR, _, _, _ = next(eye_iter)
    except StopIteration:
        break

    # Move to device
    c_data = Variable(c_data).float().to(device=device)
    t_data = Variable(t_data).float().to(device=device)
    e_data = Variable(e_data).float().to(device=device)

    c_bvp = Variable(c_bvp).float().to(device=device).unsqueeze(dim=1)
    t_bvp = Variable(t_bvp).float().to(device=device).unsqueeze(dim=1)
    e_bvp = Variable(e_bvp).float().to(device=device).unsqueeze(dim=1)

    c_HR = Variable(torch.Tensor(c_HR)).float().to(device=device)
    t_HR = Variable(torch.Tensor(t_HR)).float().to(device=device)
    e_HR = Variable(torch.Tensor(e_HR)).float().to(device=device)

    optimizer.zero_grad()

    # Forward for each region
    c_bvp_pre, c_HR_pr, c_av = BaseNet(c_data)
    t_bvp_pre, t_HR_pr, t_av = BaseNet(t_data)
    e_bvp_pre, e_HR_pr, e_av = BaseNet(e_data)

    # Supervised loss: ONLY target region (PURE_my_in)
    target_sup_loss = loss_func_NP(t_bvp_pre, t_bvp)

    # Feature-level region losses
    c_feat = F.normalize(c_av, p=2, dim=1)
    t_feat = F.normalize(t_av, p=2, dim=1)
    e_feat = F.normalize(e_av, p=2, dim=1)

    # Align target with cheek
    align_loss = torch.mean(torch.sum((t_feat - c_feat) ** 2, dim=1))

    # Separate target from eye (triplet-style)
    pos_dist = F.pairwise_distance(t_feat, c_feat)
    neg_dist = F.pairwise_distance(t_feat, e_feat)
    sep_loss = torch.mean(torch.relu(margin + pos_dist - neg_dist))

    # total_loss = target_sup_loss + lambda_align * align_loss + lambda_sep * sep_loss
    total_loss = target_sup_loss


    if torch.sum(torch.isnan(total_loss)) > 0:
        print('NaN loss encountered, stopping.')
        break

    total_loss.backward()
    optimizer.step()

    if global_step % 50 == 0:
        elapsed = time_to_str(timer() - start, 'min')
        epoch = iter_num // max(steps_per_epoch, 1)
        log_line = (
            f"Epoch {epoch} Step {global_step} | "
            f"total: {total_loss.data.cpu().numpy():.4f} | "
            f"target_sup: {target_sup_loss.data.cpu().numpy():.4f} | "
            f"align: {align_loss.data.cpu().numpy():.4f} | "
            f"sep: {sep_loss.data.cpu().numpy():.4f} | "
            f"{elapsed}"
        )
        print(log_line)
        log.write(log_line + "\n")

    global_step += 1

print("Training finished.")

# %%
# ============ Cell 5: Inferencing on UBFC_my (target domain) ============
BaseNet.eval()

HR_pr_list = []
HR_rel_list = []
BVP_ALL = []
BVP_PR_ALL = []

with torch.no_grad():
    for step, (data, bvp, HR_rel, _, _, _) in enumerate(ubfc_loader):
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

if not os.path.exists('./Result'):
    os.makedirs('./Result')

io.savemat(f'./Result/{rPPGNet_name}_HR_pr.mat', {'HR_pr': HR_pr_list})
io.savemat(f'./Result/{rPPGNet_name}_HR_rel.mat', {'HR_rel': HR_rel_list})
io.savemat(f'./Result/{rPPGNet_name}_WAVE_ALL.mat', {'Wave': BVP_ALL})
io.savemat(f'./Result/{rPPGNet_name}_WAVE_PR_ALL.mat', {'Wave': BVP_PR_ALL})

if not os.path.exists('./model'):
    os.makedirs('./model')
model_path = f'./model/{rPPGNet_name}'
torch.save(BaseNet, model_path)
print('Saved model:', os.path.abspath(model_path))

# %%
# Wave_sort: regroup per-window BVP into per-subject files (shared helper)
try:
    wave_sort_out = os.path.join(args.wave_sort_root, args.test_domain, rPPGNet_name)
    utils.train_utils.wave_sort_from_index(ubfc_index_dir, np.array(BVP_ALL), np.array(BVP_PR_ALL), wave_sort_out)
except Exception as e:
    print('Warning: Wave_sort failed:', repr(e))

# %%