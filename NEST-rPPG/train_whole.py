#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Whole-face training with infraorbital-region inference.
# Train on full-face STMaps (e.g. PURE_my), evaluate on infraorbital crops (e.g. UBFC_my_in).
# Run as script: python train_whole.py --src PURE_my -t UBFC_my_in

import os
import json
import random
from types import SimpleNamespace

from datetime import datetime
from timeit import default_timer as timer
import torch
import torch.nn as nn
import numpy as np
import scipy.io as io

from torch.utils.data import DataLoader
from torch.autograd import Variable

import MyDataset
import MyLoss
import model
from utils.core import Logger, time_to_str, get_args
import utils.train_utils as train_utils
import utils.mlflow_utils as mlflow_utils

import config

_USE_JUPYTER_CONFIG = False
WHOLE_FACE_DOMAINS = {"PURE_my", "UBFC_my", "BUAA_my"}
REGION_SUFFIXES = {"rm", "in", "eye"}

if _USE_JUPYTER_CONFIG:
    args = SimpleNamespace(
        GPU='0',
        num_workers=2,
        epochs=50,
        batchsize=100,
        lr=0.001,
        max_iter=1000,
        seed=config.SEED,
        k1=1.0, k2=0.1, k3=1.0, k4=0.1, k5=1.0, k6=0.1, k7=0.1, k8=0.1,
        temporal_aug_rate=config.TEMPORAL_AUG_RATE,
        spatial_aug_rate=config.SPATIAL_AUG_RATE,
        loss_type=config.LOSS_TYPE,
        frames_num=512,
        tgt='UBFC_my_in',
        src='PURE_my',
        stmap_name=config.STMAP_NAME,
        index_root=config.STMAP_INDEX_BASE,
        save_features=False,
        grad_clip=5.0,
    )
else:
    args = get_args()
    for attr, default in [
        ('frames_num', 512),
        ('tgt', config.TGT_DOMAIN),
        ('stmap_name', config.STMAP_NAME),
        ('index_root', config.STMAP_INDEX_BASE),
        ('max_iter', 1000),
        ('loss_type', config.LOSS_TYPE),
        ('save_features', False),
        ('seed', 0),
        ('grad_clip', 5.0),
        ('weight_info', 0.0),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)
    if not hasattr(args, 'src') or args.src is None:
        raise ValueError("Please provide --src <whole_face_domain> (e.g. --src PURE_my)")

print("=" * 60)

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _worker_init_fn(worker_id):
    base_seed = int(getattr(args, 'seed', 0))
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)

_set_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("  Random seed:", args.seed, "(deterministic training)")

tgt_domain = args.tgt
if tgt_domain not in config.FILEA_NAME:
    raise ValueError(f"Target domain has no FILEA_NAME entry: {tgt_domain}")

_head, _tail = tgt_domain.rsplit("_", 1)
if _tail != "in":
    raise ValueError(
        f"--tgt must be an infraorbital domain ending in '_in', got {tgt_domain!r}"
    )

source_domain = args.src
if source_domain not in WHOLE_FACE_DOMAINS:
    raise ValueError(
        f"--src must be a whole-face domain in {sorted(WHOLE_FACE_DOMAINS)}, got {source_domain!r}"
    )
if source_domain not in config.FILEA_NAME:
    raise ValueError(f"Source domain has no FILEA_NAME entry: {source_domain}")

_src_head, src_tail = source_domain.rsplit("_", 1)
if src_tail in REGION_SUFFIXES:
    raise ValueError(
        f"--src must be whole-face (e.g. PURE_my), not a region crop; got {source_domain!r}"
    )

index_root = args.index_root
source_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[source_domain][0])
source_index_dir = os.path.join(index_root, source_domain)
target_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[tgt_domain][0])
target_index_dir = os.path.join(index_root, tgt_domain)

frames_num = args.frames_num
batch_size = args.batchsize
num_workers = args.num_workers
GPU = args.GPU

print("Whole-face train / infraorbital infer:")
print("  source_domain :", source_domain)
print("  tgt_domain    :", tgt_domain)
print("  source root   :", source_root)
print("  target root   :", target_root)
print("  index root    :", index_root)

os.makedirs(index_root, exist_ok=True)

def _build_index_always(root_dir, index_dir, stmap_name, label):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"{label} root not found: {root_dir}")
    os.makedirs(index_dir, exist_ok=True)

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

_build_index_always(source_root, source_index_dir, args.stmap_name, source_domain)
_build_index_always(target_root, target_index_dir, args.stmap_name, tgt_domain)

print("Loading datasets...")
source_db = MyDataset.Data_DG(
    root_dir=source_index_dir,
    dataName=config.canonical_data_name(source_domain),
    STMap=args.stmap_name,
    frames_num=frames_num,
    args=args,
)
target_db = MyDataset.Data_DG(
    root_dir=target_index_dir,
    dataName=config.canonical_data_name(tgt_domain),
    STMap=args.stmap_name,
    frames_num=frames_num,
    args=args,
)
print("  source dataset:", source_domain, "num_samples =", len(source_db))
print("  target dataset:", tgt_domain, "num_samples =", len(target_db))

_generator = torch.Generator().manual_seed(args.seed)
src_loader = DataLoader(
    source_db, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    worker_init_fn=_worker_init_fn, generator=_generator,
)
tgt_loader = DataLoader(
    target_db, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    worker_init_fn=_worker_init_fn,
)

steps_per_epoch = len(src_loader)
print(f"steps_per_epoch (source) = {steps_per_epoch}")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device('cuda:' + GPU)
    print('Using device:', device)
else:
    device = torch.device('cpu')
    print('Using CPU')

BaseNet = model.BaseNet().to(device=device)
optimizer = torch.optim.Adam(BaseNet.parameters(), lr=args.lr)
loss_func_NP = MyLoss.P_loss3().to(device)
loss_func_L1 = nn.L1Loss().to(device)
loss_func_NEST_CM = MyLoss.NEST_CM().to(device)
loss_func_NEST_DM = MyLoss.NEST_DM().to(device)
loss_func_NEST_TA = MyLoss.NEST_TA(device, Num_ref=8).to(device)

os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)
Target_name = tgt_domain
rPPGNet_name = config.build_run_name(
    tgt=Target_name,
    src=source_domain,
    weight_info=float(getattr(args, 'weight_info', config.WEIGHT_INFO)),
)

log = Logger()
log_path = os.path.join(config.RESULT_LOG_DIR, rPPGNet_name + '_log.txt')
log.open(log_path, mode='a')
log.write("\n----------------------------------------------- [START %s] %s\n\n" %
          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write("TRAINING CONFIG (whole-face train / infraorbital infer)\n")
log.write("  Target domain:     %s\n" % Target_name)
log.write("  Target index:      %s\n" % target_index_dir)
log.write("  Source domain:     %s\n" % source_domain)
log.write("  Source index:      %s\n" % source_index_dir)
log.write("  Batch size:        %s\n" % batch_size)
log.write("  Max iterations:    %s\n" % args.max_iter)
log.write("  Frames per clip:   %s\n" % frames_num)
log.write("  Loss type:         %s\n" % getattr(args, 'loss_type', config.LOSS_TYPE))
log.write("  Device:            %s\n" % device)
log.write("  Model:             %s\n" % BaseNet.__class__.__name__)
log.write("  Source samples:    %s  (target: %d)\n" % (len(source_db), len(target_db)))
log.write("  Log file:          %s\n" % log_path)
log.write("=" * 60 + "\n\n")

mlflow_utils.setup(
    args,
    experiment_name=getattr(args, 'mlflow_experiment', None) or 'nest-rppg-whole',
    run_name=rPPGNet_name,
    tags={'script': 'train_whole', 'rPPGNet_name': rPPGNet_name},
)
mlflow_utils.log_params({
    'source_domain': source_domain,
    'target_domain': tgt_domain,
    'weight_info': float(getattr(args, 'weight_info', 0.0)),
    'loss_type': getattr(args, 'loss_type', config.LOSS_TYPE),
    'lr': args.lr,
    'batchsize': batch_size,
    'max_iter': args.max_iter,
    'frames_num': frames_num,
    'seed': args.seed,
    'grad_clip': float(args.grad_clip),
})

BaseNet.train()
start = timer()
max_iter = args.max_iter
src_iter = iter(src_loader)
src_iter_per_epoch = len(src_iter)
_printed_nan_debug = False

for iter_num in range(max_iter + 1):
    if iter_num > 0 and (iter_num % src_iter_per_epoch == 0):
        src_iter = src_loader.__iter__()

    data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug, subj_paths = src_iter.__next__()
    data = Variable(data).float().to(device=device)
    bvp = Variable(bvp).float().to(device=device).unsqueeze(dim=1)
    HR_rel = Variable(torch.Tensor(HR_rel)).float().to(device=device)
    data_aug = Variable(data_aug).float().to(device=device)
    bvp_aug = Variable(bvp_aug).float().to(device=device).unsqueeze(dim=1)
    HR_rel_aug = Variable(torch.Tensor(HR_rel_aug)).float().to(device=device)

    optimizer.zero_grad()
    bvp_pre, HR_pr, av = BaseNet(data)
    bvp_pre_aug, HR_pr_aug, av_aug = BaseNet(data_aug)

    if not _printed_nan_debug:
        any_nan = (
            torch.isnan(data).any() or torch.isnan(bvp_pre).any() or torch.isnan(av).any() or
            torch.isnan(data_aug).any() or torch.isnan(bvp_pre_aug).any() or torch.isnan(av_aug).any()
        )
        if any_nan:
            print("\n[NaN DEBUG] First NaN detected at iter_num =", iter_num)
            _printed_nan_debug = True

    if getattr(args, 'save_features', False):
        av_cpu = av.detach().cpu().numpy()
        for path_str, feat in zip(subj_paths, av_cpu):
            if not isinstance(path_str, str):
                path_str = str(path_str)
            feat_path = os.path.join(path_str, f"feat_iter_{iter_num:06d}.npy")
            try:
                os.makedirs(path_str, exist_ok=True)
                np.save(feat_path, feat)
            except Exception as e:
                print(f"Warning: failed to save feature to {feat_path}: {repr(e)}")

    src_loss = MyLoss.get_loss(
        bvp_pre, HR_pr, bvp, HR_rel, source_domain,
        loss_func_NP, loss_func_L1, args, iter_num,
    )
    loss_CM = -loss_func_NEST_CM(torch.cat((av, av_aug), dim=0))
    loss_DM = loss_func_NEST_DM(av, av_aug)
    loss_TA = loss_func_NEST_TA(
        torch.cat((av, av_aug), dim=0),
        torch.cat((HR_rel, HR_rel_aug), dim=0),
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
        mlflow_utils.log_params({'stopped_early': True})
        break
    loss.backward()
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
            ' |' + time_to_str(timer() - start, 'min')
        )
        log.write(log_line)
        log.write('\n')
        mlflow_utils.log_metrics({
            'loss': float(loss.data.cpu().numpy()),
            'src_loss': float(src_loss.data.cpu().numpy()),
            'loss_CM': float(loss_CM.data.cpu().numpy()),
            'loss_DM': float(loss_DM.data.cpu().numpy()),
            'loss_TA': float(loss_TA.data.cpu().numpy()),
            'elapsed_min': float((timer() - start) / 60.0),
        }, step=iter_num)

print("Training finished.")

BaseNet.eval()
HR_pr_list = []
HR_rel_list = []
BVP_ALL = []
BVP_PR_ALL = []
nan_debug_saved = False

with torch.no_grad():
    for step, (data, bvp, HR_rel, _, _, _, _paths) in enumerate(tgt_loader):
        data = Variable(data).float().to(device=device)
        bvp = Variable(bvp).float().to(device=device)
        HR_rel = Variable(HR_rel).float().to(device=device)
        bvp = bvp.unsqueeze(dim=1)

        Wave = bvp
        Wave_pr, HR_pr, _av = BaseNet(data)

        has_nan = torch.isnan(Wave_pr).any().item()
        has_inf = torch.isinf(Wave_pr).any().item()
        if (has_nan or has_inf) and not nan_debug_saved:
            print(
                f"[WARN] NaN/Inf in Wave_pr during inference: "
                f"step={step} Wave_pr_shape={tuple(Wave_pr.shape)}"
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

mlflow_utils.log_model(BaseNet)
print('Saved model to MLflow run:', mlflow_utils.get_run_id())

wave_sort_out = os.path.join(config.WAVE_SORT_ROOT, tgt_domain, rPPGNet_name)
train_utils.wave_sort_from_index(target_index_dir, np.array(BVP_ALL), np.array(BVP_PR_ALL), wave_sort_out)

try:
    last_path_file = os.path.join(config.RESULT_LOG_DIR, "last_wave_sort_path.txt")
    os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)
    with open(last_path_file, "w") as f:
        f.write(os.path.abspath(wave_sort_out) + "\n")
    print("Saved last Wave_sort path to:", last_path_file)
except Exception as e:
    print("Warning: failed to write last_wave_sort_path.txt:", repr(e))

meta_path = os.path.join(config.RESULT_LOG_DIR, "last_train_regions_meta.json")
try:
    with open(meta_path, "w") as f:
        json.dump(
            {
                "source_domain": source_domain,
                "target_domain": tgt_domain,
                "weight_info": float(getattr(args, "weight_info", 0.0)),
                "regions": "whole",
            },
            f,
            indent=2,
        )
    print("Saved train_whole meta to:", meta_path)
except Exception as e:
    print("Warning: failed to write last_train_regions_meta.json:", repr(e))

mlflow_utils.log_artifacts([log_path, meta_path])
mlflow_utils.end_run()
