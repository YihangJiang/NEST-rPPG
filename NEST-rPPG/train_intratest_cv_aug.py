#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Intratest 5-fold cross-validation using BaseNet.

Trains and evaluates on the same dataset with subject-level 80/20 splits.
Each fold trains a fresh BaseNet on 80% of subjects and tests on the remaining 20%.
Metrics are aggregated across folds at the end.

Usage:
    python train_intratest_cv.py --dataset PURE_my_in --folds 5 --max_iter 1000
    python train_intratest_cv.py --dataset UBFC_my_in --folds 5 --max_iter 1000
    python train_intratest_cv.py --dataset BUAA_my_in --folds 5 --max_iter 1000
"""

import os
import shutil
import random
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import scipy.io as io
from datetime import datetime
from timeit import default_timer as timer
from torch.autograd import Variable

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import MyDataset
import MyLoss
import model
from utils.core import Logger, time_to_str
import utils.train_utils as train_utils
from utils.eval_utils import run_eval
import config


def parse_args():
    parser = argparse.ArgumentParser(description='Intratest 5-fold CV with BaseNet')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset domain (e.g. PURE_my_in, UBFC_my_in, BUAA_my_in)')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--GPU', type=str, default='0')
    parser.add_argument('-p', '--num_workers', dest='num_workers', type=int, default=2)
    parser.add_argument('-b', '--batchsize', type=int, default=100)
    parser.add_argument('-l', '--lr', type=float, default=0.001)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--frames_num', type=int, default=512)
    parser.add_argument('--stmap_name', type=str, default=config.STMAP_NAME)
    parser.add_argument('--loss_type', type=str, default=config.LOSS_TYPE,
                        help='One / TA / CM / DM / All')
    parser.add_argument('--temporal_aug_rate', type=float, default=config.TEMPORAL_AUG_RATE)
    parser.add_argument('--spatial_aug_rate', type=float, default=config.SPATIAL_AUG_RATE)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--weight_cl', type=float, default=0.0,
                        help='Weight for NEST contrastive losses (CM+DM+TA). '
                             '0 = no contrastive (src loss only), 0.01 = with contrastive.')
    # k-weights required by MyLoss.get_loss
    parser.add_argument('--k1', type=float, default=1.0)
    parser.add_argument('--k2', type=float, default=0.1)
    parser.add_argument('--k3', type=float, default=1.0)
    parser.add_argument('--k4', type=float, default=0.1)
    parser.add_argument('--k5', type=float, default=1.0)
    parser.add_argument('--k6', type=float, default=0.1)
    parser.add_argument('--k7', type=float, default=0.1)
    parser.add_argument('--k8', type=float, default=0.1)
    # Unused but required by some internal code paths
    parser.add_argument('--epochs', type=int, default=64 * 64 * 64)
    parser.add_argument('--use_infonce', action='store_true', default=False)
    parser.add_argument('--exclude_participants', type=str, default='',
                        help='Comma-separated participant IDs to exclude (e.g. "07,08")')
    parser.add_argument('--session_filter', type=str, default='',
                        help='Keep only session folders whose name contains this substring '
                             '(e.g. "100.0" to keep only 100 lux BUAA sessions)')
    parser.add_argument('--hr_aug_max', type=float, default=1.0,
                        help='Max temporal compression factor (1.0 = off, 2.0 = up to 2× HR)')
    parser.add_argument('--hr_aug_min', type=float, default=1.0,
                        help='Min temporal stretch factor (1.0 = off, 0.5 = down to 0.5× HR). '
                             'When <1.0, enables both-ends augmentation (slow + fast HR).')
    parser.add_argument('--hr_aug_prob', type=float, default=0.5,
                        help='Probability of applying HR augmentation per training sample')
    parser.add_argument('--hr_aug_exclude', type=str, default='',
                        help='Participant IDs to skip HR augmentation for (comma-separated)')
    parser.add_argument('--run_tag', type=str, default='aug',
                        help='Tag appended to output CSV name to distinguish from baseline')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id, base_seed=0):
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)


def get_participant_id(folder_name):
    """Extract unique participant ID from a session folder name.

    PURE:  '01-02' -> '01'         (split on '-', take first token)
    BUAA:  'Sub_03lux 25.1' -> 'Sub_03'  (split on 'lux', take first token)
    UBFC:  'subject5' -> 'subject5'   (no sessions, folder is already unique)
    """
    if '-' in folder_name and not folder_name.startswith('Sub_'):
        return folder_name.split('-')[0]
    if 'lux' in folder_name:
        return folder_name.split('lux')[0]
    return folder_name


def subject_kfold_split(subjects, n_folds, fold_idx):
    """Return (train_folders, test_folders) for fold_idx (0-indexed).

    Splits at participant level to prevent the same person appearing in both
    train and test (e.g. PURE '01-01'..'01-06' all belong to participant '01').
    All session folders of a participant move together into train or test.
    """
    # Unique participants in sorted order (preserves reproducibility)
    seen = {}
    for s in subjects:
        pid = get_participant_id(s)
        if pid not in seen:
            seen[pid] = []
        seen[pid].append(s)
    participants = sorted(seen.keys())

    n = len(participants)
    fold_sizes = [n // n_folds + (1 if i < n % n_folds else 0) for i in range(n_folds)]
    starts = [sum(fold_sizes[:i]) for i in range(n_folds)]

    test_participants  = participants[starts[fold_idx]:starts[fold_idx] + fold_sizes[fold_idx]]
    train_participants = participants[:starts[fold_idx]] + participants[starts[fold_idx] + fold_sizes[fold_idx]:]

    test_folders  = [s for s in subjects if get_participant_id(s) in set(test_participants)]
    train_folders = [s for s in subjects if get_participant_id(s) in set(train_participants)]
    return train_folders, test_folders


class HRAugDataset(Dataset):
    """Training dataset with temporal compression to simulate higher HR.

    Loads a window of src_len = round(frames_num * r) frames and lets
    transforms.Resize((64, frames_num)) compress it to frames_num columns.
    Cardiac cycles appearing in src_len source frames are compressed into
    frames_num columns, so the apparent HR scales by r = src_len / frames_num.

    The compressed view is placed in the primary slot (data, bvp, HR_rel) so
    src_loss trains against the augmented-HR signal. The original frames_num
    window goes in the aux slot for NEST losses (unused when weight_cl=0).

    Participants in exclude_aug_participants are never augmented (their HR is
    already anomalous and further compression would push labels out of range).
    """

    def __init__(self, index_dir, data_name, stmap_name, frames_num, args,
                 hr_aug_max=2.0, hr_aug_min=1.0, hr_aug_prob=0.5, exclude_aug_participants=None):
        self.index_dir = index_dir
        self.data_name = data_name
        self.stmap_name = stmap_name
        self.frames_num = int(frames_num)
        self.args = args
        self.hr_aug_max = hr_aug_max
        self.hr_aug_min = hr_aug_min
        self.hr_aug_prob = hr_aug_prob
        self.exclude_aug = set(exclude_aug_participants or [])
        self.datalist = sorted(os.listdir(index_dir))
        resize_size = (64, self.frames_num)
        self.transform = transforms.Compose([
            transforms.Resize(size=resize_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.datalist)

    def check_augmentation(self, n_samples=200):
        """Diagnostic: report what fraction of samples can actually be augmented.

        Reads STMap widths for the first n_samples index files and simulates the
        augmentation decision with a fixed seed. Returns a summary dict.
        Runs in the calling process (num_workers=0 semantics).
        """
        old_state = random.getstate()
        random.seed(42)

        check_n = min(n_samples, len(self.datalist))
        can_aug_count = 0
        eligible_count = 0   # max_r > 1.2
        aug_count = 0        # r > 1.0 (actually augmented)
        r_values = []
        max_r_values = []
        w_full_values = []

        for i in range(check_n):
            mat = io.loadmat(os.path.join(self.index_dir, self.datalist[i]))
            now_path = str(mat['Path'][0])
            step_index = int(mat['Step_Index'].flat[0])
            img = cv2.imread(os.path.join(now_path, 'STMap', self.stmap_name))
            if img is None:
                continue
            _, W_full, _ = img.shape
            w_full_values.append(W_full)
            folder = os.path.basename(now_path)
            participant = get_participant_id(folder)
            can_aug = participant not in self.exclude_aug
            if can_aug:
                can_aug_count += 1
            max_r = min(self.hr_aug_max, (W_full - step_index) / self.frames_num)
            max_r_values.append(max_r)
            if max_r > 1.2:
                eligible_count += 1
            if can_aug and random.random() < self.hr_aug_prob:
                r = random.uniform(1.2, max_r) if max_r > 1.2 else 1.0
            else:
                r = 1.0
            r_values.append(r)
            if r > 1.0:
                aug_count += 1

        random.setstate(old_state)
        return {
            'checked': check_n,
            'can_aug_participant': can_aug_count,
            'eligible_max_r_gt_1.2': eligible_count,
            'actually_augmented': aug_count,
            'aug_pct': 100.0 * aug_count / check_n if check_n else 0,
            'mean_max_r': float(np.mean(max_r_values)) if max_r_values else 0,
            'mean_W_full': float(np.mean(w_full_values)) if w_full_values else 0,
            'r_sample': [round(v, 2) for v in r_values[:20]],
        }

    def _row_normalize(self, crop):
        out = crop.copy().astype(np.float32)
        for c in range(out.shape[2]):
            for r in range(out.shape[0]):
                row = out[r, :, c]
                rng = row.max() - row.min()
                out[r, :, c] = 255.0 * (row - row.min()) / (rng + 1e-5)
        return out

    def _load_label(self, now_path, step_index, length):
        bvp_raw = io.loadmat(os.path.join(now_path, 'Label', 'BVP.mat'))['BVP'].reshape(-1).astype(np.float32)
        seg = bvp_raw[step_index:step_index + length]
        seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-8)
        hr_raw = io.loadmat(os.path.join(now_path, 'Label', 'HR.mat'))['HR'].reshape(-1).astype(np.float32)
        gt = float(np.nanmean(hr_raw[step_index:step_index + length]))
        return seg, gt

    def __getitem__(self, idx):
        mat = io.loadmat(os.path.join(self.index_dir, self.datalist[idx]))
        now_path = str(mat['Path'][0])
        step_index = int(mat['Step_Index'].flat[0])

        img = cv2.imread(os.path.join(now_path, 'STMap', self.stmap_name))
        if img is None:
            # Fallback: return a zero sample (rare edge case)
            z = np.zeros((3, 64, self.frames_num), dtype=np.float32)
            zb = np.zeros(self.frames_num, dtype=np.float32)
            zt = torch.from_numpy(z)
            return (zt, zb, np.float32(0.0), zt, zb, np.float32(0.0), now_path)

        _, W_full, _ = img.shape

        # Check whether this participant is excluded from augmentation
        folder = os.path.basename(now_path)
        participant = get_participant_id(folder)
        can_aug = participant not in self.exclude_aug

        # Draw augmentation factor r
        # r > 1.0 → compress (faster HR), r < 1.0 → stretch (slower HR)
        can_go_high = self.hr_aug_max > 1.2
        can_go_low  = self.hr_aug_min < 0.8

        if can_aug and random.random() < self.hr_aug_prob:
            if can_go_high and can_go_low:
                if random.random() < 0.5:
                    # High-end: compress STMap → faster HR
                    max_r = min(self.hr_aug_max, (W_full - step_index) / self.frames_num)
                    r = random.uniform(1.2, max_r) if max_r > 1.2 else 1.0
                else:
                    # Low-end: stretch STMap → slower HR
                    r = random.uniform(self.hr_aug_min, 0.8)
            elif can_go_high:
                max_r = min(self.hr_aug_max, (W_full - step_index) / self.frames_num)
                r = random.uniform(1.2, max_r) if max_r > 1.2 else 1.0
            elif can_go_low:
                r = random.uniform(self.hr_aug_min, 0.8)
            else:
                r = 1.0
        else:
            r = 1.0

        src_len = max(1, min(round(self.frames_num * r), W_full - step_index))

        # Always build the original frames_num crop (aux slot / fallback)
        crop_orig = self._row_normalize(img[:, step_index:step_index + self.frames_num, :])
        bvp_orig, hr_orig = self._load_label(now_path, step_index, self.frames_num)
        map_orig = self.transform(Image.fromarray(np.uint8(crop_orig)))

        if src_len > self.frames_num:
            # Long crop → Resize compresses width from src_len → frames_num (faster HR)
            crop_long = self._row_normalize(img[:, step_index:step_index + src_len, :])
            bvp_long, hr_long = self._load_label(now_path, step_index, src_len)
            map_aug = self.transform(Image.fromarray(np.uint8(crop_long)))
            x_src = np.linspace(0, 1, src_len)
            x_new = np.linspace(0, 1, self.frames_num)
            bvp_aug = np.interp(x_new, x_src, bvp_long).astype(np.float32)
            hr_aug = np.float32(hr_long * src_len / self.frames_num)
            return (map_aug, bvp_aug, hr_aug,
                    map_orig, bvp_orig, np.float32(hr_orig),
                    now_path)
        elif src_len < self.frames_num:
            # Short crop → Resize stretches width from src_len → frames_num (slower HR)
            crop_short = self._row_normalize(img[:, step_index:step_index + src_len, :])
            bvp_short, hr_short = self._load_label(now_path, step_index, src_len)
            map_aug = self.transform(Image.fromarray(np.uint8(crop_short)))
            x_src = np.linspace(0, 1, src_len)
            x_new = np.linspace(0, 1, self.frames_num)
            bvp_aug = np.interp(x_new, x_src, bvp_short).astype(np.float32)
            hr_aug = np.float32(hr_short * src_len / self.frames_num)
            return (map_aug, bvp_aug, hr_aug,
                    map_orig, bvp_orig, np.float32(hr_orig),
                    now_path)
        else:
            return (map_orig, bvp_orig, np.float32(hr_orig),
                    map_orig, bvp_orig, np.float32(hr_orig),
                    now_path)


def build_fold_index(data_root, subjects, index_dir, stmap_name, frames_num, step=10):
    """Clear and rebuild STMap index for a subject subset."""
    os.makedirs(index_dir, exist_ok=True)
    for fname in os.listdir(index_dir):
        fpath = os.path.join(index_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
    MyDataset.getIndex(data_root, subjects, index_dir, stmap_name, step, frames_num)



def train_and_eval_fold(fold_idx, train_subjects, test_subjects,
                        data_root, data_name, args, device, log):
    """Train BaseNet on train_subjects, evaluate on test_subjects."""
    cv_base = os.path.join(config.STMAP_INDEX_BASE, f'cv_{args.dataset}', f'fold{fold_idx}')
    train_idx_dir = os.path.join(cv_base, 'train')
    test_idx_dir = os.path.join(cv_base, 'test')

    log.write(f"\n[Fold {fold_idx}] Building indexes...\n")
    build_fold_index(data_root, train_subjects, train_idx_dir, args.stmap_name, args.frames_num)
    build_fold_index(data_root, test_subjects, test_idx_dir, args.stmap_name, args.frames_num)

    if args.hr_aug_max > 1.0:
        exclude_aug = {p.strip() for p in args.hr_aug_exclude.split(',') if p.strip()}
        train_db = HRAugDataset(
            index_dir=train_idx_dir, data_name=data_name,
            stmap_name=args.stmap_name, frames_num=args.frames_num, args=args,
            hr_aug_max=args.hr_aug_max, hr_aug_min=args.hr_aug_min,
            hr_aug_prob=args.hr_aug_prob,
            exclude_aug_participants=exclude_aug,
        )
        sides = 'both ends' if args.hr_aug_min < 0.8 else 'high end only'
        log.write(f"  HR augmentation ({sides}): r_min={args.hr_aug_min}, r_max={args.hr_aug_max}, "
                  f"prob={args.hr_aug_prob}, excluded from aug={exclude_aug}\n")
        if fold_idx == 0:
            aug_stats = train_db.check_augmentation(n_samples=200)
            msg = (
                f"  [AUG DIAGNOSTIC fold 0] checked={aug_stats['checked']} samples | "
                f"can_aug_participant={aug_stats['can_aug_participant']} | "
                f"eligible(max_r>1.2)={aug_stats['eligible_max_r_gt_1.2']} | "
                f"actually_augmented={aug_stats['actually_augmented']} "
                f"({aug_stats['aug_pct']:.1f}%) | "
                f"mean_max_r={aug_stats['mean_max_r']:.3f} | "
                f"mean_W_full={aug_stats['mean_W_full']:.0f}\n"
                f"  r_sample={aug_stats['r_sample']}\n"
            )
            log.write(msg)
            print(msg, end='')
    else:
        train_db = MyDataset.Data_DG(
            root_dir=train_idx_dir, dataName=data_name,
            STMap=args.stmap_name, frames_num=args.frames_num, args=args
        )
    test_db = MyDataset.Data_DG(
        root_dir=test_idx_dir, dataName=data_name,
        STMap=args.stmap_name, frames_num=args.frames_num, args=args
    )
    log.write(f"  train: {len(train_subjects)} subjects, {len(train_db)} samples\n")
    log.write(f"  test:  {len(test_subjects)} subjects, {len(test_db)} samples\n")

    _wif = lambda wid: worker_init_fn(wid, args.seed)
    _gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_db, batch_size=args.batchsize, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=_wif, generator=_gen
    )
    test_loader = DataLoader(
        test_db, batch_size=args.batchsize, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=_wif
    )

    # Fresh model for every fold
    net = model.BaseNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_func_NP = MyLoss.P_loss3().to(device)
    loss_func_L1 = nn.L1Loss().to(device)
    loss_func_NEST_CM = MyLoss.NEST_CM().to(device)
    loss_func_NEST_DM = MyLoss.NEST_DM().to(device)
    loss_func_NEST_TA = MyLoss.NEST_TA(device, Num_ref=8).to(device)

    # ---- Training loop ----
    net.train()
    start = timer()
    src_iter = iter(train_loader)
    steps_per_epoch = len(train_loader)

    # Loss history for visualization
    loss_history = {'iter': [], 'total': [], 'src': [], 'CM': [], 'DM': [], 'TA': []}
    val_history  = {'iter': [], 'val_src': []}
    nan_skip_count = 0
    val_interval = 200

    for iter_num in range(args.max_iter + 1):
        if iter_num > 0 and iter_num % steps_per_epoch == 0:
            src_iter = iter(train_loader)

        data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug, _ = next(src_iter)
        data = Variable(data).float().to(device)
        bvp = Variable(bvp).float().to(device).unsqueeze(dim=1)
        HR_rel = Variable(torch.Tensor(HR_rel)).float().to(device)
        data_aug = Variable(data_aug).float().to(device)
        bvp_aug = Variable(bvp_aug).float().to(device).unsqueeze(dim=1)
        HR_rel_aug = Variable(torch.Tensor(HR_rel_aug)).float().to(device)

        optimizer.zero_grad()
        bvp_pre, HR_pr, av = net(data)
        bvp_pre_aug, HR_pr_aug, av_aug = net(data_aug)

        src_loss = MyLoss.get_loss(
            bvp_pre, HR_pr, bvp, HR_rel, data_name,
            loss_func_NP, loss_func_L1, args, iter_num
        )
        loss_CM = -loss_func_NEST_CM(torch.cat((av, av_aug), dim=0))
        loss_DM = loss_func_NEST_DM(av, av_aug)
        loss_TA = loss_func_NEST_TA(
            torch.cat((av, av_aug), dim=0),
            torch.cat((HR_rel, HR_rel_aug), dim=0)
        )

        weight_cl = float(args.weight_cl)
        if weight_cl == 0.0:
            loss = src_loss
        else:
            loss = src_loss + weight_cl * (loss_CM + loss_DM + loss_TA)

        if torch.sum(torch.isnan(loss)) > 0:
            nan_skip_count += 1
            log.write(f"  [fold {fold_idx}] NaN at iter {iter_num} — skipping batch ({nan_skip_count} total skips).\n")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()

        # Record every iteration for smooth curves
        loss_history['iter'].append(iter_num)
        loss_history['total'].append(loss.item())
        loss_history['src'].append(src_loss.item())
        loss_history['CM'].append(loss_CM.item())
        loss_history['DM'].append(loss_DM.item())
        loss_history['TA'].append(loss_TA.item())

        # Validation loss every val_interval iters
        if iter_num % val_interval == 0:
            net.eval()
            val_losses = []
            with torch.no_grad():
                for val_data, val_bvp, val_HR, _, _, _, _ in test_loader:
                    val_data = val_data.float().to(device)
                    val_bvp  = val_bvp.float().to(device).unsqueeze(dim=1)
                    val_HR   = torch.Tensor(val_HR).float().to(device)
                    val_bvp_pre, val_HR_pr, _ = net(val_data)
                    vl = MyLoss.get_loss(
                        val_bvp_pre, val_HR_pr, val_bvp, val_HR, data_name,
                        loss_func_NP, loss_func_L1, args, iter_num
                    )
                    if not torch.isnan(vl):
                        val_losses.append(vl.item())
            net.train()
            val_src = float(np.mean(val_losses)) if val_losses else float('nan')
            val_history['iter'].append(iter_num)
            val_history['val_src'].append(val_src)

            log_line = (
                f"  [fold {fold_idx}] iter {iter_num}/{args.max_iter}"
                f" | train={loss.item():.4f} src={src_loss.item():.4f}"
                f" | val={val_src:.4f}"
                f" | CM={loss_CM.item():.4f} DM={loss_DM.item():.4f} TA={loss_TA.item():.4f}"
                f" | {time_to_str(timer() - start, 'min')}\n"
            )
            log.write(log_line)

    if nan_skip_count > 0:
        log.write(f"  [fold {fold_idx}] Total NaN batches skipped: {nan_skip_count}\n")

    # ---- Training curve plot ----
    curve_dir = os.path.join(config.WAVE_SORT_ROOT, f'cv_{args.dataset}',
                             f'cv_{args.dataset}_wl{args.weight_cl}_fold{fold_idx}', 'feature')
    os.makedirs(curve_dir, exist_ok=True)
    if loss_history['iter']:
        iters = loss_history['iter']
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(iters, loss_history['src'], label='Train (BVP src)', linewidth=1.5)
        if val_history['iter']:
            axes[0].plot(val_history['iter'], val_history['val_src'],
                         label='Val (BVP src)', linewidth=1.5, linestyle='--', marker='o', markersize=3)
        axes[0].set_ylabel('BVP Signal Loss')
        axes[0].set_title(f'Fold {fold_idx} — Train vs Val loss | dataset={args.dataset} weight_cl={args.weight_cl}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(iters, loss_history['CM'], label='CM', linewidth=1.2, alpha=0.8)
        axes[1].plot(iters, loss_history['DM'], label='DM', linewidth=1.2, alpha=0.8)
        axes[1].plot(iters, loss_history['TA'], label='TA', linewidth=1.2, alpha=0.8)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('NEST component loss')
        axes[1].set_title('NEST contrastive components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        curve_path = os.path.join(curve_dir, 'training_curve.png')
        fig.savefig(curve_path, dpi=120)
        plt.close(fig)
        log.write(f"  Saved training curve: {curve_path}\n")

        # Also save raw loss history as CSV for later analysis
        csv_curve_path = os.path.join(curve_dir, 'training_loss.csv')
        val_by_iter = dict(zip(val_history['iter'], val_history['val_src']))
        with open(csv_curve_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['iter', 'total', 'src', 'val_src', 'CM', 'DM', 'TA'])
            w.writeheader()
            for i, it in enumerate(iters):
                w.writerow({'iter': it, 'total': loss_history['total'][i],
                            'src': loss_history['src'][i],
                            'val_src': val_by_iter.get(it, ''),
                            'CM': loss_history['CM'][i],
                            'DM': loss_history['DM'][i], 'TA': loss_history['TA'][i]})

    # ---- Inference ----
    net.eval()
    BVP_ALL, BVP_PR_ALL = [], []
    with torch.no_grad():
        for data, bvp, _, _, _, _, _ in test_loader:
            data = Variable(data).float().to(device)
            bvp = Variable(bvp).float().to(device).unsqueeze(dim=1)
            Wave_pr, _, _ = net(data)
            BVP_ALL.extend(bvp.cpu().numpy())
            BVP_PR_ALL.extend(Wave_pr.cpu().numpy())

    # Save wave_sort grouped by subject (matches intertest eval pipeline)
    run_name = f'cv_{args.dataset}_wl{args.weight_cl}_fold{fold_idx}'
    wave_sort_out = os.path.join(config.WAVE_SORT_ROOT, f'cv_{args.dataset}', run_name)
    # Clear stale files from previous runs before writing
    if os.path.isdir(wave_sort_out):
        shutil.rmtree(wave_sort_out)
    os.makedirs(wave_sort_out, exist_ok=True)
    train_utils.wave_sort_from_index(
        test_idx_dir, np.array(BVP_ALL), np.array(BVP_PR_ALL), wave_sort_out
    )

    # Save run_meta and eval_result.json so collect_results.py can pick them up
    meta = {
        'source_domain': args.dataset,
        'target_domain': args.dataset,
        'weight_cl': args.weight_cl,
        'fold': fold_idx,
        'run_name': run_name,
        'train_subjects': train_subjects,
        'test_subjects': test_subjects,
    }
    with open(os.path.join(wave_sort_out, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(net, os.path.join(config.MODEL_DIR, run_name))

    # Evaluate using FFT on BVP predictions — same pipeline as eval_from_bvp.py
    result = run_eval(wave_sort_out)
    hr_metrics = result['HR']

    # Save eval_result.json so collect_results.py can aggregate intratest alongside intertest
    eval_payload = {
        'source_domain': args.dataset,
        'target_domain': args.dataset,
        'weight_cl': args.weight_cl,
        'fold': fold_idx,
        'result': result,
    }
    with open(os.path.join(wave_sort_out, 'eval_result.json'), 'w') as f:
        json.dump(eval_payload, f, indent=2)

    metrics = {
        'ME': hr_metrics['ME'], 'Std': hr_metrics['Std'],
        'MAE': hr_metrics['MAE'], 'RMSE': hr_metrics['RMSE'],
        'MER': hr_metrics['MER'], 'r': hr_metrics['r'],
        'fold': fold_idx,
        'weight_cl': args.weight_cl,
        'train_subjects': len(train_subjects),
        'test_subjects': len(test_subjects),
        'train_samples': len(train_db),
        'test_samples': len(test_db),
        'wave_sort': os.path.abspath(wave_sort_out),
    }
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset not in config.FILEA_NAME:
        raise ValueError(f"'{args.dataset}' not in config.FILEA_NAME")

    data_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[args.dataset][0])
    data_name = config.canonical_data_name(args.dataset)

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    subjects = sorted([s for s in os.listdir(data_root) if not s.startswith('.')])

    excluded = {p.strip() for p in args.exclude_participants.split(',') if p.strip()}
    if excluded:
        before = len(subjects)
        subjects = [s for s in subjects if get_participant_id(s) not in excluded]
        print(f"  Excluded participants {excluded}: {before} -> {len(subjects)} subjects")

    if args.session_filter:
        before = len(subjects)
        subjects = [s for s in subjects if args.session_filter in s]
        print(f"  Session filter '{args.session_filter}': {before} -> {len(subjects)} subjects")

    device = torch.device('cuda:' + args.GPU if torch.cuda.is_available() else 'cpu')

    os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.RESULT_LOG_DIR, f'intratest_cv_{args.dataset}.txt')
    log = Logger()
    log.open(log_path, mode='a')

    print('=' * 60)
    print(f'Intratest {args.folds}-fold CV: {args.dataset}')
    print(f'  data_root     : {data_root}')
    print(f'  data_name     : {data_name}')
    print(f'  total subjects: {len(subjects)}')
    print(f'  device        : {device}')
    print(f'  max_iter      : {args.max_iter}')
    print(f'  batch_size    : {args.batchsize}')
    print(f'  loss_type     : {args.loss_type}')
    print(f'  seed          : {args.seed}')
    print('=' * 60)

    log.write(f"\n{'='*60}\n")
    log.write(f"Intratest {args.folds}-fold CV: {args.dataset}\n")
    log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"Subjects: {len(subjects)} | max_iter: {args.max_iter} | batch: {args.batchsize}\n")
    log.write(f"loss_type: {args.loss_type} | seed: {args.seed} | device: {device}\n")

    all_metrics = []
    for fold_idx in range(args.folds):
        train_subj, test_subj = subject_kfold_split(subjects, args.folds, fold_idx)
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args.folds}: "
              f"train={len(train_subj)} subjects, test={len(test_subj)} subjects")
        print(f"  Test subjects: {test_subj}")

        metrics = train_and_eval_fold(
            fold_idx, train_subj, test_subj,
            data_root, data_name, args, device, log
        )
        metrics['dataset'] = args.dataset
        all_metrics.append(metrics)

        log.write(
            f"[Fold {fold_idx}] ME={metrics['ME']:.4f} Std={metrics['Std']:.4f} "
            f"MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
            f"MER={metrics['MER']:.6f} r={metrics['r']:.4f}\n"
        )
        print(f"  -> MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
              f"ME={metrics['ME']:.4f} r={metrics['r']:.4f}")

    # Aggregate
    print(f"\n{'='*60}")
    print(f"AGGREGATE ({args.folds}-fold CV | {args.dataset})")
    print(f"{'Metric':<8} {'Mean':>10} {'Std (across folds)':>20}")
    print('-' * 40)
    summary = {}
    for metric in ['ME', 'Std', 'MAE', 'RMSE', 'MER', 'r']:
        vals = [m[metric] for m in all_metrics]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        summary[metric] = {'mean': mean_val, 'fold_std': std_val, 'per_fold': vals}
        print(f"{metric:<8} {mean_val:>10.4f} {std_val:>20.4f}")

    log.write(f"\n{'='*60}\n")
    log.write(f"AGGREGATE ({args.folds}-fold CV | {args.dataset}):\n")
    for metric, s in summary.items():
        log.write(f"  {metric}: mean={s['mean']:.4f}  fold_std={s['fold_std']:.4f}"
                  f"  per_fold={[round(v, 4) for v in s['per_fold']]}\n")

    summary_path = os.path.join(
        config.RESULT_LOG_DIR, f'intratest_cv_{args.dataset}_wl{args.weight_cl}_summary.json'
    )
    with open(summary_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'folds': args.folds,
            'weight_cl': args.weight_cl,
            'n_subjects': len(subjects),
            'args': vars(args),
            'fold_metrics': all_metrics,
            'summary': summary,
        }, f, indent=2)
    print(f"\nSummary JSON saved: {summary_path}")
    log.write(f"\nSummary JSON: {summary_path}\n")

    # CSV: one row per fold + one aggregate row
    tag = f'_{args.run_tag}' if args.run_tag else ''
    csv_path = os.path.join(
        config.RESULT_LOG_DIR, f'intratest_cv_{args.dataset}_wl{args.weight_cl}{tag}_results.csv'
    )
    fieldnames = ['dataset', 'weight_cl', 'fold', 'train_subjects', 'test_subjects',
                  'ME', 'Std', 'MAE', 'RMSE', 'MER', 'r']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m.get(k, '') for k in fieldnames})
        # Aggregate row
        writer.writerow({
            'dataset': args.dataset,
            'weight_cl': args.weight_cl,
            'fold': 'mean',
            'train_subjects': '',
            'test_subjects': '',
            **{metric: round(summary[metric]['mean'], 4) for metric in ['ME', 'Std', 'MAE', 'RMSE', 'MER', 'r']},
        })
        writer.writerow({
            'dataset': args.dataset,
            'weight_cl': args.weight_cl,
            'fold': 'fold_std',
            'train_subjects': '',
            'test_subjects': '',
            **{metric: round(summary[metric]['fold_std'], 4) for metric in ['ME', 'Std', 'MAE', 'RMSE', 'MER', 'r']},
        })
    print(f"Results CSV saved: {csv_path}")
    log.write(f"Results CSV: {csv_path}\n")


if __name__ == '__main__':
    main()
