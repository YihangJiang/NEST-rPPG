#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Region-aware & HR-augmented training script (train_aug.py).
Trains on source domain(s) with optional HR temporal resampling augmentation
(hr_aug_max / hr_aug_min) and region-aware InfoNCE alignment (weight_info),
then evaluates on target domain and logs metrics to MLflow.

Run as script:
    python train_aug.py --src 'PURE_my_in' -t 'UBFC_my_in' --regions all --hr_aug_max 2.0 --run_tag aug
"""

import os
import json
import random
from types import SimpleNamespace
from datetime import datetime
from timeit import default_timer as timer

import cv2
from PIL import Image
import numpy as np
import scipy.io as io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import MyDataset
import MyLoss
import model
from utils.core import Logger, time_to_str, get_args
import utils.train_utils as train_utils
import utils.mlflow_utils as mlflow_utils
import config


def _worker_init_fn(worker_id):
    seed = (config.SEED + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_participant_id(folder_name):
    """Extract unique participant ID from a session folder name."""
    if '-' in folder_name and not folder_name.startswith('Sub_'):
        return folder_name.split('-')[0]
    if 'lux' in folder_name:
        return folder_name.split('lux')[0]
    return folder_name


class HRAugDataset(Dataset):
    """Training dataset with temporal compression/stretch to simulate higher/lower HR.

    Loads a window of src_len = round(frames_num * r) frames and lets
    transforms.Resize((64, frames_num)) compress/stretch it to frames_num columns.
    The apparent HR scales by r = src_len / frames_num.
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
            z = np.zeros((3, 64, self.frames_num), dtype=np.float32)
            zb = np.zeros(self.frames_num, dtype=np.float32)
            zt = torch.from_numpy(z)
            return (zt, zb, np.float32(0.0), zt, zb, np.float32(0.0), now_path)

        _, W_full, _ = img.shape
        folder = os.path.basename(now_path)
        participant = get_participant_id(folder)
        can_aug = participant not in self.exclude_aug

        can_go_high = self.hr_aug_max > 1.2
        can_go_low = self.hr_aug_min < 0.8

        if can_aug and random.random() < self.hr_aug_prob:
            if can_go_high and can_go_low:
                if random.random() < 0.5:
                    max_r = min(self.hr_aug_max, (W_full - step_index) / self.frames_num)
                    r = random.uniform(1.2, max_r) if max_r > 1.2 else 1.0
                else:
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

        crop_orig = self._row_normalize(img[:, step_index:step_index + self.frames_num, :])
        bvp_orig, hr_orig = self._load_label(now_path, step_index, self.frames_num)
        map_orig = self.transform(Image.fromarray(np.uint8(crop_orig)))

        if src_len != self.frames_num:
            crop_aug = self._row_normalize(img[:, step_index:step_index + src_len, :])
            bvp_long, hr_long = self._load_label(now_path, step_index, src_len)
            map_aug = self.transform(Image.fromarray(np.uint8(crop_aug)))
            x_src = np.linspace(0, 1, src_len)
            x_new = np.linspace(0, 1, self.frames_num)
            bvp_aug = np.interp(x_new, x_src, bvp_long).astype(np.float32)
            hr_aug = np.float32(hr_long * src_len / self.frames_num)
            return (map_aug, bvp_aug, hr_aug,
                    map_orig, bvp_orig, np.float32(hr_orig),
                    now_path)
        else:
            return (map_orig, bvp_orig, np.float32(hr_orig),
                    map_orig, bvp_orig, np.float32(hr_orig),
                    now_path)


def main():
    args = get_args()
    if not hasattr(args, 'frames_num'):
        args.frames_num = 512
    if not hasattr(args, 'grad_clip'):
        args.grad_clip = 5.0
    seed = getattr(args, 'seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    GPU = getattr(args, 'GPU', '0')
    batch_size = getattr(args, 'batchsize', 100)
    num_workers = getattr(args, 'num_workers', 2)
    frames_num = args.frames_num
    stmap_name = getattr(args, 'stmap_name', config.STMAP_NAME)
    lr = getattr(args, 'lr', 0.001)

    src_arg = getattr(args, 'src', None) or getattr(args, 'source_domain', config.SRC_DOMAIN)
    tgt_arg = getattr(args, 'tgt', None) or getattr(args, 'target', None) or getattr(args, 'target_domain', config.TGT_DOMAIN)

    source_domain = src_arg
    tgt_domain = tgt_arg

    if "_" in source_domain:
        base_src = source_domain.rsplit("_", 1)[0]
    else:
        base_src = source_domain

    pos_domain = f"{base_src}_rm"
    neg_domain = f"{base_src}_eye"
    region_domains = [pos_domain, source_domain, neg_domain]

    index_root = getattr(args, 'index_root', config.STMAP_INDEX_BASE)
    region_roots = {
        d: os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[d][0])
        for d in region_domains
    }
    region_index_dirs = {d: os.path.join(index_root, d) for d in region_domains}
    source_index_dir = region_index_dirs[source_domain]

    target_root = os.path.join(config.STMAP_PARENT_ROOT, config.FILEA_NAME[tgt_domain][0])
    target_index_dir = os.path.join(index_root, tgt_domain)

    os.makedirs(index_root, exist_ok=True)

    def _build_index_if_needed(root_dir, index_dir, stmap_n, label):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"{label} root not found: {root_dir}")
        os.makedirs(index_dir, exist_ok=True)
        if len(os.listdir(index_dir)) > 0:
            print(f"  Index for {label} ready ({len(os.listdir(index_dir))} files).")
            return
        print(f"Building index for {label}:")
        files_list = sorted([f for f in os.listdir(root_dir) if not f.startswith('.')])
        MyDataset.getIndex(root_dir, files_list, index_dir, stmap_n, 10, frames_num)

    for d in region_domains:
        _build_index_if_needed(region_roots[d], region_index_dirs[d], stmap_name, d)

    _build_index_if_needed(target_root, target_index_dir, stmap_name, tgt_domain)

    hr_max = float(getattr(args, 'hr_aug_max', 1.0))
    hr_min = float(getattr(args, 'hr_aug_min', 1.0))
    hr_prob = float(getattr(args, 'hr_aug_prob', 1.0))
    exclude_list = [x.strip() for x in getattr(args, 'hr_aug_exclude', '').split(',') if x.strip()]

    use_hr_aug = (hr_max > 1.0 or hr_min < 1.0)
    print(f"Dataset HR Augmentation enabled: {use_hr_aug} (hr_aug_max={hr_max}, hr_aug_min={hr_min}, prob={hr_prob})")

    def make_src_db(idx_dir, d_name):
        if use_hr_aug:
            return HRAugDataset(
                index_dir=idx_dir,
                data_name=config.canonical_data_name(d_name),
                stmap_name=stmap_name,
                frames_num=frames_num,
                args=args,
                hr_aug_max=hr_max,
                hr_aug_min=hr_min,
                hr_aug_prob=hr_prob,
                exclude_aug_participants=exclude_list,
            )
        else:
            return MyDataset.Data_DG(
                root_dir=idx_dir,
                dataName=config.canonical_data_name(d_name),
                STMap=stmap_name,
                frames_num=frames_num,
                args=args,
            )

    source_db = make_src_db(source_index_dir, source_domain)
    pos_db = make_src_db(region_index_dirs[pos_domain], pos_domain)
    neg_db = make_src_db(region_index_dirs[neg_domain], neg_domain)

    target_db = MyDataset.Data_DG(
        root_dir=target_index_dir,
        dataName=config.canonical_data_name(tgt_domain),
        STMap=stmap_name,
        frames_num=frames_num,
        args=args,
    )

    print("Creating DataLoaders...")
    _generator = torch.Generator().manual_seed(seed)
    _dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=_worker_init_fn)

    src_loader = DataLoader(source_db, shuffle=True, generator=_generator, **_dl_kwargs)
    tgt_loader = DataLoader(target_db, shuffle=False, **_dl_kwargs)
    pos_loader = DataLoader(pos_db, shuffle=False, **_dl_kwargs)
    neg_loader = DataLoader(neg_db, shuffle=False, **_dl_kwargs)

    steps_per_epoch = len(src_loader)

    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU)
        print('Using device:', device)
    else:
        device = torch.device('cpu')
        print('Using CPU device')

    BaseNet = model.BaseNet().to(device=device)

    rPPGNet_name = config.build_run_name(
        tgt=tgt_domain,
        src=source_domain,
        weight_info=float(getattr(args, "weight_info", 0.0)),
    )
    tag = getattr(args, "run_tag", "aug")
    if tag:
        rPPGNet_name = f"{rPPGNet_name}_{tag}"

    os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.RESULT_LOG_DIR, rPPGNet_name + '_log.txt')
    log = Logger()
    log.open(log_path, mode='a')

    optimizer = torch.optim.Adam(BaseNet.parameters(), lr=lr)

    mlflow_utils.setup(
        args,
        experiment_name=getattr(args, 'mlflow_experiment', None) or 'nest-rppg-regions',
        run_name=rPPGNet_name,
        tags={'script': 'train_aug', 'source_domain': source_domain, 'target_domain': tgt_domain},
    )
    mlflow_utils.log_params({
        'source_domain': source_domain,
        'target_domain': tgt_domain,
        'pos_domain': pos_domain,
        'neg_domain': neg_domain,
        'weight_info': float(getattr(args, 'weight_info', 0.0)),
        'tau_info': float(getattr(args, 'tau_info', 0.07)),
        'regions': str(getattr(args, 'regions', 'all')),
        'loss_type': getattr(args, 'loss_type', config.LOSS_TYPE),
        'hr_aug_max': hr_max,
        'hr_aug_min': hr_min,
        'hr_aug_prob': hr_prob,
        'run_tag': str(getattr(args, 'run_tag', 'aug')),
        'lr': lr,
        'batchsize': batch_size,
        'max_iter': args.max_iter,
        'frames_num': frames_num,
        'seed': seed,
    })

    loss_func_NP = MyLoss.P_loss3().to(device)
    loss_func_L1 = nn.L1Loss().to(device)
    loss_func_NEST_CM = MyLoss.NEST_CM().to(device)
    loss_func_NEST_DM = MyLoss.NEST_DM().to(device)
    loss_func_NEST_TA = MyLoss.NEST_TA(device, Num_ref=8).to(device)

    BaseNet.train()
    start = timer()
    max_iter = args.max_iter

    src_iter = iter(src_loader)
    pos_iter = iter(pos_loader)
    neg_iter = iter(neg_loader)

    for iter_num in range(max_iter + 1):
        if iter_num > 0 and (iter_num % steps_per_epoch == 0):
            src_iter = iter(src_loader)
            pos_iter = iter(pos_loader)
            neg_iter = iter(neg_loader)

        try:
            data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug, subj_paths = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            data, bvp, HR_rel, data_aug, bvp_aug, HR_rel_aug, subj_paths = next(src_iter)

        data = Variable(data).float().to(device=device)
        bvp = Variable(bvp).float().to(device=device).unsqueeze(dim=1)
        HR_rel = Variable(torch.Tensor(HR_rel)).float().to(device=device)
        data_aug = Variable(data_aug).float().to(device=device)
        bvp_aug = Variable(bvp_aug).float().to(device=device).unsqueeze(dim=1)
        HR_rel_aug = Variable(torch.Tensor(HR_rel_aug)).float().to(device=device)

        weight_info = float(getattr(args, "weight_info", 0.0))
        use_align = weight_info > 0.0
        align_pos_loss = torch.tensor(0.0, device=device)

        if use_align:
            try:
                pos_data, _, _, _, _, _, _ = next(pos_iter)
            except StopIteration:
                pos_iter = iter(pos_loader)
                pos_data, _, _, _, _, _, _ = next(pos_iter)
            try:
                neg_data, _, _, _, _, _, _ = next(neg_iter)
            except StopIteration:
                neg_iter = iter(neg_loader)
                neg_data, _, _, _, _, _, _ = next(neg_iter)

            pos_data = Variable(pos_data).float().to(device=device)
            neg_data = Variable(neg_data).float().to(device=device)

            BaseNet.eval()
            with torch.no_grad():
                _, _, av_pos = BaseNet(pos_data)
                _, _, av_neg = BaseNet(neg_data)
            BaseNet.train()

        optimizer.zero_grad()
        predict_bvp, p_HR, av_src = BaseNet(data)
        predict_bvp_aug, p_HR_aug, av_aug = BaseNet(data_aug)

        src_loss = MyLoss.get_loss(
            predict_bvp, p_HR, bvp, HR_rel, source_domain,
            loss_func_NP, loss_func_L1, args, iter_num
        )

        loss_CM = -loss_func_NEST_CM(torch.cat((av_src, av_aug), dim=0))
        loss_DM = loss_func_NEST_DM(av_src, av_aug)
        loss_TA = loss_func_NEST_TA(
            torch.cat((av_src, av_aug), dim=0),
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

        loss_terms = {
            'CM': loss_CM.item() if isinstance(loss_CM, torch.Tensor) else float(loss_CM),
            'DM': loss_DM.item() if isinstance(loss_DM, torch.Tensor) else float(loss_DM),
            'TA': loss_TA.item() if isinstance(loss_TA, torch.Tensor) else float(loss_TA),
        }

        if use_align and (av_pos is not None) and (av_neg is not None):
            m = av_src.shape[0]
            tau = float(getattr(args, 'tau_info', 0.05))
            q = F.normalize(av_src, dim=1)
            k_pos = F.normalize(av_pos[:m], dim=1)
            k_neg = F.normalize(av_neg[:m], dim=1)
            neg_scores = torch.mm(q, k_neg.t()) / tau
            pos_scores = torch.mm(q, k_pos.t()) / tau
            regions_mode = str(getattr(args, 'regions', 'all')).lower()
            if regions_mode == "neg":
                logits = torch.cat([neg_scores], dim=1)
            elif regions_mode == "pos":
                logits = torch.cat([pos_scores], dim=1)
            else:
                logits = torch.cat([pos_scores, neg_scores], dim=1)
            labels = torch.zeros(m, dtype=torch.long, device=logits.device)
            align_pos_loss = F.cross_entropy(logits, labels)
            loss = loss + weight_info * align_pos_loss

        if torch.sum(torch.isnan(loss)) > 0:
            print('Nan')
            break
        loss.backward()
        # Prevent exploding gradients -> NaNs in decoder output (bvp_pre)
        torch.nn.utils.clip_grad_norm_(BaseNet.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()

        if iter_num % 100 == 0:
            msg = (
                f"Train Inter:{iter_num} | loss: {loss.item():.7f} | "
                f"CM: {loss_terms.get('CM', 0):.7f} | DM: {loss_terms.get('DM', 0):.7f} | "
                f"TA: {loss_terms.get('TA', 0):.7f} | AlignPos: {align_pos_loss.item():.7f} | "
                f"{time_to_str(timer() - start, 'min')}"
            )
            print(msg)
            log.write(msg + "\n")
            mlflow_utils.log_metrics({
                'loss': loss.item(),
                'src_loss': float(loss_terms.get('CM', 0) + loss_terms.get('DM', 0) + loss_terms.get('TA', 0)),
                'loss_CM': float(loss_terms.get('CM', 0)),
                'loss_DM': float(loss_terms.get('DM', 0)),
                'loss_TA': float(loss_terms.get('TA', 0)),
                'align_pos_loss': align_pos_loss.item(),
                'elapsed_min': float((timer() - start) / 60.0),
            }, step=iter_num)

    print("Training finished. Inferencing on target domain...")
    BaseNet.eval()

    HR_pr_list = []
    HR_rel_list = []
    BVP_ALL = []
    BVP_PR_ALL = []
    nan_debug_saved = False

    with torch.no_grad():
        for step, (data, bvp, HR_rel, _, _, _, paths) in enumerate(tgt_loader):
            print(step)
            data = data.float().to(device=device)
            bvp = bvp.float().to(device=device)
            HR_rel = HR_rel.float().to(device=device)
            bvp = bvp.unsqueeze(dim=1)

            Wave = bvp
            Wave_pr, HR_pr, av = BaseNet(data)

            # Debug: check decoder signal output stability during inference.
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

    wave_sort_out = os.path.join(config.WAVE_SORT_ROOT, tgt_domain, rPPGNet_name)
    train_utils.wave_sort_from_index(
        target_index_dir, np.array(BVP_ALL), np.array(BVP_PR_ALL), wave_sort_out
    )

    # Write last Wave_sort path so eval_from_bvp can follow the latest run.
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
        os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "source_domain": source_domain,
                    "target_domain": tgt_domain,
                    "weight_info": float(getattr(args, "weight_info", 0.0)),
                    "tau_info": float(getattr(args, "tau_info", 0.07)),
                    "loss_type": getattr(args, "loss_type", config.LOSS_TYPE),
                    "regions": str(getattr(args, "regions", "all")),
                    "hr_aug_max": hr_max,
                    "hr_aug_min": hr_min,
                },
                f,
                indent=2,
            )
        print("Saved train_aug meta to:", meta_path)
    except Exception as e:
        print("Warning: failed to write last_train_regions_meta.json:", repr(e))

    mlflow_utils.log_model(BaseNet)

    # Run evaluation and log eval metrics (MAE, RMSE, ME) to MLflow
    try:
        from utils.eval_utils import (
            run_eval,
            write_segment_errors_csv,
            print_hr_metrics,
        )

        eval_save_path = os.path.abspath(wave_sort_out)
        print(f"\nEvaluating Wave_sort path: {eval_save_path}")

        result, details = run_eval(eval_save_path, return_details=True)

        feature_dir = os.path.join(eval_save_path, "feature")
        os.makedirs(feature_dir, exist_ok=True)
        csv_path = os.path.join(feature_dir, "segment_errors.csv")
        write_segment_errors_csv(details, csv_path)

        hr_metrics = result.get("HR", {})
        eval_mae = float(hr_metrics.get("MAE", float("nan")))
        eval_rmse = float(hr_metrics.get("RMSE", float("nan")))
        eval_me = float(hr_metrics.get("ME", float("nan")))

        print(f"Evaluation Results -> MAE: {eval_mae:.4f} | RMSE: {eval_rmse:.4f} | ME: {eval_me:.4f}")

        mlflow_utils.log_metrics({
            "eval_MAE": eval_mae,
            "eval_RMSE": eval_rmse,
            "eval_ME": eval_me,
        })

        print_hr_metrics(result, source_domain=source_domain, target_domain=tgt_domain)
    except Exception as exc:
        print(f"Warning: automatic evaluation / MLflow logging failed: {exc}")

    mlflow_utils.log_artifacts([log_path, meta_path])
    mlflow_utils.end_run()


if __name__ == '__main__':
    main()
