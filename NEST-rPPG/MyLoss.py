""" Utilities """
import os
import shutil
import torch

import torch.nn as nn
from torch.autograd import Function
import numpy as np
import argparse
from torch.autograd import Variable
from numpy import random
import math


class P_loss3(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, gt_lable, pre_lable):
        M, N, A = gt_lable.shape
        gt_lable = gt_lable - torch.mean(gt_lable, dim=2).view(M, N, 1)
        pre_lable = pre_lable - torch.mean(pre_lable, dim=2).view(M, N, 1)
        aPow = torch.sqrt(torch.sum(torch.mul(gt_lable, gt_lable), dim=2))
        bPow = torch.sqrt(torch.sum(torch.mul(pre_lable, pre_lable), dim=2))
        pearson = torch.sum(torch.mul(gt_lable, pre_lable), dim=2) / (aPow * bPow + 0.001)
        loss = 1 - torch.sum(torch.sum(pearson, dim=1), dim=0)/(gt_lable.shape[0] * gt_lable.shape[1])
        return loss



class SP_loss(nn.Module):
    def __init__(self, device, clip_length=256, delta=3, loss_type=1, use_wave=False):
        super(SP_loss, self).__init__()

        self.clip_length = clip_length
        self.time_length = clip_length
        self.device = device
        self.delta = delta
        self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = 40
        self.high_bound = 150

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype = torch.float).to(self.device)
        self.bpm_range = self.bpm_range / 60.0

        self.pi = 3.14159265
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        self.two_pi_n = two_pi_n.to(self.device)
        self.hanning = hanning.to(self.device)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.l1 = nn.L1Loss()

        self.loss_type = loss_type
        self.eps = 0.0001

        self.lambda_l1 = 0.1
        self.use_wave = use_wave

    def forward(self, wave, gt, pred = None, flag = None):  # all variable operation
        fps = 30

        hr = gt.clone()

        hr[hr.ge(self.high_bound)] = self.high_bound-1
        hr[hr.le(self.low_bound)] = self.low_bound

        if pred is not None:
            pred = torch.mul(pred, fps)
            pred = pred * 60 / self.clip_length

        batch_size = wave.shape[0]

        f_t = self.bpm_range / fps
        preds = wave * self.hanning

        preds = preds.view(batch_size, 1, -1)
        f_t = f_t.repeat(batch_size, 1).view(batch_size, -1, 1)

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute = torch.sum(preds * torch.sin(f_t*tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t*tmp), dim=-1) ** 2

        target = hr - self.low_bound
        target = target.type(torch.long).view(batch_size)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound



        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            norm_t = (torch.ones(batch_size).to(self.device) / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t

            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1;

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss = loss + loss_snr

        return loss, whole_max_idx




class NEST_CM(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, av, ratio=0.1):

        # av is (batch_size, 960). We divide it into 4 blocks of channels.
        blocks = [
            ("b0", av[:, 0:64], 64),
            ("b1", av[:, 64:192], 128),
            ("b2", av[:, 192:448], 256),
            ("b3", av[:, 448:960], 512),
        ]

        cov_losses = []
        for name, block, dim in blocks:
            try:
                s = torch.linalg.svdvals(block)
            except Exception as e:
                # Print diagnostics if SVD fails on this block
                with torch.no_grad():
                    block_cpu = block.detach().float().cpu()
                    max_abs = float(block_cpu.abs().max())
                    nan_count = int(torch.isnan(block_cpu).sum())
                    # print(f"[NEST_CM] SVD failure on {name} shape={tuple(block.shape)} "
                    #       f"max|x|={max_abs:.4e} nan_count={nan_count} error={repr(e)}")
                raise

            # Condition estimate (before normalization)
            with torch.no_grad():
                s_cpu = s.detach().float().cpu()
                s_min = float(s_cpu.min())
                s_max = float(s_cpu.max())
                s_sum = float(s_cpu.sum())
                cond = s_max / (s_min + 1e-12) if s_min > 0 else float("inf")
                # print(f"[NEST_CM] {name} svdvals: shape={tuple(block.shape)} "
                #       f"sum={s_sum:.4e} min={s_min:.4e} max={s_max:.4e} cond~={cond:.4e}")

            s = torch.div(s, torch.sum(s))
            thr = ratio / float(dim)
            cov_loss = torch.sum(s[s < thr])
            cov_losses.append(cov_loss)

        cov_loss0, cov_loss1, cov_loss2, cov_loss3 = cov_losses

        return (cov_loss0 + cov_loss1 + cov_loss2 + cov_loss3)/4
    
class NEST_DM(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = 0.02
    def forward(self, av, av_aug):
        # av, av_aug Batchsize, Channel
        av_mod = torch.div(av, torch.norm(av, p=2, dim=1).unsqueeze(1))
        av_aug_mod = torch.div(av_aug, torch.norm(av_aug, p=2, dim=1).unsqueeze(1))
        mm = torch.mm(av_mod, av_aug_mod.permute(1, 0))
        diag = torch.diag(mm)
        loss = torch.mean(-torch.log(torch.div(torch.exp(diag/self.t), (torch.sum(torch.exp(mm/self.t), dim=0)))))
        return loss



class NEST_TA(nn.Module):
    def __init__(self, device, Num_ref=4, std=5):
        super().__init__()
        self.Num_ref = Num_ref
        self.Std = std

    def cos_sim(self, l1, l2):
        l1_mod = torch.div(l1, 1e-10 + torch.norm(l1, p=2, dim=1).unsqueeze(1))
        l2_mod = torch.div(l2, 1e-10 + torch.norm(l2, p=2, dim=1).unsqueeze(1))
        sim = torch.mean(torch.sum(torch.mul(l1_mod, l2_mod), dim=1))
        return sim

    def Gaussian_Smooth(self, sample, mean, std=5):
        gmm_wight = torch.exp(-torch.abs(sample - mean) ** 2 / (2 * std ** 2)) / (torch.sqrt(torch.tensor(2 * math.pi)) * std)
        gmm_wight = gmm_wight / torch.sum(gmm_wight, dim=1).unsqueeze(1)
        return gmm_wight

    def forward(self, Struct, Label):
        batch_size = Struct.shape[0]
        Ref_Index = np.tile(np.arange(0, batch_size, 1), batch_size)

        Label_ref = Label[Ref_Index].reshape(batch_size, batch_size).detach()
        Res_abs = torch.abs(Label_ref - Label.unsqueeze(1))
        _, sort_index = torch.sort(Res_abs, dim=1, descending=False)

        Struct_ref = Struct[sort_index[:, 0:self.Num_ref]]
        Label_ref = Label[sort_index[:, 0:self.Num_ref]].detach()


        gmm_wight = self.Gaussian_Smooth(Label_ref, mean=Label.unsqueeze(1), std=self.Std)


        Struct_mean = torch.sum(torch.mul(gmm_wight.unsqueeze(-1), Struct_ref), dim=1)
        Label_mean = torch.sum(torch.mul(gmm_wight, Label_ref), dim=1)

        Label_res = Label_ref - Label_mean.unsqueeze(1)
        Struct_res = Struct_ref - Struct_mean.unsqueeze(1)



        Label_d = Label.unsqueeze(1) - Label_mean.unsqueeze(1)

        ratio = (gmm_wight * torch.div(Label_d, 1e-10 + Label_res)).unsqueeze(2)

        Struct_smooth = torch.sum(ratio * Struct_res, dim=1) + Struct_mean

        sim = self.cos_sim(Struct, Struct_mean)

        return 1 - sim

def get_loss(bvp_pre, hr_pre, bvp_gt, hr_gt, dataName, \
             loss_sig0, loss_hr, args, inter_num):
    k = 2.0 / (1.0 + np.exp(-10.0 * inter_num/args.max_iter)) - 1.0

    k1, k2, k3, k4, k5, k6, k7, k8 = args.k1, k*args.k2, args.k3, k*args.k4, args.k5, k*args.k6, k*args.k7, k*args.k8

    # HR loss commented out; only signal (BVP) loss is used.
    if dataName.startswith('PURE'):
        loss = k1 * loss_sig0(bvp_pre, bvp_gt)
        # loss = (k1*loss_sig0(bvp_pre, bvp_gt) + k2*loss_hr(torch.squeeze(hr_pre), hr_gt))/2
    elif dataName.startswith('UBFC'):
        loss = k3 * loss_sig0(bvp_pre, bvp_gt)
        # loss = (k3 * loss_sig0(bvp_pre, bvp_gt) + k4 * loss_hr(torch.squeeze(hr_pre), hr_gt))/2
    elif dataName.startswith('BUAA'):
        loss = k5 * loss_sig0(bvp_pre, bvp_gt)
        # loss = (k5 * loss_sig0(bvp_pre, bvp_gt) + k6 * loss_hr(torch.squeeze(hr_pre), hr_gt))/2
    elif dataName.startswith('VIPL'):
        loss = k7 * loss_sig0(bvp_pre, bvp_gt)
        # loss = k7 * loss_hr(torch.squeeze(hr_pre), hr_gt)
    elif dataName.startswith('V4V'):
        # V4V has no BVP label (bvp_gt is placeholder); signal loss not applicable.
        loss = torch.tensor(0.0, device=bvp_pre.device, dtype=bvp_pre.dtype)
        # loss = k8 * loss_hr(torch.squeeze(hr_pre), hr_gt)
    else:
        raise ValueError(f"get_loss: unknown dataName '{dataName}'")

    if torch.sum(torch.isnan(loss)) > 0:
        print('Tere in nan loss found in' + dataName)
    return loss