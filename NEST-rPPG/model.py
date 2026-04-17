# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import utils
from torchvision import models
import numpy as np

np.set_printoptions(threshold=np.inf)
sys.path.append('..')


class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, stride=2, downsample=1, Res=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
                 )
        self.downsample = downsample
        self.Res = Res

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        return F.relu(out)


class TemporalAwareStem(nn.Module):
    """
    First layer for STMaps (H = skin rows, W = time / frames).

    Replaces the isotropic ResNet 7x7/2 with an asymmetric kernel: 3x7 with stride (2,2)
    and padding (1,3) so output H×W matches the original stem, while giving a wider
    receptive field along the temporal (frame) axis for pulse-related structure.

    Output: (B, 64, H', W') compatible with ResNet-18 layer1.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 7),
            stride=(2, 2),
            padding=(1, 3),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        # Use new torchvision API: weights instead of deprecated pretrained=True
        from torchvision.models import ResNet18_Weights
        model_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 256, [2, 1], downsample=1),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 64, [1, 1], downsample=1),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 32, [2, 1], downsample=1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(32, 1, [1, 1], downsample=1),
        )

    def get_av(self, x, eps: float = 1e-6):
        """
        Global channel descriptor with stable per-sample min-max normalization.

        Steps per sample:
        - Global average pool over spatial/temporal dims -> (C,) channel means
        - Min-max normalize across channels with an epsilon in the denominator
          to avoid division by zero when max == min.
        """
        # x: (B, C, H, T) -> av_raw: (B, C)
        av = x.mean(dim=(-1, -2))
        # Per-sample min / max across channels
        min_val, _ = torch.min(av, dim=1, keepdim=True)
        max_val, _ = torch.max(av, dim=1, keepdim=True)
        # Stable denominator
        denom = (max_val - min_val).clamp_min(eps)
        av = (av - min_val) / denom
        # Extra safety: replace any NaN/Inf with zeros
        av = torch.nan_to_num(av, nan=0.0, posinf=0.0, neginf=0.0)
        return av

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        av1 = self.get_av(x)
        x = self.layer2(x)
        av2 = self.get_av(x)
        x = self.layer3(x)
        av3 = self.get_av(x)
        em = self.layer4(x)
        av4 = self.get_av(em)

        av = torch.cat([av1, av2, av3, av4], dim=1)

        HR = self.fc(self.avgpool(em).view(x.size(0), -1))
        # For Sig
        x = self.up1(em)
        x = self.up2(x)
        x = self.up3(x)
        Sig = self.up4(x).squeeze(dim=1)

        return Sig, HR, av


def _fuse_skip(x, skip, merge: nn.Module):
    """Resize skip to decoder spatial size, concat, merge channels."""
    skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
    x = torch.cat([x, skip], dim=1)
    return merge(x)


class BaseNetSkip(nn.Module):
    """
    Same encoder / HR / av heads as BaseNet, but:
    - TemporalAwareStem (3x7, stride 2) instead of ImageNet 7x7 stem
    - BVP decoder fuses skip features from layer3, layer2, layer1 (U-Net-style)

    Forward API matches BaseNet: returns (Sig, HR, av).
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights

        model_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Temporal-aware stem (no ImageNet conv1 weights); layer1-4 still pretrained.
        self.stem = TemporalAwareStem(3, 64)
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

        # Same upsampling chain as BaseNet (bottleneck -> BVP)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 256, [2, 1], downsample=1),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 64, [1, 1], downsample=1),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 32, [2, 1], downsample=1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(32, 1, [1, 1], downsample=1),
        )

        # Merge concat(x_dec, skip) -> same channel count as x_dec before concat
        self.merge1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.merge2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.merge3 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def get_av(self, x, eps: float = 1e-6):
        av = x.mean(dim=(-1, -2))
        min_val, _ = torch.min(av, dim=1, keepdim=True)
        max_val, _ = torch.max(av, dim=1, keepdim=True)
        denom = (max_val - min_val).clamp_min(eps)
        av = (av - min_val) / denom
        av = torch.nan_to_num(av, nan=0.0, posinf=0.0, neginf=0.0)
        return av

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        av1 = self.get_av(x)
        e1 = x

        x = self.layer2(x)
        av2 = self.get_av(x)
        e2 = x

        x = self.layer3(x)
        av3 = self.get_av(x)
        e3 = x

        em = self.layer4(x)
        av4 = self.get_av(em)

        av = torch.cat([av1, av2, av3, av4], dim=1)

        HR = self.fc(self.avgpool(em).view(em.size(0), -1))

        x = self.up1(em)
        x = _fuse_skip(x, e3, self.merge1)
        x = self.up2(x)
        x = _fuse_skip(x, e2, self.merge2)
        x = self.up3(x)
        x = _fuse_skip(x, e1, self.merge3)
        x = self.up4(x)
        Sig = x.squeeze(dim=1)

        return Sig, HR, av


# =============================================================================
# TemporalAttention — helper module used by ArcNet
# =============================================================================

class TemporalAttention(nn.Module):
    """
    Lightweight multi-head self-attention over the temporal axis of a feature map.

    Input:  (B, C, H, T)
    Output: (B, C, H, T)  — same shape, refined by temporal context

    Strategy:
    - Global-average-pool over H -> (B, C, T) so each time step has a C-dim descriptor
    - Run multi-head self-attention across T tokens
    - Broadcast the attended weights back to (B, C, H, T) via channel-wise gating

    This helps the model learn WHICH frames carry heartbeat-relevant signal,
    which is especially useful for noisy regions like the infraorbital area.
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Project C -> d_model for attention (keeps it lightweight)
        self.d_model = max(64, channels // 2)
        self.proj_in  = nn.Linear(channels, self.d_model)
        self.attn     = nn.MultiheadAttention(
            self.d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.proj_out = nn.Linear(self.d_model, channels)
        self.norm     = nn.LayerNorm(channels)
        self.gate     = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, T = x.shape
        # Temporal descriptor: mean over spatial H -> (B, T, C)
        desc = x.mean(dim=2).permute(0, 2, 1)          # (B, T, C)
        # Self-attention over time tokens
        q = self.proj_in(desc)                          # (B, T, d_model)
        attn_out, _ = self.attn(q, q, q)               # (B, T, d_model)
        attn_out = self.proj_out(attn_out)              # (B, T, C)
        # Residual + norm -> gating weights
        gate = self.gate(self.norm(attn_out + desc))    # (B, T, C)
        gate = gate.permute(0, 2, 1).unsqueeze(2)      # (B, C, 1, T)
        # Modulate original features
        return x * gate


# =============================================================================
# ArcNet  (aliased as arc_net for compatibility with train_regions.py)
# =============================================================================

class ArcNet(nn.Module):
    """
    arc_net: drop-in replacement for BaseNet / BaseNetSkip.

    Improvements over BaseNet:
      1. TemporalAwareStem   — asymmetric 3x7 conv better suited to ST maps than
                               the ImageNet isotropic 7x7 stem.
      2. TemporalAttention   — inserted between layer3 and layer4; lets the model
                               attend to heartbeat-relevant frames and suppress
                               domain-variant noise (e.g. eye-region artefacts).
                               Complements the InfoNCE contrastive loss directly.
      3. U-Net skip fusions  — decoder fuses encoder features from layer1/2/3 for
                               sharper BVP waveform reconstruction.

    Forward API:  forward(x) -> (Sig, HR, av)
                  Identical contract to BaseNet and BaseNetSkip.

    Output shapes (example input (B, 3, 9, 128)):
      Sig : (B, H', T')   BVP waveform
      HR  : (B, 1)        heart rate scalar
      av  : (B, 960)      concat of av1..av4, same as BaseNet
    """

    def __init__(self, attn_heads: int = 8, attn_dropout: float = 0.1):
        super().__init__()
        from torchvision.models import ResNet18_Weights

        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stem   = TemporalAwareStem(3, 64)   # replaces resnet conv1/bn1/relu
        self.layer1 = resnet.layer1              # out: (B,  64, H/2,  T/2)
        self.layer2 = resnet.layer2              # out: (B, 128, H/4,  T/4)
        self.layer3 = resnet.layer3              # out: (B, 256, H/8,  T/8)
        self.layer4 = resnet.layer4              # out: (B, 512, H/16, T/16)

        # ── Temporal attention (between layer3 and layer4) ───────────────────
        # channel count matches layer3 output = 256
        self.temporal_attn = TemporalAttention(
            channels=256, num_heads=attn_heads, dropout=attn_dropout
        )

        # ── HR head ──────────────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, 1)

        # ── BVP decoder ──────────────────────────────────────────────────────
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 256, [2, 1], downsample=1),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 64, [1, 1], downsample=1),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 32, [2, 1], downsample=1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(32, 1, [1, 1], downsample=1),
        )

        # ── Skip-connection merge convolutions ────────────────────────────────
        # After up1: decoder=256ch, skip(layer3)=256ch -> concat 512
        self.merge1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # After up2: decoder=64ch, skip(layer2)=128ch -> concat 192
        self.merge2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # After up3: decoder=32ch, skip(layer1)=64ch -> concat 96
        self.merge3 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_av(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Stable per-sample min-max normalized channel descriptor."""
        av = x.mean(dim=(-1, -2))                              # (B, C)
        min_val, _ = torch.min(av, dim=1, keepdim=True)
        max_val, _ = torch.max(av, dim=1, keepdim=True)
        denom = (max_val - min_val).clamp_min(eps)
        av = (av - min_val) / denom
        return torch.nan_to_num(av, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : ST map tensor  (B, 3, H, T)

        Returns:
            Sig : BVP waveform   (B, H', T')  — same shape contract as BaseNet
            HR  : heart rate     (B, 1)
            av  : feature vector (B, 960)     — cat[av1,av2,av3,av4], same as BaseNet
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        x = self.stem(x)

        x  = self.layer1(x);  av1 = self.get_av(x);  e1 = x   # saved for skip
        x  = self.layer2(x);  av2 = self.get_av(x);  e2 = x   # saved for skip
        x  = self.layer3(x);  av3 = self.get_av(x)

        # Temporal attention refines layer3 features before bottleneck
        x  = self.temporal_attn(x);  e3 = x                   # saved for skip

        em = self.layer4(x);  av4 = self.get_av(em)

        # Feature vector for contrastive loss (same shape as BaseNet: B x 960)
        av = torch.cat([av1, av2, av3, av4], dim=1)

        # ── HR head ──────────────────────────────────────────────────────────
        HR = self.fc(self.avgpool(em).view(em.size(0), -1))    # (B, 1)

        # ── BVP decoder with U-Net skip fusions ──────────────────────────────
        x = self.up1(em)
        x = _fuse_skip(x, e3, self.merge1)
        x = self.up2(x)
        x = _fuse_skip(x, e2, self.merge2)
        x = self.up3(x)
        x = _fuse_skip(x, e1, self.merge3)
        x = self.up4(x)
        Sig = x.squeeze(dim=1)

        return Sig, HR, av


# Alias: train_regions.py can import arc_net directly
arc_net = ArcNet