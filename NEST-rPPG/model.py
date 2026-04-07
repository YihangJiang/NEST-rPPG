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

    Replaces the isotropic ResNet 7×7/2 with an asymmetric kernel: 3×7 with stride (2,2)
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

    # def get_av(self, x):
    #     av = torch.mean(torch.mean(x, dim=-1), dim=-1)
    #     min, _ = torch.min(av, dim=1, keepdim=True)
    #     max, _ = torch.max(av, dim=1, keepdim=True)
    #     print(min, max)
    #     av = torch.mul((av-min),((max-min).pow(-1)))
    #     return av
    def get_av(self, x, eps: float = 1e-6):
        """
        Global channel descriptor with stable per-sample min–max normalization.

        Steps per sample:
        - Global average pool over spatial/temporal dims -> (C,) channel means
        - Min–max normalize across channels with an epsilon in the denominator
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
    - TemporalAwareStem (3×7, stride 2) instead of ImageNet 7×7 stem
    - BVP decoder fuses skip features from layer3, layer2, layer1 (U-Net-style)

    Forward API matches BaseNet: returns (Sig, HR, av).
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights

        model_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Temporal-aware stem (no ImageNet conv1 weights); layer1–4 still pretrained.
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