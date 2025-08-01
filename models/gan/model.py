"""
model.py (Pix2PixCC-style 3D cGAN)

Description:
    PyTorch implementation of a 3D conditional GAN adapted from Pix2PixCC for voxel-to-voxel
    regression tasks (e.g., reconstructing initial cosmological density from evolved overdensity).
    This version replaces all 2D operations with their 3D counterparts while preserving:
        - Multi-scale PatchGAN discriminators
        - Channel balancing logic
        - Feature matching loss
        - LSGAN objective (least-squares)
        - Correlation / Concordance Correlation Coefficient loss
        - Identity output for regression targets

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-07-31

Reference:
    - JeongHyunJin/Pix2PixCC: https://github.com/JeongHyunJin/Pix2PixCC
    - "Pix2PixCC" paper: https://arxiv.org/pdf/2204.12068.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 상대 import: 같은 패키지 내 utils 사용
from .utils import get_grid, get_norm_layer, get_pad_layer


# ------------------------------------------------------------------
# Compatibility helper for pad layer (handles versions with/without dims arg)
# ------------------------------------------------------------------
def get_pad(pad_type):
    try:
        return get_pad_layer(pad_type, dims=3)
    except TypeError:
        return get_pad_layer(pad_type)


# ------------------------------------------------------------------
# Basic blocks (3D)
# ------------------------------------------------------------------

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ResidualBlock3D(nn.Module):
    def __init__(self, channels, padding_layer, norm_layer, activation):
        super().__init__()
        self.block = nn.Sequential(
            padding_layer(1),
            nn.Conv3d(channels, channels, kernel_size=3, padding=0, stride=1),
            norm_layer(channels),
            activation,
            padding_layer(1),
            nn.Conv3d(channels, channels, kernel_size=3, padding=0, stride=1),
            norm_layer(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ------------------------------------------------------------------
# Generator (3D Pix2PixCC-style)
# ------------------------------------------------------------------

class GeneratorPix2PixCC3D(nn.Module):
    def __init__(self, opt):
        super().__init__()
        input_ch = opt.input_ch
        output_ch = opt.target_ch
        ngf = opt.n_gf

        # normalization and padding from utils (supporting 3D)
        norm = get_norm_layer(opt.norm_type)
        act = Mish()
        pad = get_pad(opt.padding_type)
        trans_conv = opt.trans_conv

        layers = []

        # Initial convolution
        layers += [pad(3), nn.Conv3d(input_ch, ngf, kernel_size=7, padding=0), norm(ngf), act]

        curr_channels = ngf
        # Downsampling
        for _ in range(opt.n_downsample):
            layers += [
                nn.Conv3d(curr_channels, curr_channels * 2, kernel_size=5, padding=2, stride=2),
                norm(curr_channels * 2),
                act,
            ]
            curr_channels *= 2

        # Residual blocks
        for _ in range(opt.n_residual):
            layers += [ResidualBlock3D(curr_channels, pad, norm, act)]

        # Upsampling
        for _ in range(opt.n_downsample):
            if trans_conv:
                layers += [
                    nn.ConvTranspose3d(curr_channels, curr_channels // 2, kernel_size=3, padding=1, stride=2, output_padding=1),
                    norm(curr_channels // 2),
                    act,
                ]
            else:
                layers += [
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                    pad(1),
                    nn.Conv3d(curr_channels, curr_channels // 2, kernel_size=3, padding=0),
                    norm(curr_channels // 2),
                    act,
                ]
            curr_channels //= 2

        # Final projection
        layers += [pad(3), nn.Conv3d(curr_channels, output_ch, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*layers)
        self.final_activation = nn.Identity()  # regression output

    def forward(self, x):
        return self.final_activation(self.model(x))


# ------------------------------------------------------------------
# Patch Discriminator (3D)
# ------------------------------------------------------------------

class PatchDiscriminator3D(nn.Module):
    def __init__(self, opt):
        super().__init__()

        if opt.ch_balance > 0:
            ch_ratio = float(opt.input_ch) / float(opt.target_ch) * opt.ch_balance
            if ch_ratio > 1:
                input_channels = opt.input_ch + opt.target_ch * int(ch_ratio)
            elif ch_ratio < 1:
                input_channels = opt.input_ch * int(1 / ch_ratio) + opt.target_ch
            else:
                input_channels = opt.input_ch + opt.target_ch
        else:
            input_channels = opt.input_ch + opt.target_ch

        ndf = opt.n_df
        norm_layer = get_norm_layer('InstanceNorm3d')  # fixed internal normalization
        act = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        blocks.append(nn.Sequential(
            nn.Conv3d(input_channels, ndf, kernel_size=4, padding=1, stride=2),
            act
        ))
        blocks.append(nn.Sequential(
            nn.Conv3d(ndf, 2 * ndf, kernel_size=4, padding=1, stride=2),
            norm_layer(2 * ndf),
            act
        ))
        blocks.append(nn.Sequential(
            nn.Conv3d(2 * ndf, 4 * ndf, kernel_size=4, padding=1, stride=2),
            norm_layer(4 * ndf),
            act
        ))
        blocks.append(nn.Sequential(
            nn.Conv3d(4 * ndf, 8 * ndf, kernel_size=4, padding=1, stride=1),
            norm_layer(8 * ndf),
            act
        ))
        blocks.append(nn.Sequential(
            nn.Conv3d(8 * ndf, 1, kernel_size=4, padding=1, stride=1)
        ))  # output patch logits

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        features = []
        h = x
        for block in self.blocks:
            h = block(h)
            features.append(h)
        return features  # list of intermediate 3D feature volumes


# ------------------------------------------------------------------
# Multi-scale Discriminator (3D)
# ------------------------------------------------------------------

class MultiScaleDiscriminator3D(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.n_D = opt.n_D
        for i in range(self.n_D):
            setattr(self, f"scale_{i}", PatchDiscriminator3D(opt))
        self.avg_pool3d = nn.AvgPool3d(kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        results = []
        input_for_scale = x
        for i in range(self.n_D):
            disc = getattr(self, f"scale_{i}")
            results.append(disc(input_for_scale))
            if i != self.n_D - 1:
                input_for_scale = self.avg_pool3d(input_for_scale)
        return results  # list over scales, each is list of feature volumes


# ------------------------------------------------------------------
# Loss wrapper (3D adaptation)
# ------------------------------------------------------------------

class Pix2PixCCLoss3D:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(f"cuda:{opt.gpu_ids}" if opt.gpu_ids != -1 else "cpu")
        self.dtype = torch.float16 if getattr(opt, "data_type", 32) == 16 else torch.float32

        self.MSE = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = opt.n_D

    def __call__(self, D, G, input_tensor, target_tensor):
        """
        Compute discriminator and generator losses for 3D volumes.
        Returns:
            loss_D, loss_G, target_tensor, fake_tensor
        """
        fake = G(input_tensor)

        # Real / fake pair construction with channel balancing
        if self.opt.ch_balance > 0:
            real_pair, fake_pair = self._balanced_pairs(input_tensor, target_tensor, fake)
            real_features = D(real_pair)
            fake_features_detached = D(fake_pair.detach())
        else:
            real_features = D(torch.cat((input_tensor, target_tensor), dim=1))
            fake_features_detached = D(torch.cat((input_tensor, fake.detach()), dim=1))

        # Discriminator loss (LSGAN)
        loss_D = 0.0
        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device, self.dtype)
            fake_grid = get_grid(fake_features_detached[i][-1], is_real=False).to(self.device, self.dtype)
            loss_D += (self.MSE(real_features[i][-1], real_grid) + self.MSE(fake_features_detached[i][-1], fake_grid)) * 0.5

        # Generator side
        if self.opt.ch_balance > 0:
            fake_pair = self._make_fake_pair(input_tensor, fake)
            fake_features = D(fake_pair)
            real_pair = self._make_real_pair(input_tensor, target_tensor)
            real_features = D(real_pair)
        else:
            fake_features = D(torch.cat((input_tensor, fake), dim=1))
            real_features = D(torch.cat((input_tensor, target_tensor), dim=1))

        loss_G = 0.0
        loss_G_FM = 0.0
        for i in range(self.n_D):
            real_grid = get_grid(fake_features[i][-1], is_real=True).to(self.device, self.dtype)
            loss_G += self.MSE(fake_features[i][-1], real_grid) * 0.5 * self.opt.lambda_LSGAN

            for j in range(len(fake_features[i])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())
        loss_G += loss_G_FM * (1.0 / self.opt.n_D) * self.opt.lambda_FM

        # Correlation / CCC loss (downsampled)
        for i in range(self.opt.n_CC):
            real_down = target_tensor.to(self.device, self.dtype)
            fake_down = fake.to(self.device, self.dtype)
            for _ in range(i):
                real_down = F.avg_pool3d(real_down, kernel_size=3, stride=2, padding=1)
                fake_down = F.avg_pool3d(fake_down, kernel_size=3, stride=2, padding=1)
            loss_CC = self._inspector(real_down, fake_down)
            loss_G += loss_CC * (1.0 / self.opt.n_CC) * self.opt.lambda_CC

        return loss_D, loss_G, target_tensor, fake

    def _balanced_pairs(self, inp, target, fake):
        if self.opt.ch_balance <= 0:
            real_pair = torch.cat((inp, target), dim=1)
            fake_pair = torch.cat((inp, fake), dim=1)
            return real_pair, fake_pair

        ch_ratio = float(self.opt.input_ch) / float(self.opt.target_ch) * self.opt.ch_balance

        real_pair = torch.cat((inp, target), dim=1)
        fake_pair = torch.cat((inp, fake.detach()), dim=1)

        if ch_ratio > 1:
            for _ in range(int(ch_ratio) - 1):
                real_pair = torch.cat((real_pair, target), dim=1)
                fake_pair = torch.cat((fake_pair, fake.detach()), dim=1)
        elif ch_ratio < 1:
            for _ in range(int(1 / ch_ratio) - 1):
                real_pair = torch.cat((inp, real_pair), dim=1)
                fake_pair = torch.cat((inp, fake_pair), dim=1)

        return real_pair, fake_pair

    def _make_fake_pair(self, inp, fake):
        if self.opt.ch_balance <= 0:
            return torch.cat((inp, fake), dim=1)

        ch_ratio = float(self.opt.input_ch) / float(self.opt.target_ch) * self.opt.ch_balance
        fake_pair = torch.cat((inp, fake), dim=1)

        if ch_ratio > 1:
            for _ in range(int(ch_ratio) - 1):
                fake_pair = torch.cat((fake_pair, fake), dim=1)
        elif ch_ratio < 1:
            for _ in range(int(1 / ch_ratio) - 1):
                fake_pair = torch.cat((inp, fake_pair), dim=1)
        return fake_pair

    def _make_real_pair(self, inp, target):
        if self.opt.ch_balance <= 0:
            return torch.cat((inp, target), dim=1)

        ch_ratio = float(self.opt.input_ch) / float(self.opt.target_ch) * self.opt.ch_balance
        real_pair = torch.cat((inp, target), dim=1)

        if ch_ratio > 1:
            for _ in range(int(ch_ratio) - 1):
                real_pair = torch.cat((real_pair, target), dim=1)
        elif ch_ratio < 1:
            for _ in range(int(1 / ch_ratio) - 1):
                real_pair = torch.cat((inp, real_pair), dim=1)
        return real_pair

    def _inspector(self, target, fake):
        rd = target - torch.mean(target)
        fd = fake - torch.mean(fake)
        r_num = torch.sum(rd * fd)
        r_den = torch.sqrt(torch.sum(rd ** 2)) * torch.sqrt(torch.sum(fd ** 2))
        PCC_val = r_num / (r_den + self.opt.eps)

        if getattr(self.opt, "ccc", False):
            numerator = 2 * PCC_val * torch.std(target) * torch.std(fake)
            denominator = (torch.var(target) + torch.var(fake) +
                           (torch.mean(target) - torch.mean(fake)) ** 2)
            CCC_val = numerator / (denominator + self.opt.eps)
            loss_CC = 1.0 - CCC_val
        else:
            loss_CC = 1.0 - PCC_val
        return loss_CC
