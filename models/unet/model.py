"""
model.py (UNet3D)

Description:
    Implementation of a 3D U-Net model (V-Net style) for volumetric regression tasks using PyTorch.
    The architecture features a symmetrical encoder-decoder structure with skip connections, designed for
    dense prediction tasks on 3D scientific data such as cosmological density fields.
    
    Compared to standard implementations, this model uses:
        - Reduced channel widths for memory efficiency and faster training,
        - A linear (identity) final activation function instead of tanh, 
          to better suit regression targets with values beyond [-1, 1].

    Suitable for tasks such as reconstructing cosmological initial conditions from evolved density fields
    where precise scalar regression is required.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-13

Reference:
    Inspired by the V-Net and U-Net architectures for 3D medical image segmentation and adapted for scientific data modeling.
    Adapted from the TensorFlow architecture script:
    https://github.com/redeostm/ML_LocalEnv/blob/main/generatorSingle.py
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockEnc(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        self.pad = nn.ReplicationPad3d(2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=0)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ConvBlockDec(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.pad = nn.ReplicationPad3d(1)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=1, padding=0)
        self.bn = nn.BatchNorm3d(out_channels)  # ✅ conv 이후 적용
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.upsample(x)

        # 크기 자동 보정
        if x.shape[2:] != skip.shape[2:]:
            diff = [s - x for s, x in zip(skip.shape[2:], x.shape[2:])]
            x = F.pad(x, [
                diff[2] // 2, diff[2] - diff[2] // 2,
                diff[1] // 2, diff[1] - diff[1] // 2,
                diff[0] // 2, diff[0] - diff[0] // 2
            ])

        x = torch.cat([x, skip], dim=1)
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)



class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoding (채널 수 축소)
        self.enc1 = ConvBlockEnc(1, 32)
        self.enc2 = ConvBlockEnc(32, 64)
        self.enc3 = ConvBlockEnc(64, 128)
        self.enc4 = ConvBlockEnc(128, 256)
        self.enc5 = ConvBlockEnc(256, 512)

        # Decoding (채널 수 대응)
        self.dec4 = ConvBlockDec(512 + 256, 256)
        self.dec3 = ConvBlockDec(256 + 128, 128)
        self.dec2 = ConvBlockDec(128 + 64, 64)
        self.dec1 = ConvBlockDec(64 + 32, 32)
        self.out = ConvBlockDec(32 + 1, 1)  # input도 1채널

        # 🔁 마지막 Activation: Tanh → Identity (또는 제거)
        self.final_activation = nn.Identity()

    def forward(self, x):
        # Encoding path
        x1 = x
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x6 = self.enc5(x5)

        # Decoding path
        d5 = self.dec4(x6, x5)
        d4 = self.dec3(d5, x4)
        d3 = self.dec2(d4, x3)
        d2 = self.dec1(d3, x2)
        d1 = self.out(d2, x1)

        return self.final_activation(d1)
