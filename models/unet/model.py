"""
model.py

Description:
    3D U-Net (V-Net style) architecture for cosmological density field reconstruction.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-10

Reference:
    This implementation is adapted from the 3D V-Net model in:
    https://github.com/redeostm/ML_LocalEnv/blob/main/generatorSingle.py

License:
    For academic use only. Contact the author before redistribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlockEnc(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        self.pad = nn.ReplicationPad3d(2)  # for 5x5x5
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
        self.pad = nn.ReplicationPad3d(1)  # for 3x3x3
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=1, padding=0)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.bn(x)
        x = self.pad(x)
        x = self.conv(x)
        return self.relu(x)


class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoding
        self.enc1 = ConvBlockEnc(2, 128)
        self.enc2 = ConvBlockEnc(128, 256)
        self.enc3 = ConvBlockEnc(256, 512)
        self.enc4 = ConvBlockEnc(512, 1024)
        self.enc5 = ConvBlockEnc(1024, 2048)

        # Decoding
        self.dec4 = ConvBlockDec(2048 + 1024, 1024)
        self.dec3 = ConvBlockDec(1024 + 512, 512)
        self.dec2 = ConvBlockDec(512 + 256, 256)
        self.dec1 = ConvBlockDec(256 + 128, 128)
        self.out = ConvBlockDec(128 + 2, 1)  # concatenate with input

        self.final_activation = nn.Tanh()

    def forward(self, x):
        # Encoding
        x1 = x  # (2, 128^3)
        x2 = self.enc1(x1)   # (128, 64^3)
        x3 = self.enc2(x2)   # (256, 32^3)
        x4 = self.enc3(x3)   # (512, 16^3)
        x5 = self.enc4(x4)   # (1024, 8^3)
        x6 = self.enc5(x5)   # (2048, 4^3)

        # Decoding
        d5 = self.dec4(x6, x5)
        d4 = self.dec3(d5, x4)
        d3 = self.dec2(d4, x3)
        d2 = self.dec1(d3, x2)
        d1 = self.out(d2, x1)  # 마지막은 input과 concatenate

        return self.final_activation(d1)