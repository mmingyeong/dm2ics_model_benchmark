"""
model.py (VoxelViT3D: Custom 3D Vision Transformer for voxel-wise regression)

Description:
    Implementation of a custom 3D Vision Transformer (VoxelViT3D) model
    designed for dense voxel-to-voxel regression tasks in volumetric data.

    This architecture divides the 3D volume into patches along depth, height, and width,
    encodes them using transformer blocks, and decodes the output embeddings back into
    full-resolution 3D volumes.

    Applications:
    - Cosmic density field reconstruction
    - Medical 3D image translation
    - Scientific volume modeling and super-resolution

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-07-30
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np

from einops import rearrange

class Rearrange(nn.Module):
    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x):
        return rearrange(x, self.pattern, **self.axes_lengths)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT3D(nn.Module):
    def __init__(
        self,
        image_size=60,
        frames=60,
        image_patch_size=10,
        frame_patch_size=10,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        in_channels=1,
        out_channels=1,
        dim_head=64
    ):
        super().__init__()

        ih, iw = image_size, image_size
        ph, pw = image_patch_size, image_patch_size
        pf = frame_patch_size

        assert ih % ph == 0 and iw % pw == 0 and frames % pf == 0

        self.patch_dims = (frames // pf, ih // ph, iw // pw)
        num_patches = np.prod(self.patch_dims)
        patch_dim = in_channels * ph * pw * pf

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d pd) (h ph) (w pw) -> b (d h w) (pd ph pw c)',
                      pd=pf, ph=ph, pw=pw),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_voxel_patch = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_channels * pf * ph * pw)
        )

        self.out_channels = out_channels
        self.patch_shape = (pf, ph, pw)

    def forward(self, x):
        # x: [B, C=1, D, H, W]
        b = x.shape[0]

        x = self.to_patch_embedding(x)           # [B, N, dim]
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)                  # [B, N, dim]
        x = self.to_voxel_patch(x)              # [B, N, voxels]

        d, h, w = self.patch_dims
        pd, ph, pw = self.patch_shape

        x = x.view(b, d, h, w, self.out_channels, pd, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(b, self.out_channels,
                   d * pd,
                   h * ph,
                   w * pw)
        return x
