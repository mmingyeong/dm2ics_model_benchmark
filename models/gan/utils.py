"""
Utility programs of the Pix2PixCC-style 3D cGAN Model (pruned and compatible)

Includes only what's used by the current model / training / prediction pipeline:
  - Real/fake grid for LSGAN targets
  - Normalization and padding layer factories (supports dims argument for compatibility)
  - Weight initialization for 3D Conv and Norm layers

Adapted for 3D by Mingyeong Yang
"""

import torch
import torch.nn as nn
from functools import partial


# ======================================================================
# [1] True or False grid (for LSGAN real/fake targets)
# ======================================================================
def get_grid(input, is_real=True):
    if is_real:
        return torch.ones_like(input, dtype=input.dtype, device=input.device)
    else:
        return torch.zeros_like(input, dtype=input.dtype, device=input.device)


# ======================================================================
# [2] Normalization layer factory (3D only)
# ======================================================================
def get_norm_layer(type):
    if type == 'BatchNorm3d':
        return partial(nn.BatchNorm3d, affine=True)
    elif type == 'InstanceNorm3d':
        return partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError(f"Normalization type '{type}' not supported. Use 'BatchNorm3d' or 'InstanceNorm3d'.")


# ======================================================================
# [3] Padding layer factory (kept signature with dims for compatibility)
# ======================================================================
def get_pad_layer(type, dims=3):
    """
    Returns a padding layer constructor. `dims` selects 2 or 3 for spatial dimensionality.
    """
    if dims == 3:
        if type == 'reflection':
            return lambda d: nn.ReflectionPad3d(d)
        elif type == 'replication':
            return lambda d: nn.ReplicationPad3d(d)
        elif type == 'zero':
            return lambda d: nn.ConstantPad3d(d, 0.0)
        else:
            raise NotImplementedError(f"Padding type '{type}' is not valid for 3D. Choose among ['reflection', 'replication', 'zero'].")
    elif dims == 2:
        if type == 'reflection':
            return nn.ReflectionPad2d
        elif type == 'replication':
            return nn.ReplicationPad2d
        elif type == 'zero':
            return nn.ZeroPad2d
        else:
            raise NotImplementedError(f"Padding type '{type}' is not valid for 2D. Choose among ['reflection', 'replication', 'zero'].")
    else:
        raise ValueError("dims must be 2 or 3.")


# ======================================================================
# [4] Weight initialization (3D-capable)
# ======================================================================
def weights_init(module):
    if isinstance(module, nn.Conv3d):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm3d):
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
