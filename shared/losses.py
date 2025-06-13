"""
shared/losses.py

Loss function definitions for training cosmological reconstruction models.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-06-11
Reference: https://github.com/redeostm/ML_LocalEnv
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn.functional as F

def mse_loss(pred, target):
    """
    Compute the Mean Squared Error (MSE) between prediction and target.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor.
    target : torch.Tensor
        Ground truth tensor.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss.
    """
    return F.mse_loss(pred, target)


def spectral_loss(pred, target):
    """
    Compute spectral loss between prediction and target in Fourier space.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor of shape (B, C, D, H, W).
    target : torch.Tensor
        Ground truth tensor of same shape.

    Returns
    -------
    torch.Tensor
        Scalar spectral loss value.
    """
    # Apply FFT and shift zero freq to center
    pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1), norm='ortho')
    target_fft = torch.fft.fftn(target, dim=(-3, -2, -1), norm='ortho')

    # Compute power spectrum loss
    loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
    return loss

def hybrid_loss(pred, target, alpha=0.1):
    """
    Combine MSE loss and spectral loss.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor.
    target : torch.Tensor
        Ground truth tensor.
    alpha : float
        Weight for the spectral loss component.

    Returns
    -------
    torch.Tensor
        Combined loss value.
    """
    mse = mse_loss(pred, target)
    spec = spectral_loss(pred, target)
    return mse + alpha * spec
