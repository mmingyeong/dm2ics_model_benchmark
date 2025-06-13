"""
shared/metrics.py

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-06-11

Description:
This module provides evaluation metrics for model predictions, including
Mean Squared Error (MSE) and isotropic Power Spectrum analysis.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import fftn
from scipy.stats import binned_statistic

def compute_mse(y_pred, y_true):
    """
    Compute mean squared error between prediction and ground truth.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted tensor of shape (N, 1, D, H, W).
    y_true : torch.Tensor
        Ground truth tensor of shape (N, 1, D, H, W).

    Returns
    -------
    float
        Mean squared error value.
    """
    return F.mse_loss(y_pred, y_true).item()

def compute_power_spectrum(volume, box_size=50.0, bins=30):
    """
    Compute isotropic power spectrum of a 3D volume.

    Parameters
    ----------
    volume : numpy.ndarray
        A 3D density or overdensity field.
    box_size : float
        Physical size of the cube in cMpc/h.
    bins : int
        Number of bins in k-space.

    Returns
    -------
    k_centers : numpy.ndarray
        Bin centers in k-space.
    Pk : numpy.ndarray
        Power spectrum values.
    """
    L = box_size
    N = volume.shape[0]
    delta_k = fftn(volume)
    power = np.abs(delta_k)**2

    k_freq = np.fft.fftfreq(N, d=L/N)
    kx, ky, kz = np.meshgrid(k_freq, k_freq, k_freq, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
    Pk_vals = power.flatten()

    k_bins = np.linspace(0, k_mag.max(), bins + 1)
    Pk, _, _ = binned_statistic(k_mag, Pk_vals, bins=k_bins, statistic='mean')
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    return k_centers, Pk
