"""
plot_power_spectrum.py

Description:
    Plots 1D power spectrum curves of predicted and ground truth 3D volumes.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-10

Usage:
    python evaluation/plot_power_spectrum.py \
        --pred power_pred.npy \
        --gt power_gt.npy \
        --outfig results/unet/power_comparison.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from shared.logger import get_logger

logger = get_logger("plot_power")


def plot_power_spectrum(pred_path, gt_path, save_path=None, loglog=True):
    ps_pred = np.load(pred_path)
    ps_gt = np.load(gt_path)

    k = np.arange(len(ps_pred))  # Wavenumber bins

    plt.figure(figsize=(7, 5))
    if loglog:
        plt.loglog(k[1:], ps_gt[1:], label="Ground Truth", linewidth=2)
        plt.loglog(k[1:], ps_pred[1:], label="Prediction", linestyle="--", linewidth=2)
    else:
        plt.plot(k, ps_gt, label="Ground Truth", linewidth=2)
        plt.plot(k, ps_pred, label="Prediction", linestyle="--", linewidth=2)

    plt.xlabel("Wavenumber $k$")
    plt.ylabel("Power Spectrum $P(k)$")
    plt.title("Power Spectrum Comparison")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"📊 Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot power spectrum comparison")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted power spectrum .npy")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth power spectrum .npy")
    parser.add_argument("--outfig", type=str, default="power_comparison.png", help="Path to save figure")
    parser.add_argument("--linear", action="store_true", help="Plot in linear instead of log-log scale")
    args = parser.parse_args()

    plot_power_spectrum(
        pred_path=args.pred,
        gt_path=args.gt,
        save_path=args.outfig,
        loglog=not args.linear
    )
