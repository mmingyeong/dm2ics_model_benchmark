"""
compute_metrics.py

Description:
    Compute MSE, MAE, PSNR, and Power Spectrum between predicted and target HDF5 subcubes.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-10
"""

import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
from shared.logger import get_logger

logger = get_logger("compute_metrics")


def mse(x, y):
    return ((x - y) ** 2).mean()


def mae(x, y):
    return np.abs(x - y).mean()


def psnr(x, y, max_val=1.0):
    mse_val = mse(x, y)
    return 20 * np.log10(max_val) - 10 * np.log10(mse_val + 1e-8)


def power_spectrum_3d(volume):
    """
    Compute isotropic power spectrum of a 3D volume.
    """
    fft = np.fft.fftn(volume)
    ps = np.abs(fft) ** 2
    ps = np.fft.fftshift(ps)

    shape = volume.shape
    center = [s // 2 for s in shape]
    R = np.sqrt(sum(((np.arange(s) - c)[:, None, None])**2 if i == 0 else
                    ((np.arange(s) - c)[None, :, None])**2 if i == 1 else
                    ((np.arange(s) - c)[None, None, :])**2
                    for i, (s, c) in enumerate(zip(shape, center))))
    R = R.astype(np.int32)

    max_r = R.max()
    ps1d = np.bincount(R.ravel(), weights=ps.ravel()) / np.bincount(R.ravel())
    return ps1d[:max_r]


def main(args):
    # Load prediction and target
    with h5py.File(args.pred_path, 'r') as f_pred, h5py.File(args.target_path, 'r') as f_gt:
        pred = f_pred[args.dataset][:]
        gt = f_gt[args.dataset][:]

    assert pred.shape == gt.shape, "Shape mismatch between prediction and target"

    N = pred.shape[0]
    logger.info(f"📊 Loaded {N} samples of shape {pred.shape[1:]}")

    mse_total = 0.0
    mae_total = 0.0
    psnr_total = 0.0
    ps_list_pred = []
    ps_list_gt = []

    for i in tqdm(range(N), desc="🔍 Evaluating"):
        x = pred[i, 0]
        y = gt[i, 0]

        mse_total += mse(x, y)
        mae_total += mae(x, y)
        psnr_total += psnr(x, y)

        if args.compute_power:
            ps_list_pred.append(power_spectrum_3d(x))
            ps_list_gt.append(power_spectrum_3d(y))

    logger.info("✅ Evaluation completed.")
    logger.info(f"🔢 MSE  = {mse_total / N:.6e}")
    logger.info(f"🔢 MAE  = {mae_total / N:.6e}")
    logger.info(f"🔢 PSNR = {psnr_total / N:.4f} dB")

    if args.compute_power:
        ps_avg_pred = np.mean(np.array(ps_list_pred), axis=0)
        ps_avg_gt = np.mean(np.array(ps_list_gt), axis=0)

        # Save power spectrum to .npy
        np.save(args.power_out_prefix + "_pred.npy", ps_avg_pred)
        np.save(args.power_out_prefix + "_gt.npy", ps_avg_gt)
        logger.info(f"📈 Saved power spectrum to {args.power_out_prefix}_*.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics between prediction and ground truth HDF5 datasets.")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to predicted output HDF5 file.")
    parser.add_argument("--target_path", type=str, required=True,
                        help="Path to ground truth HDF5 file.")
    parser.add_argument("--dataset", type=str, default="subcubes",
                        help="Dataset name inside HDF5 file.")
    parser.add_argument("--compute_power", action="store_true",
                        help="Compute average 3D power spectrum.")
    parser.add_argument("--power_out_prefix", type=str, default="power_spectrum",
                        help="Prefix for saved power spectrum .npy files.")

    args = parser.parse_args()
    main(args)
