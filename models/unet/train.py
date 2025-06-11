"""
train.py

Description:
    Training script for the 3D U-Net (V-Net style) model using PyTorch.
    This script loads paired HDF5 volumes (input, target), applies cyclical learning rate,
    and saves both the best and final model checkpoints.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-10

Reference:
    Adapted from the TensorFlow training script:
    https://github.com/redeostm/ML_LocalEnv/blob/main/runSingle.py
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from shared.data_loader import get_dataloader
from shared.logger import get_logger
from models.unet.model import UNet3D


def get_clr_scheduler(optimizer, min_lr, max_lr, cycle_length=8):
    """
    Create a cyclical learning rate scheduler using a triangular waveform.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to apply the learning rate schedule.
    min_lr : float
        Minimum learning rate.
    max_lr : float
        Maximum learning rate.
    cycle_length : int
        Number of epochs for one full cycle.

    Returns
    -------
    scheduler : LambdaLR
        PyTorch learning rate scheduler instance.
    """
    def triangular_clr(epoch):
        mid = cycle_length // 2
        ep = epoch % cycle_length
        if ep <= mid:
            return min_lr + (max_lr - min_lr) * (ep / mid)
        else:
            return max_lr - (max_lr - min_lr) * ((ep - mid) / mid)
    return LambdaLR(optimizer, lr_lambda=triangular_clr)


def train(args):
    """
    Main training loop for the U-Net model.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments specifying training parameters.
    """
    logger = get_logger("train_unet")
    logger.info("🚀 Starting U-Net training with the following configuration:")
    logger.info(vars(args))

    # Load data
    train_loader = get_dataloader(
        args.input_path,
        args.output_path,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Initialize model
    model = UNet3D().to(args.device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.min_lr)
    scheduler = get_clr_scheduler(optimizer, args.min_lr, args.max_lr)

    best_loss = float("inf")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Epoch loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)

        for x, y in loop:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader.dataset)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"📉 Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(args.ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✅ Best model saved: {best_model_path}")

    # Save final model
    final_path = os.path.join(args.ckpt_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"📦 Final model saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on cosmological density fields.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input HDF5 file (e.g., overdensity subcubes).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output HDF5 file (e.g., initial condition subcubes).")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs.")
    parser.add_argument("--min_lr", type=float, default=1e-4,
                        help="Minimum learning rate for CLR.")
    parser.add_argument("--max_lr", type=float, default=1e-3,
                        help="Maximum learning rate for CLR.")
    parser.add_argument("--ckpt_dir", type=str, default="results/unet/",
                        help="Directory to save checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device: 'cuda' or 'cpu'.")
    args = parser.parse_args()

    train(args)
