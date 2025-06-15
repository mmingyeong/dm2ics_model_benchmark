"""
train.py

Description:
    Training script for the Fourier Neural Operator (FNO) model.
    Integrates custom model, logger, and data loader definitions.
    Supports training loop with learning rate scheduler, TensorBoard logging,
    checkpoint saving, and evaluation using a hybrid loss function.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-13

Reference:
    Adapted from: https://github.com/abelsr/Fourier-Neural-Operator/blob/main/training/train.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import pandas as pd

from models.fno.model import FNO
from shared.data_loader import get_dataloader
from shared.logger import get_logger
from shared.losses import hybrid_loss


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_clr_scheduler(optimizer, min_lr, max_lr, cycle_length=8):
    def triangular_clr(epoch):
        mid = cycle_length // 2
        ep = epoch % cycle_length
        scale = ep / mid if ep <= mid else (cycle_length - ep) / mid
        return min_lr / max_lr + (1 - min_lr / max_lr) * scale
    return LambdaLR(optimizer, lr_lambda=triangular_clr)


def train(args):
    logger = get_logger("train_fno")
    logger.info("🚀 Starting FNO training with the following configuration:")
    logger.info(vars(args))

    train_loader = get_dataloader(args.input_path, args.output_path,
                                  batch_size=args.batch_size, split="train", sample_fraction=args.sample_fraction)
    val_loader = get_dataloader(args.input_path, args.output_path,
                                batch_size=args.batch_size, split="val", sample_fraction=args.sample_fraction)

    model = FNO(
        in_channels=1,
        out_channels=1,
        modes1=12, modes2=12, modes3=12,
        width=32,
        add_grid=True
    ).to(args.device)

    criterion = lambda pred, target: hybrid_loss(pred, target, alpha=args.alpha)
    optimizer = Adam(model.parameters(), lr=args.min_lr)
    scheduler = get_clr_scheduler(optimizer, args.min_lr, args.max_lr)
    early_stopper = EarlyStopping(patience=args.patience)

    # Prepare checkpoint and log filenames
    os.makedirs(args.ckpt_dir, exist_ok=True)
    sample_percent = int(args.sample_fraction * 100)
    model_prefix = f"fno_sample{sample_percent}_epoch{args.epochs}"
    best_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_best.pt")
    final_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_final.pt")
    log_path = os.path.join(args.ckpt_dir, f"{model_prefix}_log_train.csv")

    log_records = []

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
        avg_train_loss = epoch_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                pred_val = model(x_val)
                loss_val = criterion(pred_val, y_val)
                val_loss += loss_val.item() * x_val.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"📉 Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        log_records.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr
        })

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

        if avg_val_loss <= early_stopper.best_loss:
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✅ Best model saved: {best_model_path}")

    # Save final model and logs
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"📦 Final model saved: {final_model_path}")

    pd.DataFrame(log_records).to_csv(log_path, index=False)
    logger.info(f"📝 Training log saved to: {log_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FNO with validation and early stopping.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ckpt_dir", type=str, default="results/fno/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    args = parser.parse_args()

    train(args)
