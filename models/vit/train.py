
"""
train.py (ViT3D with Full 3D Volume Regression)

Description:
    Training script for ViT3D modified to perform full 3D density map regression
    using Mean Squared Error (MSE) loss. The model predicts a full (D, H, W) volume instead of scalar.

Author:
    Mingyeong Yang (mmingyeong@kasi.re.kr)
Created:
    2025-07-30
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import pandas as pd

from shared.data_loader import get_dataloader
from shared.logger import get_logger
from shared.losses import mse_loss
from models.vit.model import ViT3D


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
        if ep <= mid:
            scale = ep / mid
        else:
            scale = (cycle_length - ep) / mid
        return min_lr / max_lr + (1 - min_lr / max_lr) * scale

    return LambdaLR(optimizer, lr_lambda=triangular_clr)


def train(args):
    logger = get_logger("train_vit_3dvolume")
    logger.info("🚀 Starting ViT3D training for 3D volume regression:")
    logger.info(vars(args))

    train_loader = get_dataloader(
        args.input_path, args.output_path,
        batch_size=args.batch_size, split="train", sample_fraction=args.sample_fraction,
        shuffle=True, num_workers=0
    )
    val_loader = get_dataloader(
        args.input_path, args.output_path,
        batch_size=args.batch_size, split="val", sample_fraction=args.sample_fraction,
        shuffle=False, num_workers=0
    )

    logger.info(f"📊 Train samples: {len(train_loader.dataset)}")
    logger.info(f"📊 Validation samples: {len(val_loader.dataset)}")

    model = ViT3D(
        image_size=args.image_size,
        frames=args.frames,
        image_patch_size=args.image_patch_size,
        frame_patch_size=args.frame_patch_size,
        dim=args.emb_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        in_channels=1,
        out_channels=1
    ).to(args.device)

    criterion = mse_loss
    optimizer = Adam(model.parameters(), lr=args.min_lr)
    scheduler = get_clr_scheduler(optimizer, args.min_lr, args.max_lr)
    early_stopper = EarlyStopping(patience=args.patience)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    sample_percent = int(args.sample_fraction * 100)
    model_prefix = f"vit_{args.model_name}_3dreg_sample{sample_percent}_epoch{args.epochs}"
    best_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_best.pt")
    final_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_final.pt")
    log_path = os.path.join(args.ckpt_dir, f"{model_prefix}_log_train.csv")

    log_records = []
    best_val_loss = float('inf')  # ✅ Best tracking

    for epoch in range(args.epochs):
        logger.info(f"🔁 Epoch {epoch+1}/{args.epochs} started.")
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=True)

        for step, (x, y) in enumerate(loop):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

            if step % 10 == 0:
                logger.debug(f"🧮 Train Step {step} | Batch Loss: {loss.item():.6f}")

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        logger.info(f"📊 Avg Train Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, (x_val, y_val) in enumerate(val_loader):
                x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                pred_val = model(x_val)
                loss_val = criterion(pred_val, y_val)
                val_loss += loss_val.item() * x_val.size(0)

                if val_step % 10 == 0:
                    logger.debug(f"🔎 Val Step {val_step} | Batch Loss: {loss_val.item():.6f}")

        avg_val_loss = val_loss / len(val_loader.dataset)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"📉 Epoch {epoch+1:03d} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        log_records.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr
        })

        # ✅ Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✅ New best model saved at epoch {epoch+1}: {best_model_path}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            logger.warning(f"🛑 Early stopping triggered at epoch {epoch+1} (best loss: {early_stopper.best_loss:.6f})")
            break

    torch.save(model.state_dict(), final_model_path)
    logger.info(f"📦 Final model saved: {final_model_path}")
    pd.DataFrame(log_records).to_csv(log_path, index=False)
    logger.info(f"📝 Training log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT3D for full 3D regression.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ckpt_dir", type=str, default="results/vit/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_fraction", type=float, default=1.0)

    parser.add_argument("--model_name", type=str, choices=["full"], default="full")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--frame_patch_size", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=512)

    args = parser.parse_args()

    try:
        train(args)
    except Exception as e:
        import traceback
        print("🔥 Training failed due to exception:")
        traceback.print_exc()
        sys.exit(1)