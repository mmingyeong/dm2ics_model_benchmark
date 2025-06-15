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

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import pandas as pd

from shared.data_loader import get_dataloader
from shared.logger import get_logger
from shared.losses import hybrid_loss
from models.unet.model import UNet3D


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
logger = get_logger("train_unet")
logger.info("🚀 Starting U-Net training with the following configuration:")
logger.info(vars(args))

print("📌 Before train_loader")
train_loader = get_dataloader(args.input_path, args.output_path,
batch_size=args.batch_size, split="train", sample_fraction=args.sample_fraction)
print("✅ After train_loader")

val_loader = get_dataloader(args.input_path, args.output_path,
batch_size=args.batch_size, split="val", sample_fraction=args.sample_fraction)
print("✅ After val_loader")

model = UNet3D().to(args.device)
print("✅ After model.to()")


# Initialize model, loss, optimizer, scheduler
model = UNet3D().to(args.device)
criterion = lambda pred, target: hybrid_loss(pred, target, alpha=args.alpha)
optimizer = Adam(model.parameters(), lr=args.min_lr)
scheduler = get_clr_scheduler(optimizer, args.min_lr, args.max_lr)
early_stopper = EarlyStopping(patience=args.patience)

# Prepare checkpoint paths with sample_fraction and epoch count
os.makedirs(args.ckpt_dir, exist_ok=True)
sample_percent = int(args.sample_fraction * 100)
model_prefix = f"unet_sample{sample_percent}_epoch{args.epochs}"
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
if epoch % 10 == 0:
logger.info(f"🔍 Val pred: min={pred_val.min().item():.4f}, max={pred_val.max().item():.4f}, mean={pred_val.mean().item():.4f}")
avg_val_loss = val_loss / len(val_loader.dataset)

current_lr = scheduler.get_last_lr()[0]
logger.info(f"📉 Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

log_records.append({
"epoch": epoch + 1,
"train_loss": avg_train_loss,
"val_loss": avg_val_loss,
"lr": current_lr
})

# Early stopping check
early_stopper(avg_val_loss)
if early_stopper.early_stop:
logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
break

# Save best model
if avg_val_loss <= early_stopper.best_loss:
torch.save(model.state_dict(), best_model_path)
logger.info(f"✅ Best model saved: {best_model_path}")

# Save final model and logs
torch.save(model.state_dict(), final_model_path)
logger.info(f"📦 Final model saved: {final_model_path}")

pd.DataFrame(log_records).to_csv(log_path, index=False)
logger.info(f"📝 Training log saved to: {log_path}")


if __name__ == "__main__":
parser = argparse.ArgumentParser(description="Train U-Net with validation and early stopping.")
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--min_lr", type=float, default=1e-4)
parser.add_argument("--max_lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--alpha", type=float, default=0.05,
help="Weight for spectral loss in hybrid loss function.")
parser.add_argument("--ckpt_dir", type=str, default="results/unet/")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--sample_fraction", type=float, default=1.0,
help="Fraction of dataset to use for training/validation. E.g., 0.1 for 10%")
args = parser.parse_args()

train(args)