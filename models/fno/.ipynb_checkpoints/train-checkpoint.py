"""
train_lightning_fno.py

Description:
    PyTorch Lightning training script for the Fourier Neural Operator (FNO).
    Includes early stopping and model checkpointing for best validation loss.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-13

Reference:
    Adapted from Fourier-Neural-Operator/train_lightning.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.fno.model import FNO
from shared.losses import hybrid_loss
from shared.logger import get_logger
from shared.data_loader import get_dataloader

logger = get_logger("train_fno_lightning")


class LitFNO(pl.LightningModule):
    """
    Lightning wrapper for the FNO model with training and validation steps.
    """
    def __init__(self, model: nn.Module, alpha: float = 0.1, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.lr = lr
        self.save_hyperparameters(ignore=["model"])
        self.criterion = lambda pred, target: hybrid_loss(pred, target, alpha=self.alpha)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_dir", type=str, default="logs/lightning_fno/")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/lightning_fno/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result_dir", type=str, default="results/fno/",
                    help="Directory to save logs and checkpoints.")
    args = parser.parse_args()

    logger.info("🚀 Starting Lightning FNO Training...")
    logger.info(vars(args))

    train_loader = get_dataloader(args.input_path, args.output_path, batch_size=args.batch_size, split="train")
    val_loader = get_dataloader(args.input_path, args.output_path, batch_size=args.batch_size, split="val")

    # FNO model
    model = FNO(
        in_channels=1,
        out_channels=1,
        modes1=16,
        modes2=16,
        modes3=16,
        width=128,
        lifting_channels=64,
        add_grid=True
    )

    lit_model = LitFNO(model=model, alpha=args.alpha, lr=args.lr)

    # Logger and Callbacks
    log_dir = os.path.join(args.result_dir, "logs")
    ckpt_dir = os.path.join(args.result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    lightning_logger = CSVLogger(save_dir=log_dir, name="fno")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-fno",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.device.startswith("cuda") else "cpu",
        logger=lightning_logger,
        callbacks=[early_stopping, checkpoint_cb],
        log_every_n_steps=10
    )

    # Train
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final model as .pt (standard PyTorch format)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final_model.pt"))
    logger.info(f"📦 Final PyTorch model saved to {os.path.join(ckpt_dir, 'final_model.pt')}")



if __name__ == "__main__":
    main()
