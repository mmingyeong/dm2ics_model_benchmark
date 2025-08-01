"""
train.py

Description:
    Training script for the 3D Pix2PixCC-style conditional GAN (voxel-to-voxel regression).
    Uses a multi-scale PatchGAN discriminator, feature matching, LSGAN loss, and optional
    correlation/CCC loss. Designed to reconstruct initial conditions from evolved density fields.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-07-31

Reference:
    - Pix2PixCC: https://github.com/JeongHyunJin/Pix2PixCC
    - U-Net training style adapted from internal project conventions.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import sys
import os
import argparse
from pathlib import Path
import datetime
import traceback

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np

# 사용자 프로젝트 내부 모듈 (가정)
from shared.logger import get_logger
from shared.data_loader import get_dataloader  # 또는 너의 cGAN용 dataset wrapper
from models.gan.model import GeneratorPix2PixCC3D, MultiScaleDiscriminator3D, Pix2PixCCLoss3D  # 상대경로 맞춰서 import
from models.gan.utils import weights_init  # 초기화 함수가 utils에 있다면


# Early stopping (optional)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D Pix2PixCC-style cGAN.")
    parser.add_argument("--train_input", type=str, required=True, help="Path or pattern for training input volumes")
    parser.add_argument("--train_target", type=str, required=True, help="Path or pattern for training target volumes")
    parser.add_argument("--val_input", type=str, default=None)
    parser.add_argument("--val_target", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5, help="For CLR scheduler lower bound")
    parser.add_argument("--max_lr", type=float, default=1e-4, help="For CLR scheduler upper bound")
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda_LSGAN", type=float, default=1.0)
    parser.add_argument("--lambda_FM", type=float, default=10.0)
    parser.add_argument("--lambda_CC", type=float, default=1.0)
    parser.add_argument("--n_CC", type=int, default=1)
    parser.add_argument("--ch_balance", type=float, default=0.0)
    parser.add_argument("--n_D", type=int, default=2)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--n_residual", type=int, default=4)
    parser.add_argument("--n_gf", type=int, default=32)
    parser.add_argument("--n_df", type=int, default=32)
    parser.add_argument("--trans_conv", action="store_true", help="Use transpose conv for upsampling")
    parser.add_argument("--padding_type", type=str, default="replication")
    parser.add_argument("--data_type", type=int, choices=[16, 32], default=32)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cycle_length", type=int, default=8)
    parser.add_argument("--ckpt_dir", type=str, default="results/cgan/")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_freq", type=int, default=1, help="Save every N epochs")
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--display_freq", type=int, default=1, help="Save example images every N epochs")
    parser.add_argument("--report_freq", type=int, default=50, help="Log batch-level loss every N batches")
    args = parser.parse_args()
    return args


def build_opt_namespace(args):
    # Pix2PixCCLoss3D and models expect an opt-like object with certain attributes
    class Opt:
        pass

    opt = Opt()
    # required attributes
    opt.input_ch = 1  # adjust if input has more channels
    opt.target_ch = 1
    opt.n_gf = args.n_gf
    opt.n_df = args.n_df
    opt.n_downsample = args.n_downsample
    opt.n_residual = args.n_residual
    opt.trans_conv = args.trans_conv
    opt.padding_type = args.padding_type
    opt.norm_type = "InstanceNorm3d"
    opt.ch_balance = args.ch_balance
    opt.n_D = args.n_D
    opt.lambda_LSGAN = args.lambda_LSGAN
    opt.lambda_FM = args.lambda_FM
    opt.lambda_CC = args.lambda_CC
    opt.n_CC = args.n_CC
    opt.ccc = False
    opt.eps = 1e-8
    opt.gpu_ids = 0  # single GPU; adapt if multi
    opt.data_type = args.data_type
    return opt


def volume_to_image_tensor(vol_tensor, mode='slice'):
    """
    Simplified conversion of a 3D volume tensor to a 2D uint8 numpy image for saving.
    Takes middle slice along depth for visualization.
    """
    if vol_tensor.dim() == 5:
        tensor = vol_tensor[0]  # (C, D, H, W)
    elif vol_tensor.dim() == 4:
        tensor = vol_tensor
    else:
        raise ValueError("Expected 4D or 5D tensor for volume-to-image conversion.")

    # assume (C, D, H, W)
    C, D, H, W = tensor.shape
    idx = D // 2
    slice_2d = tensor[:, idx, :, :]  # (C, H, W)
    np_image = slice_2d.cpu().float().numpy()
    if np_image.ndim == 3:
        if np_image.shape[0] == 1:
            np_image = np_image[0]
        elif np_image.shape[0] == 3:
            np_image = np.transpose(np_image, (1, 2, 0))
        else:
            np_image = np.mean(np_image, axis=0)
    # dynamic range from [-1,1] to [0,255]
    np_image = (np_image + 1) / 2  # assume network outputs roughly in [-1,1]
    np_image = np.clip(np_image * 255, 0, 255).astype(np.uint8)
    if np_image.ndim == 2:
        np_image = np.stack([np_image] * 3, axis=-1)
    return np_image  # HWC uint8


def save_example_images(x, y, fake, save_dir, step_label):
    os.makedirs(save_dir, exist_ok=True)
    real_img = volume_to_image_tensor(y, mode='slice')
    fake_img = volume_to_image_tensor(fake, mode='slice')
    input_img = volume_to_image_tensor(x, mode='slice')

    Image.fromarray(real_img).save(os.path.join(save_dir, f"{step_label}_real.png"))
    Image.fromarray(fake_img).save(os.path.join(save_dir, f"{step_label}_fake.png"))
    Image.fromarray(input_img).save(os.path.join(save_dir, f"{step_label}_input.png"))


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device(args.device)
    logger = get_logger("train_cgan3d")
    logger.info("Starting Pix2PixCC-style 3D cGAN training")
    logger.info(vars(args))
    start_time = datetime.datetime.now()

    # Build unified opt for model/loss
    opt = build_opt_namespace(args)
    opt.model_dir = os.path.join(args.ckpt_dir, "models")
    opt.image_dir = os.path.join(args.ckpt_dir, "images")
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.image_dir, exist_ok=True)

    # Data loaders (you need to adapt get_dataloader or provide a similar paired-volume dataset)
    train_loader = get_dataloader(
        args.train_input, args.train_target,
        batch_size=args.batch_size, split="train", sample_fraction=args.sample_fraction,
        shuffle=True, num_workers=0
    )
    val_loader = None
    if args.val_input and args.val_target:
        val_loader = get_dataloader(
            args.val_input, args.val_target,
            batch_size=args.batch_size, split="val", sample_fraction=args.sample_fraction,
            shuffle=False, num_workers=0
        )

    # Models
    G = GeneratorPix2PixCC3D(opt).to(device)
    D = MultiScaleDiscriminator3D(opt).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # Optimizers
    optimizer_G = Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_G = get_clr_scheduler(optimizer_G, args.min_lr, args.max_lr, cycle_length=args.cycle_length)
    scheduler_D = get_clr_scheduler(optimizer_D, args.min_lr, args.max_lr, cycle_length=args.cycle_length)

    criterion = Pix2PixCCLoss3D(opt)
    early_stopper = EarlyStopping(patience=args.patience)

    # Checkpoints
    best_model_path = os.path.join(opt.model_dir, "cgan_best.pt")
    final_model_path = os.path.join(opt.model_dir, "cgan_final.pt")
    log_path = os.path.join(opt.model_dir, "training_log.csv")
    records = []

    start_epoch = 0
    if args.resume:
        if os.path.isfile(best_model_path):
            logger.info(f"Resuming from best checkpoint: {best_model_path}")
            state = torch.load(best_model_path, map_location=device)
            G.load_state_dict(state["G"])
            D.load_state_dict(state["D"])
            optimizer_G.load_state_dict(state["opt_G"])
            optimizer_D.load_state_dict(state["opt_D"])
            start_epoch = state.get("epoch", 0)
        else:
            logger.warning("Resume flag set but no checkpoint found; starting fresh.")

    total_steps = len(train_loader.dataset) * args.epochs
    current_step = 0

    for epoch in range(start_epoch, args.epochs):
        G.train()
        D.train()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=True)

        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)

            # GAN forward
            loss_D, loss_G, _, fake = criterion(D, G, x, y)

            # Update generator
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Update discriminator
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            epoch_loss_G += loss_G.item() * x.size(0)
            epoch_loss_D += loss_D.item() * x.size(0)
            current_step += x.size(0)

            # Batch-level reporting
            if (batch_idx + 1) % args.report_freq == 0:
                percent = current_step / total_steps * 100
                logger.info(f"[Epoch {epoch+1}] Step {current_step}/{total_steps} ({percent:.2f}%) G_loss: {loss_G.item():.4f} D_loss: {loss_D.item():.4f}")

            loop.set_postfix({
                "G_loss": loss_G.item(),
                "D_loss": loss_D.item()
            })

        scheduler_G.step()
        scheduler_D.step()

        avg_train_G = epoch_loss_G / len(train_loader.dataset)
        avg_train_D = epoch_loss_D / len(train_loader.dataset)

        # Validation (optional)
        val_loss = None
        if val_loader is not None:
            G.eval()
            D.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    loss_D_val, loss_G_val, _, _ = criterion(D, G, x_val, y_val)
                    val_loss_accum += loss_G_val.item() * x_val.size(0)
            avg_val_loss = val_loss_accum / len(val_loader.dataset)
            val_loss = avg_val_loss
            logger.info(f"Epoch {epoch+1:03d} | Train G: {avg_train_G:.6f} D: {avg_train_D:.6f} | Val G: {avg_val_loss:.6f}")
        else:
            logger.info(f"Epoch {epoch+1:03d} | Train G: {avg_train_G:.6f} D: {avg_train_D:.6f}")

        # Logging record
        current_lr = scheduler_G.get_last_lr()[0]
        records.append({
            "epoch": epoch + 1,
            "train_G_loss": avg_train_G,
            "train_D_loss": avg_train_D,
            "val_G_loss": val_loss if val_loss is not None else None,
            "lr": current_lr,
        })

        # Early stopping
        monitor_loss = val_loss if val_loss is not None else avg_train_G
        early_stopper(monitor_loss)
        if early_stopper.early_stop:
            logger.info(f"🛑 Early stopping at epoch {epoch+1}")
            break

        # Save best model
        is_best = (val_loss is not None and val_loss <= early_stopper.best_loss) or (
            val_loss is None and avg_train_G <= early_stopper.best_loss
        )
        if is_best:
            torch.save({
                "epoch": epoch + 1,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": optimizer_G.state_dict(),
                "opt_D": optimizer_D.state_dict(),
            }, best_model_path)
            logger.info(f"✅ Best checkpoint saved to {best_model_path}")

        # Per-epoch checkpoint
        if (epoch + 1) % args.save_freq == 0:
            epoch_path = os.path.join(opt.model_dir, f"cgan_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": optimizer_G.state_dict(),
                "opt_D": optimizer_D.state_dict(),
            }, epoch_path)
            logger.info(f"💾 Epoch checkpoint saved to {epoch_path}")

        # Example image saving (real/fake/input) every display_freq epochs
        if (epoch + 1) % args.display_freq == 0:
            # use the last batch from this epoch
            save_example_images(x, y, fake, opt.image_dir, step_label=f"epoch{epoch+1}")
            logger.info(f"🖼 Saved example real/fake/input images for epoch {epoch+1}")

    # Final save
    torch.save({
        "epoch": epoch + 1,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_G": optimizer_G.state_dict(),
        "opt_D": optimizer_D.state_dict(),
    }, final_model_path)
    logger.info(f"📦 Final model saved: {final_model_path}")

    # Log CSV
    pd.DataFrame(records).to_csv(log_path, index=False)
    logger.info(f"📝 Training log written to {log_path}")

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    logger.info(f"⏱ Total training time: {elapsed}")


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except Exception:
        print("🔥 Training failed due to exception:")
        traceback.print_exc()
        sys.exit(1)
