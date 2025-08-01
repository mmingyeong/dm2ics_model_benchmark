"""
predict.py

Run inference using a trained 3D Pix2PixCC-style conditional GAN and save predictions,
preserving original input HDF5 filenames.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-07-31

Reference:
    - Pix2PixCC: https://github.com/JeongHyunJin/Pix2PixCC
    - Internal U-Net inference style adapted for conditional GAN.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import sys
import os
import argparse
from glob import glob
from pathlib import Path
import datetime

import torch
import h5py
import numpy as np
from tqdm import tqdm

from shared.logger import get_logger
from shared.data_loader import HDF5Dataset  # assumes dataset yields (input, name) or similar
from models.gan.model import GeneratorPix2PixCC3D  # conditional generator (3D)
# If using multi-channel conditioning you may need to adapt dataset to supply both input and condition

# Utility to wrap the opt expected by the model/loss instantiation
def build_opt_namespace_from_args(args):
    class Opt:
        pass

    opt = Opt()
    opt.input_ch = 1
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
    opt.gpu_ids = 0
    opt.data_type = args.data_type
    # for compatibility (not used here but maybe for image saving)
    opt.image_dir = args.output_dir
    return opt


def run_prediction(input_dir, output_dir, model_path, args):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device)
    logger = get_logger("predict_cgan3d")
    logger.info("Starting prediction")
    logger.info(f"Model: {model_path}")
    logger.info(f"Input directory: {input_dir}, Output directory: {output_dir}")
    logger.info(vars(args))
    start_time = datetime.datetime.now()

    # Build opt and model
    opt = build_opt_namespace_from_args(args)
    generator = GeneratorPix2PixCC3D(opt).to(device)
    logger.info(f"Loading generator checkpoint from {model_path}")
    state = torch.load(model_path, map_location=device)
    # support different checkpoint key formats
    if "G" in state:
        generator.load_state_dict(state["G"])
    else:
        generator.load_state_dict(state)
    generator.eval()

    # Prepare list of input files (HDF5)
    all_input_files = sorted(glob(os.path.join(input_dir, "*.h5")))
    if len(all_input_files) == 0:
        logger.error(f"No .h5 files found in {input_dir}")
        return

    # You can customize selection logic; here we process all
    logger.info(f"Found {len(all_input_files)} input files. Running inference on all.")

    for input_path in tqdm(all_input_files, desc="Predicting volumes"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        logger.info(f"Processing {filename}")

        with h5py.File(input_path, "r") as f_in:
            # Expect dataset named "subcubes" or adjust if different
            if "subcubes" in f_in:
                x = f_in["subcubes"][:]  # shape (N, D, H, W) or (N, C, D, H, W)
            else:
                # fallback: first dataset
                key = list(f_in.keys())[0]
                x = f_in[key][:]

        # Normalize / reshape to (B, C, D, H, W)
        if x.ndim == 4:
            x = x[:, None]  # add channel dim
        elif x.ndim == 5:
            pass
        else:
            raise ValueError(f"Unexpected input shape {x.shape} in {filename}")

        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

        preds = []
        with torch.no_grad():
            for i in range(0, x_tensor.shape[0], args.batch_size):
                x_batch = x_tensor[i : i + args.batch_size]
                y_batch = generator(x_batch).cpu().numpy()
                preds.append(y_batch)

        y_pred = np.concatenate(preds, axis=0)  # (N, C, D, H, W)
        logger.info(f"Prediction shape for {filename}: {y_pred.shape}")

        # Save prediction with same structure
        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset("subcubes", data=y_pred, compression="gzip")
        logger.info(f"Saved prediction to {output_path}")

    elapsed = datetime.datetime.now() - start_time
    logger.info(f"✅ All done. Total time: {elapsed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pix2PixCC-style 3D cGAN inference.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input HDF5 volume files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained generator checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # GAN / model hyperparameters for reconstructing opt namespace
    parser.add_argument("--n_gf", type=int, default=32)
    parser.add_argument("--n_df", type=int, default=32)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--n_residual", type=int, default=4)
    parser.add_argument("--trans_conv", action="store_true")
    parser.add_argument("--padding_type", type=str, default="replication")
    parser.add_argument("--ch_balance", type=float, default=0.0)
    parser.add_argument("--n_D", type=int, default=2)
    parser.add_argument("--lambda_LSGAN", type=float, default=1.0)
    parser.add_argument("--lambda_FM", type=float, default=10.0)
    parser.add_argument("--lambda_CC", type=float, default=1.0)
    parser.add_argument("--n_CC", type=int, default=1)
    parser.add_argument("--ccc", action="store_true")
    parser.add_argument("--data_type", type=int, choices=[16, 32], default=32)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        run_prediction(args.input_dir, args.output_dir, args.model_path, args)
    except Exception:
        print("🔥 Prediction failed due to exception:")
        traceback.print_exc()
        sys.exit(1)
