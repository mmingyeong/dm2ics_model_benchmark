"""
models/unet/predict.py

Run inference using a trained U-Net model and save predictions to HDF5.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-06-11
Reference: https://github.com/redeostm/ML_LocalEnv
"""

import torch
import h5py
import os
import numpy as np
from tqdm import tqdm
from shared.data_loader import HDF5Dataset
from models.unet.model import UNet3D
from shared.logger import get_logger

logger = get_logger("predict")

def run_prediction(input_dir, output_dir, model_path, device="cuda"):
    """
    Load a trained model and generate predictions on all HDF5 input files.

    Parameters
    ----------
    input_dir : str
        Path to directory containing input subcube HDF5 files.
    output_dir : str
        Directory to save prediction results as HDF5 files.
    model_path : str
        Path to the trained model (.pt file).
    device : str
        Device to run the model on, e.g., "cuda" or "cpu".
    """
    os.makedirs(output_dir, exist_ok=True)
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".h5")])

    logger.info(f"🔍 Found {len(input_files)} input files.")
    logger.info(f"📦 Loading model from {model_path}")

    model = UNet3D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for filename in tqdm(input_files, desc="🚀 Running predictions"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with h5py.File(input_path, "r") as f:
            x = f["subcubes"][:]

        # Ensure correct shape and dtype
        if x.ndim == 4:
            x_tensor = torch.tensor(x[:, None], dtype=torch.float32).to(device)  # (N, 1, D, H, W)
        elif x.ndim == 5:
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        with torch.no_grad():
            y_pred = model(x_tensor).cpu().numpy()  # (N, 1, D, H, W)

        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset("subcubes", data=y_pred, compression="gzip")

        logger.info(f"✅ Saved prediction to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with a trained U-Net model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input HDF5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    run_prediction(args.input_dir, args.output_dir, args.model_path, args.device)
