"""
models/unet/predict.py

Run inference using a trained U-Net model and save predictions to HDF5,
preserving the original input HDF5 filenames.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-06-11
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import h5py
import numpy as np
from tqdm import tqdm
from glob import glob

from shared.data_loader import HDF5Dataset
from models.unet.model import UNet3D
from shared.logger import get_logger

logger = get_logger("predict")


def run_prediction(input_dir, output_dir, model_path, device="cuda", batch_size=4):
    """
    Run inference using the test split and save each prediction with the original filename.

    Parameters
    ----------
    input_dir : str
        Directory containing input HDF5 files (full set).
    output_dir : str
        Directory where predicted HDF5 files will be saved.
    model_path : str
        Path to trained model (.pt file).
    device : str
        "cuda" or "cpu".
    batch_size : int
        Batch size for inference.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"📦 Loading model from: {model_path}")
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_input_files = sorted(glob(os.path.join(input_dir, "*.h5")))
    test_input_files = all_input_files[10:12]  # test split 기준

    logger.info(f"🧪 Selected {len(test_input_files)} test input files.")
    logger.debug(f"📁 Input files:\n" + "\n".join(test_input_files))

    for input_path in tqdm(test_input_files, desc="🚀 Running test predictions"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        logger.info(f"📥 Loading input file: {input_path}")
        with h5py.File(input_path, "r") as f:
            x = f["subcubes"][:]

        logger.info(f"🔢 Input shape: {x.shape}, dtype: {x.dtype}")
        if x.ndim == 4:
            x = x[:, None]  # Add channel dim
        elif x.ndim != 5:
            raise ValueError(f"❌ Unexpected input shape: {x.shape}")

        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

        preds = []
        with torch.no_grad():
            for i in range(0, x_tensor.shape[0], batch_size):
                x_batch = x_tensor[i:i + batch_size]
                logger.debug(f"🔁 Predicting batch {i}:{i + batch_size}")
                y_batch = model(x_batch).cpu().numpy()
                preds.append(y_batch)

        y_pred = np.concatenate(preds, axis=0)
        logger.info(f"📐 Prediction complete | Shape: {y_pred.shape}, dtype: {y_pred.dtype}")

        logger.info(f"📤 Saving prediction to: {output_path}")
        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset("subcubes", data=y_pred, compression="gzip")

        logger.info(f"✅ Done: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run U-Net inference on test files, preserving filenames.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with all input HDF5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")

    args = parser.parse_args()
    run_prediction(args.input_dir, args.output_dir, args.model_path, args.device, args.batch_size)
