"""
models/vit/predict.py

Run inference using a trained ViT3D model and save predictions to HDF5,
preserving the original input HDF5 filenames.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-07-30
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import h5py
import numpy as np
from tqdm import tqdm
from glob import glob

from models.vit.model import ViT3D
from shared.logger import get_logger

logger = get_logger("predict_vit")

def run_prediction(input_dir, output_dir, model_path, device="cuda", batch_size=4,
                   image_size=128, frames=64,
                   image_patch_size=16, frame_patch_size=8,
                   emb_dim=256, depth=6, heads=8, mlp_dim=512):
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"📦 Loading model from: {model_path}")
    model = ViT3D(
        image_size=image_size,
        frames=frames,
        image_patch_size=image_patch_size,
        frame_patch_size=frame_patch_size,
        dim=emb_dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        in_channels=1,
        out_channels=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_input_files = sorted(glob(os.path.join(input_dir, "*.h5")))
    test_input_files = all_input_files[10:12]  # test split

    logger.info(f"🦪 Selected {len(test_input_files)} test input files.")
    logger.debug(f"📁 Input files:\n" + "\n".join(test_input_files))

    for input_path in tqdm(test_input_files, desc="🚀 Running test predictions"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        logger.info(f"📅 Loading input file: {input_path}")
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
                y_batch = model(x_batch).cpu().numpy()  # shape: [B, 1, D, H, W]
                preds.append(y_batch)

        y_pred = np.concatenate(preds, axis=0)  # shape: [N, 1, D, H, W]
        y_pred = np.squeeze(y_pred, axis=1)      # shape: [N, D, H, W]

        logger.info(f"📀 Prediction complete | Shape: {y_pred.shape}, dtype: {y_pred.dtype}")
        logger.info(f"📄 Saving prediction to: {output_path}")
        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset("subcubes", data=y_pred, compression="gzip")

        logger.info(f"✅ Done: {filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ViT3D inference on test files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input HDF5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)

    # Model settings
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--frame_patch_size", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=512)

    args = parser.parse_args()

    run_prediction(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        frames=args.frames,
        image_patch_size=args.image_patch_size,
        frame_patch_size=args.frame_patch_size,
        emb_dim=args.emb_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim
    )