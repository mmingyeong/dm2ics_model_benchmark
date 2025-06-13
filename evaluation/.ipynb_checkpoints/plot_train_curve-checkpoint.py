import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curve(csv_path, save_path=None):
    """
    Plot training and validation loss curves from CSV log.

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing epoch-wise training logs.
    save_path : str or None
        If specified, save the plot image to this path. Otherwise, display interactively.
    """
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training & Validation Loss Curve", fontsize=14)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"✅ Loss curve saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss curves.")
    parser.add_argument("--log_csv", type=str, required=True, help="Path to training log CSV (e.g., log_train.csv)")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the figure")
    args = parser.parse_args()

    plot_training_curve(args.log_csv, args.save_path)
