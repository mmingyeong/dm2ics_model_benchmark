import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import gaussian_kde, skew, kurtosis
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def plot_loss_curve_multi(log_csv_paths: dict, output_dir: str = None, show: bool = False):
    """
    Plot training/validation loss curves for multiple models on a single figure.

    Parameters
    ----------
    log_csv_paths : dict
        Mapping from model label to CSV log file path containing at least 'epoch' and 'train_loss'.
    output_dir : str
        Directory to save the combined loss curve.
    show : bool
        If True, display the figure.
    """
    plt.figure(figsize=(8, 5))
    for model_name, csv_path in log_csv_paths.items():
        if not os.path.exists(csv_path):
            logging.warning(f"Loss curve skipped for {model_name}: file not found -> {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.warning(f"Failed to read {csv_path} for {model_name}: {e}")
            continue
        if 'epoch' not in df.columns or 'train_loss' not in df.columns:
            logging.warning(f"Loss curve skipped for {model_name}: missing required columns.")
            continue
        plt.plot(df['epoch'], df['train_loss'], label=f'{model_name} Train', marker='o')
        if 'val_loss' in df.columns:
            plt.plot(df['epoch'], df['val_loss'], label=f'{model_name} Val', linestyle='--', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "multi_loss_curve.png")
        plt.savefig(save_path, dpi=150)
        logging.info(f"📉 Multi loss curve saved to {save_path}")
    if show:
        plt.show()
    plt.close()


def compare_density_distribution_multi(
    y_true: np.ndarray,
    model_outputs: dict,
    save_dir: str = "./",
    index: int = 0,
    nbins: int = 50,
    bin_range: tuple = (0.5, 20),
    show: bool = False,
):
    """
    Unified comparison of the 1+δ distribution (PDF) for multiple predicted models and ground truth.

    Parameters
    ----------
    y_true : ndarray
        Ground truth overdensity field δ (3D).
    model_outputs : dict
        Dict mapping model label to predicted overdensity field δ (3D).
    save_dir : str
        Directory to save figure.
    index : int
        Subcube index for annotation.
    nbins : int
        Number of log-spaced bins.
    bin_range : tuple
        (min, max) range for 1+δ.
    show : bool
        If True, display figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Ground truth 1 + δ
    one_plus_true = 1 + y_true.flatten()
    mask_true = np.isfinite(one_plus_true) & (one_plus_true > 0)
    one_plus_true = one_plus_true[mask_true]

    bins = np.logspace(np.log10(bin_range[0]), np.log10(bin_range[1]), nbins)
    hist_true, _ = np.histogram(one_plus_true, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, hist_true, label="True", color="black", linewidth=2)

    for model_name, pred in model_outputs.items():
        one_plus_pred = 1 + pred.flatten()
        mask_pred = np.isfinite(one_plus_pred) & (one_plus_pred > 0)
        one_plus_pred = one_plus_pred[mask_pred]
        hist_pred, _ = np.histogram(one_plus_pred, bins=bins, density=True)
        plt.plot(bin_centers, hist_pred, label=model_name, linestyle="--", linewidth=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$1 + \delta$")
    plt.ylabel("PDF (normalized)")
    plt.title(f"Unified (1+δ) Distribution Comparison (Subcube {index})")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"unified_density_distribution_idx{index}.png")
    plt.savefig(save_path, dpi=150)
    logging.info(f"📈 Unified 1+δ comparison saved to: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_projection_comparison_multi(
    input_cube: np.ndarray,
    gt_cube: np.ndarray,
    model_outputs: dict,
    axis: int = 0,
    index: int = None,
    save_dir: str = None,
    show: bool = False,
):
    """
    Unified 2D projection comparison: input, ground truth, and multiple model predictions.

    Parameters
    ----------
    input_cube : ndarray
        Evolved input cube.
    gt_cube : ndarray
        Ground truth cube.
    model_outputs : dict
        Mapping model label -> predicted cube.
    axis : int
        Axis along which to project.
    index : int
        Subcube index.
    save_dir : str
        Output directory.
    show : bool
        If True, display figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    input_proj = np.log1p(np.sum(input_cube, axis=axis))
    gt_proj = np.log1p(np.sum(gt_cube, axis=axis))
    n_models = len(model_outputs)
    fig, axs = plt.subplots(1, 2 + n_models, figsize=(6 * (2 + n_models), 5))

    # Input and GT
    axs[0].imshow(input_proj, origin="lower", cmap="viridis")
    axs[0].set_title("Input (log1p)")
    axs[1].imshow(gt_proj, origin="lower", cmap="viridis")
    axs[1].set_title("Ground Truth (log1p)")

    # Predictions
    for i, (model_name, pred_cube) in enumerate(model_outputs.items(), start=2):
        pred_proj = np.log1p(np.sum(pred_cube, axis=axis))
        axs[i].imshow(pred_proj, origin="lower", cmap="viridis")
        axs[i].set_title(f"{model_name} Prediction (log1p)")

    for ax in axs:
        ax.set_xlabel("X [voxel] (1 voxel = 0.82 Mpc/h)")
        ax.set_ylabel("Y [voxel] (1 voxel = 0.82 Mpc/h)")
        ax.axis("off")
        # add individual colorbars
        im = ax.get_images()
        if im:
            plt.colorbar(im[0], ax=ax, fraction=0.046, pad=0.04).set_label("log(1 + projected density)")

    info = f"Projection axis = {axis} | Subcube index = {index}"
    plt.suptitle(info, fontsize=15, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"unified_projection_idx{index}_axis{axis}.png")
    plt.savefig(save_path, dpi=150)
    logging.info(f"🖼️  Unified projection comparison saved to {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_pixel_scatter_multi(
    y_true: np.ndarray,
    model_outputs: dict,
    index: int = None,
    save_dir: str = None,
    log_transform: bool = True,
    show: bool = False,
):
    """
    Unified pixel-to-pixel scatter comparing multiple model predictions against ground truth.

    Parameters
    ----------
    y_true : ndarray
    model_outputs : dict
        Mapping model label -> predicted cube.
    index : int
        Subcube index.
    save_dir : str
        Output directory.
    log_transform : bool
        Whether to apply log1p to values.
    show : bool
        If True, display figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    if log_transform:
        y_true_plot = np.log1p(y_true)
    else:
        y_true_plot = y_true
    true_flat = y_true_plot.flatten()

    plt.figure(figsize=(6, 6))
    for model_name, y_pred in model_outputs.items():
        if log_transform:
            y_pred_plot = np.log1p(y_pred)
        else:
            y_pred_plot = y_pred
        pred_flat = y_pred_plot.flatten()
        plt.scatter(true_flat, pred_flat, s=1, alpha=0.3, label=model_name)

    min_val = np.min(true_flat)
    max_val = np.max(true_flat)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1")
    plt.xlabel(f"Ground Truth ({'log1p' if log_transform else 'linear'})")
    plt.ylabel(f"Prediction ({'log1p' if log_transform else 'linear'})")
    plt.title(f"Multi-Model Pixel-to-Pixel Comparison (Subcube {index})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"unified_pixel_scatter_idx{index}.png")
    plt.savefig(save_dir and save_path, dpi=150) if save_dir else plt.savefig(f"unified_pixel_scatter_idx{index}.png", dpi=150)
    logging.info(f"📊 Multi pixel scatter plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()


def analyze_gaussianity_multi(
    y_true: np.ndarray,
    model_outputs: dict,
    index: int = None,
    save_dir: str = "./",
    log_transform: bool = True,
    bins: int = 100,
    show: bool = False,
):
    """
    Unified Gaussianity comparison: histograms + normal fit for multiple models.

    Parameters
    ----------
    y_true : ndarray
    model_outputs : dict
        Mapping model label -> predicted cube.
    index : int
        Subcube index.
    save_dir : str
        Directory to save figure.
    log_transform : bool
        Apply log1p to data before analysis.
    bins : int
        Histogram bins.
    show : bool
        If True, display figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    if log_transform:
        true_flat = np.log1p(y_true).flatten()
        suffix = "log1p"
    else:
        true_flat = y_true.flatten()
        suffix = "linear"
    true_flat = true_flat[np.isfinite(true_flat)]

    plt.figure(figsize=(8, 6))
    plt.hist(true_flat, bins=bins, density=True, alpha=0.4, label=f"True ({suffix})")

    # normal fit to true
    mu, sigma = np.mean(true_flat), np.std(true_flat)
    x_vals = np.linspace(np.min(true_flat), np.max(true_flat), 300)
    normal_pdf = stats.norm.pdf(x_vals, loc=mu, scale=sigma)
    plt.plot(x_vals, normal_pdf, 'k--', label=f"Normal Fit (True, μ={mu:.2f}, σ={sigma:.2f})")

    for model_name, y_pred in model_outputs.items():
        if log_transform:
            pred_flat = np.log1p(y_pred).flatten()
        else:
            pred_flat = y_pred.flatten()
        pred_flat = pred_flat[np.isfinite(pred_flat)]
        plt.hist(pred_flat, bins=bins, density=True, alpha=0.4, label=f"{model_name} ({suffix})")

    plt.xlabel(f"Voxel Value ({suffix})")
    plt.ylabel("Probability Density")
    plt.title(f"Multi-Model Gaussianity Analysis (Subcube {index})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"unified_gaussianity_idx{index}_{suffix}.png")
    plt.savefig(save_path, dpi=150)
    logging.info(f"📈 Unified Gaussianity plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()


def compare_voxel_pdf_multi(
    y_true: np.ndarray,
    model_outputs: dict,
    index: int = None,
    save_dir: str = "./",
    log_transform: bool = True,
    show: bool = False,
):
    """
    Compare estimated voxel PDFs (KDE) across multiple models and ground truth.

    Parameters
    ----------
    y_true : ndarray
    model_outputs : dict
        Mapping model label -> predicted cube.
    index : int
        Subcube index.
    save_dir : str
        Output directory.
    log_transform : bool
        Apply log1p before KDE.
    show : bool
        Whether to display figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    if log_transform:
        true_flat = np.log1p(y_true).flatten()
        suffix = "log1p"
    else:
        true_flat = y_true.flatten()
        suffix = "linear"
    true_flat = true_flat[np.isfinite(true_flat)]
    kde_true = gaussian_kde(true_flat)

    # aggregate x range across true and all preds
    x_min, x_max = true_flat.min(), true_flat.max()
    for _, y_pred in model_outputs.items():
        if log_transform:
            pred_flat = np.log1p(y_pred).flatten()
        else:
            pred_flat = y_pred.flatten()
        pred_flat = pred_flat[np.isfinite(pred_flat)]
        x_min = min(x_min, pred_flat.min())
        x_max = max(x_max, pred_flat.max())

    x_vals = np.linspace(x_min, x_max, 500)
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, kde_true(x_vals), label=f"True ({suffix})", linewidth=2, color='tab:blue')

    for model_name, y_pred in model_outputs.items():
        if log_transform:
            pred_flat = np.log1p(y_pred).flatten()
        else:
            pred_flat = y_pred.flatten()
        pred_flat = pred_flat[np.isfinite(pred_flat)]
        kde_pred = gaussian_kde(pred_flat)
        plt.plot(x_vals, kde_pred(x_vals), label=f"{model_name} ({suffix})", linewidth=2, linestyle="--")

    plt.xlabel(f"Voxel Value ({suffix})")
    plt.ylabel("Estimated PDF")
    plt.title(f"Multi-Model PDF Comparison (Subcube {index})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"unified_pdf_comparison_idx{index}_{suffix}.png")
    plt.savefig(save_path, dpi=150)
    logging.info(f"📉 Unified PDF comparison saved to: {save_path}")
    if show:
        plt.show()
    plt.close()
