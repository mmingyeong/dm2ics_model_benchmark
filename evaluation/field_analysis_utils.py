import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import scipy.stats as stats
from scipy.stats import gaussian_kde, binned_statistic, skew, kurtosis


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def plot_loss_curve(log_csv_path, output_dir=None, show=False):
    """
    Plot training and validation loss curves from CSV log.

    Parameters
    ----------
    log_csv_path : str
        Path to CSV log file with columns 'epoch', 'train_loss', and optionally 'val_loss'.
    output_dir : str, optional
        Directory to save the output plot.
    show : bool, optional
        If True, show plot inline.
    """
    if not os.path.exists(log_csv_path):
        logging.warning(f"Loss curve skipped: file not found -> {log_csv_path}")
        return

    df = pd.read_csv(log_csv_path)
    if 'epoch' not in df.columns or 'train_loss' not in df.columns:
        logging.warning("Loss curve skipped: missing 'epoch' or 'train_loss' in CSV.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', linestyle='--', marker='s')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(save_path)
        logging.info(f"📉 Loss curve saved to {save_path}")
    if show:
        plt.show()
    plt.close()


def compare_density_distribution(
    y_true,
    y_pred,
    model_name="Model",
    save_dir="./",
    index=None,
    show=False,
    nbins=50,
    bin_range=(0.5, 20),
    bins=None,
    plot_mean_only=True  # 이 예제에선 아직 미사용
):
    """
    Compare the 1 + δ density distribution of true and predicted 3D overdensity fields.

    Parameters
    ----------
    y_true : ndarray
        Ground truth 3D overdensity field (δ).
    y_pred : ndarray
        Predicted 3D overdensity field (δ).
    model_name : str
        Name of the model for labeling.
    save_dir : str
        Directory to save output.
    index : int
        Subcube index.
    show : bool
        Whether to display plot.
    nbins : int
        Number of bins in log(1 + δ) space.
    bin_range : tuple of float
        Min/max for 1 + δ for histogram binning (e.g., (0.5, 20)).
    bins : ndarray or None
        Custom bin edges in 1 + δ space (overrides nbins and bin_range if provided).
    plot_mean_only : bool
        Whether to plot just the comparison for one sample (True) or support ensemble (future extension).
    """

    # Flatten and shift: δ → 1 + δ
    one_plus_true = 1 + y_true.flatten()
    one_plus_pred = 1 + y_pred.flatten()

    # Remove invalid (NaN, Inf, ≤ 0)
    mask = (
        np.isfinite(one_plus_true) & np.isfinite(one_plus_pred) &
        (one_plus_true > 0) & (one_plus_pred > 0)
    )
    one_plus_true = one_plus_true[mask]
    one_plus_pred = one_plus_pred[mask]

    # Use log binning
    if bins is None:
        bins = np.logspace(np.log10(bin_range[0]), np.log10(bin_range[1]), nbins)

    # Histogram and normalize to get PDF
    hist_true, _ = np.histogram(one_plus_true, bins=bins, density=True)
    hist_pred, _ = np.histogram(one_plus_pred, bins=bins, density=True)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, hist_true, label="True", color='black', linewidth=2)
    plt.plot(bin_centers, hist_pred, label="Predicted", linestyle='--', color='tab:orange', linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$1 + \delta$")
    plt.ylabel("PDF (normalized)")
    plt.title(f"(1+δ) Distribution Comparison (Subcube {index})")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{model_name.lower()}_delta_distribution_idx{index}.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=150)
    logging.info(f"📈 1+δ distribution comparison saved to: {fpath}")

    if show:
        plt.show()
    else:
        plt.close()


    
def plot_projection_comparison(input_cube, gt_cube, pred_cube, axis=0, index=None,
                               model_name="Model", save_dir=None, show=False):
    """
    Plot 2D projection of 3D cubes (input, ground truth, prediction).

    Parameters
    ----------
    input_cube, gt_cube, pred_cube : ndarray
        3D density cubes.
    axis : int
        Axis along which to project (0, 1, or 2).
    index : int or None
        Index for annotation (e.g., subcube number).
    model_name : str
        Name of the model for labeling.
    save_dir : str, optional
        Directory to save figure.
    show : bool, optional
        If True, show plot inline.
    """
    input_proj = np.log1p(np.sum(input_cube, axis=axis))
    gt_proj = np.log1p(np.sum(gt_cube, axis=axis))
    pred_proj = np.log1p(np.sum(pred_cube, axis=axis))

    projections = [input_proj, gt_proj, pred_proj]
    titles = ["Input (log1p)", "Ground Truth (log1p)", f"{model_name} Prediction (log1p)"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, (proj, title) in enumerate(zip(projections, titles)):
        im = axs[i].imshow(proj, origin="lower", cmap="viridis")
        axs[i].set_title(title, fontsize=13)
        axs[i].set_xlabel("X [voxel] (1 voxel = 0.82 Mpc/h)")
        axs[i].set_ylabel("Y [voxel] (1 voxel = 0.82 Mpc/h)")
        cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        cbar.set_label("log(1 + Projected Density)", fontsize=10)

    info = f"Projection axis = {axis} | Subcube index = {index} | Model = {model_name}"
    plt.suptitle(info, fontsize=15, y=1.05)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name.lower()}_projection_idx{index}_axis{axis}.png")
        plt.savefig(path)
        logging.info(f"🖼️  Projection comparison saved to {path}")
    if show:
        plt.show()
    plt.close()


def plot_pixel_scatter(y_true, y_pred, index=None, model_name="Model", save_dir=None, log_transform=True, show=False):
    """
    Scatter plot of predicted vs. ground truth voxel values.

    Parameters
    ----------
    y_true, y_pred : ndarray
        3D voxel fields.
    log_transform : bool
        Whether to apply log1p before comparison.
    show : bool
        If True, show plot inline.
    """
    if log_transform:
        y_true = np.log1p(y_true)
        y_pred = np.log1p(y_pred)

    true_flat = y_true.flatten()
    pred_flat = y_pred.flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(true_flat, pred_flat, s=1, alpha=0.3, color='royalblue')
    plt.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--')
    plt.xlabel("Ground Truth (log1p)")
    plt.ylabel("Prediction (log1p)")
    plt.title(f"Pixel-to-Pixel Comparison\nModel: {model_name}, Index: {index}")
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name.lower()}_scatter_idx{index}.png")
        plt.savefig(path)
        logging.info(f"📊 Pixel scatter plot saved to {path}")
    if show:
        plt.show()
    plt.close()

def analyze_gaussianity(y_true, y_pred, index=None, model_name="Model",
                        save_dir="./", log_transform=True, bins=100, show=False):
    """
    Compare histogram, skewness, and kurtosis between true and predicted 3D voxel fields,
    including a Gaussian fit for reference.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth 3D field.
    y_pred : np.ndarray
        Predicted 3D field.
    index : int or None
        Subcube index for labeling.
    model_name : str
        Model name for labeling the figure.
    save_dir : str
        Directory to save output figure.
    log_transform : bool
        Whether to apply log1p transformation to input fields.
    bins : int
        Number of histogram bins.
    show : bool
        If True, show the plot interactively.
    """
    # NaN-safe flattening with optional log transform
    if log_transform:
        y_true = np.log1p(y_true)
        y_pred = np.log1p(y_pred)
        suffix = "log1p"
    else:
        suffix = "linear"

    # Flatten and filter NaNs/Infs
    true_flat = y_true.flatten()
    pred_flat = y_pred.flatten()
    valid_mask = np.isfinite(true_flat) & np.isfinite(pred_flat)
    true_flat = true_flat[valid_mask]
    pred_flat = pred_flat[valid_mask]

    # Statistical moments
    true_skew = stats.skew(true_flat, bias=False)
    true_kurt = stats.kurtosis(true_flat, bias=False)
    pred_skew = stats.skew(pred_flat, bias=False)
    pred_kurt = stats.kurtosis(pred_flat, bias=False)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(true_flat, bins=bins, density=True, alpha=0.6, label=f"True ({suffix})")
    plt.hist(pred_flat, bins=bins, density=True, alpha=0.6, label=f"Predicted ({suffix})")

    # Normal distribution fit to true
    x_vals = np.linspace(min(true_flat.min(), pred_flat.min()),
                         max(true_flat.max(), pred_flat.max()), 300)
    mu, sigma = np.mean(true_flat), np.std(true_flat)
    normal_pdf = stats.norm.pdf(x_vals, loc=mu, scale=sigma)
    plt.plot(x_vals, normal_pdf, 'k--', label=f'Normal Fit (True, μ={mu:.2f}, σ={sigma:.2f})')

    # Labels and layout
    plt.xlabel(f"Voxel Value ({suffix})")
    plt.ylabel("Probability Density")
    plt.title(f"Gaussianity Analysis (Subcube {index})\n"
              f"[Skewness] True: {true_skew:.2f}, Pred: {pred_skew:.2f} | "
              f"[Kurtosis] True: {true_kurt:.2f}, Pred: {pred_kurt:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name.lower()}_gaussianity_idx{index}_{suffix}.png"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    logging.info(f"📈 Gaussianity plot saved to {path}")

    if show:
        plt.show()
    else:
        plt.close()



def compare_voxel_pdf(y_true, y_pred, index=None, model_name="Model",
                      save_dir="./", log_transform=True, bins=100, show=False):
    """
    Compare the estimated probability density functions (PDF) of voxel values between
    the predicted and true 3D density fields using Gaussian KDE.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth 3D voxel field.
    y_pred : np.ndarray
        Predicted 3D voxel field.
    index : int or None
        Subcube index for labeling.
    model_name : str
        Model name to annotate plot and filename.
    save_dir : str
        Directory to save the plot.
    log_transform : bool
        If True, apply log1p transformation to input fields.
    bins : int
        Number of bins (not used here, but retained for interface consistency).
    show : bool
        If True, show the plot inline.
    """
    # Apply log1p if needed
    if log_transform:
        y_true = np.log1p(y_true)
        y_pred = np.log1p(y_pred)
        suffix = "log1p"
    else:
        suffix = "linear"

    # Flatten and NaN-safe
    true_flat = y_true.flatten()
    pred_flat = y_pred.flatten()
    valid_mask = np.isfinite(true_flat) & np.isfinite(pred_flat)
    true_flat = true_flat[valid_mask]
    pred_flat = pred_flat[valid_mask]

    # KDE
    kde_true = gaussian_kde(true_flat)
    kde_pred = gaussian_kde(pred_flat)

    x_min = min(true_flat.min(), pred_flat.min())
    x_max = max(true_flat.max(), pred_flat.max())
    x_vals = np.linspace(x_min, x_max, 500)

    pdf_true = kde_true(x_vals)
    pdf_pred = kde_pred(x_vals)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, pdf_true, label=f"True ({suffix})", linewidth=2, color='tab:blue')
    plt.plot(x_vals, pdf_pred, label=f"Predicted ({suffix})", linewidth=2, linestyle="--", color='tab:orange')
    plt.fill_between(x_vals, pdf_true, alpha=0.2, color='tab:blue')
    plt.fill_between(x_vals, pdf_pred, alpha=0.2, color='tab:orange')

    plt.xlabel(f"Voxel Value ({suffix})")
    plt.ylabel("Estimated PDF")
    plt.title(f"PDF Comparison (Subcube {index})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name.lower()}_pdf_comparison_idx{index}_{suffix}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    logging.info(f"📉 PDF comparison saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
