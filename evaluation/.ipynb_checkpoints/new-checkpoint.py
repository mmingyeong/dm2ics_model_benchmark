import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import logging

# ===== Configure logging =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===== File paths =====
input_file = "/caefs/data/IllustrisTNG/subcube/input/subcubes_stride4_50mpc_010.h5"
target_file = "/caefs/data/IllustrisTNG/subcube/output/subcubes_stride4_50mpc_010.h5"
unet_pred_file = "/caefs/data/IllustrisTNG/predictions/unet/Sample100_epoch100/subcubes_stride4_50mpc_010.h5"
fno_pred_file  = "/caefs/data/IllustrisTNG/predictions/fno/Sample100_epoch100/subcubes_stride4_50mpc_010.h5"

# ===== Data loading =====
def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    with h5py.File(file_path, "r") as f:
        data = f["subcubes"]
        shape = data.shape
        logger.info(f"Raw dataset shape: {shape}")

        if len(shape) == 4:
            logger.info("Detected shape (N, D, H, W) – using directly.")
            return data[:]
        elif len(shape) == 3:
            logger.warning("Detected 2D shape (N, H, W) – stacking to (N, D, H, W) with D=1.")
            return np.expand_dims(data[:], axis=1)
        elif len(shape) == 5 and shape[1] == 1:
            logger.info("Detected shape (N, 1, D, H, W) – removing channel dimension.")
            return data[:, 0, :, :, :]
        else:
            raise ValueError(f"Unsupported data shape: {shape}")

# ===== Load subcube data =====
gt_data = load_data(target_file)
unet_data = load_data(unet_pred_file)
fno_data = load_data(fno_pred_file)

# ===== Settings =====
sample_voxels = 100_000
log_bins = np.linspace(-5, 5, 200)
centers = 0.5 * (log_bins[:-1] + log_bins[1:])

# ===== Joint Distribution =====
gt_sample = gt_data.reshape(-1)
unet_sample = unet_data.reshape(-1)
fno_sample = fno_data.reshape(-1)
idx = np.random.choice(len(gt_sample), size=sample_voxels, replace=False)
bins = np.logspace(-5, 5, 200)
H_unet, xedges, yedges = np.histogram2d(gt_sample[idx], unet_sample[idx], bins=[bins, bins], density=True)
H_fno, _, _ = np.histogram2d(gt_sample[idx], fno_sample[idx], bins=[bins, bins], density=True)

# ===== Histogram (log10) stats =====
def log_histogram_stats(data, bins):
    hists = []
    for i in range(data.shape[0]):
        vals = np.log10(data[i].flatten())
        hist, _ = np.histogram(vals, bins=bins, density=True)
        hists.append(hist)
    hists = np.array(hists)
    return np.mean(hists, axis=0), np.std(hists, axis=0)

gt_hist_mean, gt_hist_std = log_histogram_stats(gt_data, log_bins)
unet_hist_mean, unet_hist_std = log_histogram_stats(unet_data, log_bins)
fno_hist_mean, fno_hist_std = log_histogram_stats(fno_data, log_bins)

# ===== Correlation =====
def compute_correlation(data):
    delta = data / np.mean(data) - 1
    flat = delta.flatten()
    corr = fftconvolve(flat, flat[::-1], mode="full")
    corr = corr[corr.size // 2:]
    corr /= corr[0]
    r = np.arange(len(corr))
    return r[1:50], corr[1:50]

def correlation_stats(data):
    corrs = []
    for i in range(data.shape[0]):
        r, c = compute_correlation(data[i])
        corrs.append(c)
    corrs = np.array(corrs)
    return r, np.median(corrs, axis=0), np.std(corrs, axis=0)

r, gt_corr_median, gt_corr_std = correlation_stats(gt_data)
_, unet_corr_median, unet_corr_std = correlation_stats(unet_data)
_, fno_corr_median, fno_corr_std = correlation_stats(fno_data)

# ===== Plotting =====
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# --- Joint Distribution ---
for i, (H, label) in enumerate(zip([H_unet, H_fno], ["U-Net", "FNO"])):
    ax = axs[i, 0]
    im = ax.imshow(np.log10(H + 1e-10), origin="lower",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   aspect="auto", cmap="viridis")
    ax.plot([1e-5, 1e5], [1e-5, 1e5], "k--", linewidth=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-5, 1e5); ax.set_ylim(1e-5, 1e5)
    ax.set_title(label); ax.set_xlabel(r"$\rho_{\mathrm{truth}} / \rho_0$")
    ax.set_ylabel(r"$\rho_{\mathrm{pred}} / \rho_0$")

# --- Histogram (log10) ---
for i, (mean, std, label) in enumerate(zip(
        [unet_hist_mean, fno_hist_mean],
        [unet_hist_std, fno_hist_std],
        ["U-Net", "FNO"])):
    ax = axs[i, 1]
    ax.plot(centers, gt_hist_mean, color="black", label="Truth")
    ax.plot(centers, mean, color="red", label="Prediction")
    ax.fill_between(centers, mean - std, mean + std, color="red", alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlim(-5, 5)
    ax.set_xlabel(r"$\log_{10}(\rho / \rho_0)$")
    ax.set_ylabel(r"$df / d\log_{10} \rho$")
    ax.set_title(label)
    ax.legend()

# --- Correlation Function ---
for i, (med, std, label) in enumerate(zip(
        [unet_corr_median, fno_corr_median],
        [unet_corr_std, fno_corr_std],
        ["U-Net", "FNO"])):
    ax = axs[i, 2]
    ax.plot(r, gt_corr_median, color="black", label="Truth")
    ax.plot(r, med, color="red", label="Prediction")
    ax.fill_between(r, med - std, med + std, color="red", alpha=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r\ [h^{-1}\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\langle \delta(x)\delta(x+r) \rangle$")
    ax.set_title(label)
    ax.legend()

plt.tight_layout()
plt.savefig("figure5_style_comparison.png", dpi=300)
plt.show()
