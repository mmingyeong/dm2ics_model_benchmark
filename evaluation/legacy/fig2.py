import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.signal import fftconvolve
import logging
import warnings

# ===== Configure logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ===== Suppress warnings =====
warnings.filterwarnings("ignore")

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

        # Case 1: (N, D, H, W) – already correct shape
        if len(shape) == 4:
            logger.info("Detected shape (N, D, H, W) – using directly.")
            return data[:]

        # Case 2: (N, H, W) – 2D projection, expand D=1
        elif len(shape) == 3:
            logger.warning("Detected 2D shape (N, H, W) – stacking to (N, D, H, W) with D=1.")
            return np.expand_dims(data[:], axis=1)

        # Case 3: (N, 1, D, H, W) – remove channel dimension
        elif len(shape) == 5 and shape[1] == 1:
            logger.info("Detected shape (N, 1, D, H, W) – removing channel dimension.")
            return data[:, 0, :, :, :]

        # Case 4: Unexpected shape
        else:
            raise ValueError(f"Unsupported data shape: {shape}")


gt = load_data(target_file)[0]
unet_pred = load_data(unet_pred_file)[0]
fno_pred = load_data(fno_pred_file)[0]

def print_density_stats(name, raw_data):
    raw_min = np.min(raw_data)
    raw_max = np.max(raw_data)
    raw_mean = np.mean(raw_data)
    raw_std = np.std(raw_data)

    density = compute_density(raw_data)
    den_min = np.min(density)
    den_max = np.max(density)
    den_mean = np.mean(density)
    den_std = np.std(density)

    logger.info(f"[{name}] Raw  → min: {raw_min:.4e}, max: {raw_max:.4e}, mean: {raw_mean:.4e}, std: {raw_std:.4e}")
    logger.info(f"[{name}] Density → min: {den_min:.4e}, max: {den_max:.4e}, mean: {den_mean:.4e}, std: {den_std:.4e}")

# ===== Density transformation =====
def compute_density(data):
    # Identity function: already linear density
    return data


# ==== 통계 디버깅 ====
print_density_stats("Ground Truth", gt)
print_density_stats("U-Net", unet_pred)
print_density_stats("FNO", fno_pred)


def match_shape(truth, pred):
    logger.debug(f"Original shapes: truth={truth.shape}, pred={pred.shape}")
    min_shape = tuple(min(t, p) for t, p in zip(truth.shape, pred.shape))
    logger.debug(f"Matching to min shape: {min_shape}")
    slices = tuple(slice(0, s) for s in min_shape)
    return truth[slices], pred[slices]

# ===== 2D histogram =====
def compute_2d_histogram(x, y, bins=200):
    logger.debug("Computing 2D histogram")
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    H = H.T
    flat = H.flatten()
    flat = flat[flat > 0]
    levels = np.percentile(flat, [50, 90, 99])
    logger.debug(f"Histogram levels: {levels}")
    return H, xedges, yedges, levels

def plot_joint(ax, truth, pred, label):
    logger.info(f"Plotting joint distribution: {label}")
    truth, pred = match_shape(truth, pred)  # 먼저 shape 맞추기
    x = compute_density(truth).flatten()    # 이후 density 변환
    y = compute_density(pred).flatten()
    assert x.shape == y.shape, f"x and y must be same length. Got {x.shape} and {y.shape}"
    H, xedges, yedges, levels = compute_2d_histogram(x, y)
    H_log = np.log10(H + 1e-10)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(H_log, extent=extent, origin="lower", cmap="viridis", aspect="auto")
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    ax.contour(X, Y, H, levels=levels, colors=["orange", "orangered", "red"], linewidths=1.2)
    ax.plot([1e-5, 1e5], [1e-5, 1e5], 'k--', linewidth=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-5, 1e5); ax.set_ylim(1e-5, 1e5)
    ax.set_xlabel(r"$\rho_{\mathrm{truth}} / \rho_0$")
    ax.set_ylabel(r"$\rho_{\mathrm{pred}} / \rho_0$")
    ax.set_title(label, loc="left")


# ===== Correlation computation =====
def compute_correlation(data):
    logger.debug("Computing correlation function")
    delta = data / np.mean(data) - 1
    flat = delta.flatten()
    corr = fftconvolve(flat, flat[::-1], mode="full")
    corr = corr[corr.size // 2:]
    corr /= corr[0]
    r = np.arange(len(corr))
    return r[1:50], corr[1:50]

# ===== Histogram plot =====
def plot_histogram(ax, truth, pred, label):
    logger.info(f"Plotting histogram: {label}")
    truth, pred = match_shape(truth, pred)
    bins = np.linspace(-5, 5, 200)
    gt_log = np.log10(compute_density(truth).flatten())
    pr_log = np.log10(compute_density(pred).flatten())
    gt_hist, _ = np.histogram(gt_log, bins=bins, density=True)
    pr_hist, _ = np.histogram(pr_log, bins=bins, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(centers, gt_hist, color="black", label="Truth")
    ax.plot(centers, pr_hist, color="red", label="Prediction")
    ax.fill_between(centers, pr_hist * 0.9, pr_hist * 1.1, color="red", alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlim(-5, 5)
    ax.set_xlabel(r"$\log_{10}(\rho / \rho_0)$")
    ax.set_ylabel(r"$df / d\log_{10} \rho$")
    ax.set_title(label, loc="left")
    ax.legend()

# ===== Correlation plot =====
def plot_correlation(ax, truth, pred, label):
    logger.info(f"Plotting correlation: {label}")
    truth, pred = match_shape(truth, pred)
    r, gt_corr = compute_correlation(compute_density(truth))
    _, pr_corr = compute_correlation(compute_density(pred))
    ax.plot(r, gt_corr, color="black", label="Truth")
    ax.plot(r, pr_corr, color="red", label="Prediction")
    ax.fill_between(r, pr_corr * 0.9, pr_corr * 1.1, color="red", alpha=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r\ [h^{-1}\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\langle \delta(x)\delta(x+r) \rangle$")
    ax.set_title(label, loc="left")
    ax.legend()

# ===== Main plot block =====
logger.info("Generating figure...")
fig, axs = plt.subplots(2, 3, figsize=(15, 9))

plot_joint(axs[0, 0], gt, unet_pred, "U-Net")
plot_histogram(axs[0, 1], gt, unet_pred, "U-Net")
plot_correlation(axs[0, 2], gt, unet_pred, "U-Net")

plot_joint(axs[1, 0], gt, fno_pred, "FNO")
plot_histogram(axs[1, 1], gt, fno_pred, "FNO")
plot_correlation(axs[1, 2], gt, fno_pred, "FNO")

plt.tight_layout()
fig_path = "statistical_comparison_unet_fno.png"
plt.savefig(fig_path, dpi=300)
logger.info(f"Figure saved to {fig_path}")
plt.show()
