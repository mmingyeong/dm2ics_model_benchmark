# Ref: https://arxiv.org/pdf/2502.03139
# APPENDIX A: SUMMARY STATISTICS

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import time
from scipy.fft import fftn, fftfreq
from scipy.stats import binned_statistic


class PowerSpectrumAnalyzer:
    def __init__(self, delta_true, delta_pred, box_size, nbins=100, output_dir="analysis_results"):
        """
        Parameters
        ----------
        delta_true : ndarray
            Ground truth overdensity field.
        delta_pred : ndarray
            Predicted overdensity field.
        box_size : float
            Box size in Mpc/h.
        nbins : int
            Number of k bins.
        output_dir : str
            Output directory to save results.
        """
        assert delta_true.shape == delta_pred.shape, "Shapes of input fields must match"
        self.delta_true = delta_true
        self.delta_pred = delta_pred
        self.box_size = box_size
        self.nbins = nbins
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    def _compute_power_spectrum(self, delta):
        N = delta.shape[0]
        dx = self.box_size / N
        volume = self.box_size ** 3

        delta_k = fftn(delta)
        power_k = (volume / N**6) * np.abs(delta_k)**2

        kfreq = fftfreq(N, d=dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
        p_flat = power_k.flatten()

        kmin = k_mag[k_mag > 0].min()
        kmax = k_mag.max()
        k_edges = np.logspace(np.log10(kmin), np.log10(kmax), self.nbins + 1)
        k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
        Pk_mean, _, _ = binned_statistic(k_mag, p_flat, bins=k_edges, statistic='mean')
        delta2_k = (k_centers ** 3) * Pk_mean / (2 * np.pi ** 2)

        return k_centers, Pk_mean, delta2_k

    def run_analysis(self):
        # Step 1: Power spectrum 계산
        logging.info("Computing P(k) for true field")
        self.k, self.P_true, self.delta2_true = self._compute_power_spectrum(self.delta_true)

        logging.info("Computing P(k) for predicted field")
        _, self.P_pred, self.delta2_pred = self._compute_power_spectrum(self.delta_pred)

        # Step 2: Cross-spectrum 계산
        logging.info("Computing cross-spectrum")
        N = self.delta_true.shape[0]
        dx = self.box_size / N
        volume = self.box_size ** 3

        delta_true_k = fftn(self.delta_true)
        delta_pred_k = fftn(self.delta_pred)

        cross_k = delta_true_k * np.conj(delta_pred_k)
        cross_power = (volume / N**6) * np.real(cross_k)

        kfreq = fftfreq(N, d=dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
        cross_flat = cross_power.flatten()

        k_edges = np.logspace(np.log10(k_mag[k_mag > 0].min()), np.log10(k_mag.max()), self.nbins + 1)
        P_cross, _, _ = binned_statistic(k_mag, cross_flat, bins=k_edges, statistic='mean')

        # Step 3: Transfer & Cross-Correlation
        self.T_k = self.P_pred / (self.P_true + 1e-10)
        self.T_k = self.P_true / (self.P_pred + 1e-10)  # if you want T_true→pred instead
        self.C_k = P_cross / (np.sqrt(self.P_true * self.P_pred) + 1e-10)

        logging.info(f"Max P_true: {self.P_true.max():.2e}, Max P_pred: {self.P_pred.max():.2e}")
        logging.info(f"Max P_cross: {P_cross.max():.2e}")

        # Step 4: 시각화
        self._plot_power()
        self._plot_transfer()

    def _plot_power(self):
        plt.figure(figsize=(8, 6))
        plt.loglog(self.k, self.delta2_true, label='True Δ²(k)')
        plt.loglog(self.k, self.delta2_pred, label='Predicted Δ²(k)')
        plt.xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        plt.ylabel(r'$\Delta^2(k)$')
        plt.title("Dimensionless Power Spectrum")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.output_dir, "delta2k_comparison.png"))
        plt.close()

    def _plot_transfer(self):
        # Transfer Function T(k)
        plt.figure(figsize=(8, 5))
        plt.semilogx(self.k, self.T_k, label=r'Transfer Function $T(k)$', color='tab:blue')
        plt.xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        plt.ylabel(r'$T(k)$')
        plt.title("Transfer Function")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.output_dir, "transfer_function.png"))
        plt.close()

        # Cross-Correlation C(k)
        plt.figure(figsize=(8, 5))
        plt.semilogx(self.k, self.C_k, label=r'Cross-Correlation $C(k)$', color='tab:orange')
        plt.xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        plt.ylabel(r'$C(k)$')
        plt.title("Cross-Correlation Coefficient")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.output_dir, "cross_correlation.png"))
        plt.close()
