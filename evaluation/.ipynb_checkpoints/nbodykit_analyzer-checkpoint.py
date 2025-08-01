import os
import numpy as np
import matplotlib.pyplot as plt
from nbodykit.lab import ArrayMesh, FFTPower
from nbodykit import setup_logging
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MultiDensityFieldAnalyzer:
    """
    Multi-model/epoch analyzer for 3D density fields using nbodykit.

    Compares multiple predicted density fields against a common ground truth by
    computing and plotting P(k), transfer functions T(k), and correlation coefficients C(k).

    Parameters
    ----------
    true_density : ndarray
        Ground truth density field (3D array).
    pred_dict : dict
        Mapping from label (e.g., model name) to predicted density field (3D array).
    box_size : float
        Physical box size in Mpc/h.
    output_dir : str
        Directory where plots are saved.
    """

    def __init__(self, true_density, pred_dict: dict, box_size: float, output_dir="analysis_results"):
        setup_logging()
        self.true_density = np.array(true_density)
        self.pred_dict = {label: np.array(pred) for label, pred in pred_dict.items()}
        self.box_size = box_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Will store computed spectra per model
        self.results = {}  # label -> dict with k, P_true, P_pred, P_cross, T_k, C_k

    def _compute_single(self, label, delta_true, delta_pred):
        """
        Compute spectra and summary statistics for one predicted field.
        """
        mesh_true = ArrayMesh(delta_true, BoxSize=self.box_size)
        mesh_pred = ArrayMesh(delta_pred, BoxSize=self.box_size)

        pk_true = FFTPower(mesh_true, mode="1d").power
        pk_pred = FFTPower(mesh_pred, mode="1d").power
        pk_cross = FFTPower(mesh_true, second=mesh_pred, mode="1d").power

        k = pk_true["k"]
        P_true = pk_true["power"].real
        P_pred = pk_pred["power"].real
        P_cross = pk_cross["power"].real

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            T_k = np.where(P_true > 0, P_pred / P_true, 0.0)
            C_k = np.where((P_true > 0) & (P_pred > 0), P_cross / np.sqrt(P_true * P_pred), 0.0)

        self.results[label] = {
            "k": k,
            "P_true": P_true,
            "P_pred": P_pred,
            "P_cross": P_cross,
            "T_k": T_k,
            "C_k": C_k,
        }

    def run_all(self, mode="Unified"):
        """
        Compute all spectra for every prediction and produce unified & per-model plots.
        """
        logging.info("🔍 Computing multi-model density field statistics...")
        
        # First compute for each model
        # True field normalized to overdensity if needed? Assumes input is delta (overdensity).
        for label, pred in self.pred_dict.items():
            self._compute_single(label, self.true_density, pred)
        
        if mode=="Unified":
            # Unified / comparison plots
            self.plot_power_spectra_unified()
            self.plot_transfer_functions_unified()
            self.plot_correlation_coefficients_unified()

        elif mode=="individual":
            # Also save per-model individual versions
            for label in self.results:
                self._save_individual_plots(label)
        else:
            print("wrong mode!")
            return
        
        logging.info(f"✅ All multi-model analysis completed. Results saved to: {self.output_dir}")

    def plot_power_spectra_unified(self):
        plt.figure(figsize=(8, 6))
        # True P(k) plotted once
        any_label = next(iter(self.results))
        k = self.results[any_label]["k"]
        P_true = self.results[any_label]["P_true"]
        plt.loglog(k, P_true, label="True P(k)", color="black", linewidth=2)

        for label, res in self.results.items():
            plt.loglog(res["k"], res["P_pred"], label=f"{label} P_pred(k)", linestyle="--")
            plt.loglog(res["k"], res["P_cross"], label=f"{label} Cross P(k)", alpha=0.6)

        plt.xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        plt.ylabel(r"$P(k)$")
        plt.title("Unified Power Spectrum Comparison") 
        plt.legend(fontsize="small", ncol=1)
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "unified_power_spectrum_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logging.info(f"📊 Saved unified power spectrum to: {path}")

    def plot_transfer_functions_unified(self):
        plt.figure(figsize=(8, 5))
        for label, res in self.results.items():
            plt.semilogx(res["k"], res["T_k"], label=f"{label} T(k)")
        plt.xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        plt.ylabel(r"$T(k) = P_{\mathrm{pred}} / P_{\mathrm{true}}$")
        plt.title("Unified Transfer Functions") 
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.output_dir, "unified_transfer_function.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logging.info(f"📈 Saved unified transfer function to: {path}")

    def plot_correlation_coefficients_unified(self):
        plt.figure(figsize=(8, 5))
        for label, res in self.results.items():
            plt.semilogx(res["k"], res["C_k"], label=f"{label} C(k)")
        plt.xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        plt.ylabel(r"$C(k) = \frac{P_{\mathrm{cross}}}{\sqrt{P_{\mathrm{true}} P_{\mathrm{pred}}}}$")
        plt.title("Unified Cross-Correlation Coefficients") 
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.output_dir, "unified_correlation_coefficients.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logging.info(f"🔗 Saved unified correlation coefficient to: {path}")

    def _save_individual_plots(self, label):
        res = self.results[label]
        # Individual power spectrum
        plt.figure(figsize=(8, 6))
        plt.loglog(res["k"], res["P_true"], label="True P(k)", color="black", linewidth=2)
        plt.loglog(res["k"], res["P_pred"], label=f"{label} P_pred(k)", linestyle="--")
        plt.loglog(res["k"], res["P_cross"], label=f"{label} Cross P(k)", alpha=0.7)
        plt.xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        plt.ylabel(r"$P(k)$")
        plt.title(f"Power Spectrum Comparison ({label})")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.tight_layout()
        path_pk = os.path.join(self.output_dir, f"{label}_power_spectrum.png")
        plt.savefig(path_pk, dpi=150)
        plt.close()
        logging.info(f"📊 Saved individual power spectrum for {label} to: {path_pk}")

        # Transfer function
        plt.figure(figsize=(8, 5))
        plt.semilogx(res["k"], res["T_k"], label=f"{label} T(k)", color="darkorange")
        plt.xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        plt.ylabel(r"$T(k)$")
        plt.title(f"Transfer Function ({label})")
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        path_t = os.path.join(self.output_dir, f"{label}_transfer_function.png")
        plt.savefig(path_t, dpi=150)
        plt.close()
        logging.info(f"📈 Saved individual transfer function for {label} to: {path_t}")

        # Correlation coefficient
        plt.figure(figsize=(8, 5))
        plt.semilogx(res["k"], res["C_k"], label=f"{label} C(k)", color="royalblue")
        plt.xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        plt.ylabel(r"$C(k)$")
        plt.title(f"Cross-Correlation Coefficient ({label})")
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        path_c = os.path.join(self.output_dir, f"{label}_correlation_coefficient.png")
        plt.savefig(path_c, dpi=150)
        plt.close()
        logging.info(f"🔗 Saved individual correlation coefficient for {label} to: {path_c}")
