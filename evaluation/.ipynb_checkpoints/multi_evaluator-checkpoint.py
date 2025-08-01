import os
import h5py
import numpy as np
import logging

from nbodykit_analyzer import MultiDensityFieldAnalyzer
from multi_field_analysis_utils import (
    plot_loss_curve_multi,
    compare_density_distribution_multi,
    plot_projection_comparison_multi,
    plot_pixel_scatter_multi,
    analyze_gaussianity_multi,
    compare_voxel_pdf_multi,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class MultiEvaluator:
    """
    Unified evaluator that handles multiple model/epoch predictions and produces both
    per-model and combined (multi-model) analysis plots, including power spectrum analysis.

    Parameters
    ----------
    input_file : str
        Path to HDF5 file containing evolved input subcubes.
    target_file : str
        Path to HDF5 file containing ground truth initial condition subcubes.
    prediction_infos : list of dict
        Each dict needs:
            - 'file': path to prediction HDF5 file
            - 'label': label used in plots
            - 'log_file': optional path to training log CSV
    index : int
        Subcube index to evaluate.
    box_size : float
        Physical box size in Mpc/h.
    show : bool
        Whether to call plt.show() on plots.
    """

    def __init__(self, input_file, target_file, prediction_infos, index=0, box_size=50.0, show=False):
        self.input_file = input_file
        self.target_file = target_file
        self.prediction_infos = prediction_infos  # list of dicts with 'file','label', optional 'log_file'
        self.index = index
        self.box_size = box_size
        self.show = show

        self.input_cube = self._load_cube(self.input_file, self.index)
        self.gt_cube = self._load_cube(self.target_file, self.index)

        self.output_dir = f"multi_eval_Sample{index}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Prepare containers
        self.model_outputs = {}  # label -> pred cube
        self.log_files = {}      # label -> log path (if any)

        for info in self.prediction_infos:
            label = info["label"]
            pred_cube = self._load_cube(info["file"], self.index, pred=True)
            self.model_outputs[label] = pred_cube
            if "log_file" in info and info["log_file"] is not None:
                self.log_files[label] = info["log_file"]

    def _load_cube(self, file_path, index, pred=False):
        with h5py.File(file_path, "r") as f:
            key = list(f.keys())[0]
            data = f[key][index]
            if pred and data.ndim == 4:
                return np.array(data[0])
            return np.array(data)

    def plot_loss_curve(self, label, log_file=None, output_dir=None):
        """
        Plot loss curve for a specific model label. If log_file is provided, use it;
        otherwise fall back to stored self.log_files[label]. Saves into output_dir or
        per-label subdirectory by default.
        """
        chosen_log = log_file if log_file is not None else self.log_files.get(label)
        if chosen_log is None:
            logging.warning(f"No log file available for label '{label}'. Skipping loss curve.")
            return

        target_dir = output_dir if output_dir is not None else os.path.join(self.output_dir, label.replace(" ", "_"))
        os.makedirs(target_dir, exist_ok=True)
        logging.info(f"📉 Plotting loss curve for {label} from {chosen_log}")
        plot_loss_curve_multi({label: chosen_log}, output_dir=target_dir, show=self.show)
        
    def evaluate_all(self, mode="Unified"):
        logging.info("🔍 Starting full multi-model evaluation...")

        if mode=="individual":
            # 1. Per-model (individual) evaluation subdirectories
            for label, pred_cube in self.model_outputs.items():
                subdir = os.path.join(self.output_dir, label.replace(" ", "_"))
                os.makedirs(subdir, exist_ok=True)

                # Loss curve if exists
                if label in self.log_files:
                    logging.info(f"📉 Plotting loss curve for {label}")
                    plot_loss_curve_multi({label: self.log_files[label]}, output_dir=subdir, show=self.show)

                # Density distribution (individual)
                compare_density_distribution_multi(
                    y_true=self.gt_cube,
                    model_outputs={label: pred_cube},
                    save_dir=subdir,
                    index=self.index,
                    show=self.show
                )

                # Projection comparison (individual uses unified version with single model)
                plot_projection_comparison_multi(
                    input_cube=self.input_cube,
                    gt_cube=self.gt_cube,
                    model_outputs={label: pred_cube},
                    axis=0,
                    index=self.index,
                    save_dir=subdir,
                    show=self.show
                )

                # Pixel scatter
                plot_pixel_scatter_multi(
                    y_true=self.gt_cube,
                    model_outputs={label: pred_cube},
                    index=self.index,
                    save_dir=subdir,
                    log_transform=True,
                    show=self.show
                )

                # Multi-model density field analyzer invoked with single prediction for per-model
                analyzer = MultiDensityFieldAnalyzer(
                    true_density=self.gt_cube,
                    pred_dict={label: pred_cube},
                    box_size=self.box_size,
                    output_dir=subdir
                )
                analyzer.run_all(mode=mode)

                # Gaussianity
                analyze_gaussianity_multi(
                    y_true=self.gt_cube,
                    model_outputs={label: pred_cube},
                    index=self.index,
                    save_dir=subdir,
                    log_transform=True,
                    show=self.show
                )

                # PDF comparison
                compare_voxel_pdf_multi(
                    y_true=self.gt_cube,
                    model_outputs={label: pred_cube},
                    index=self.index,
                    save_dir=subdir,
                    log_transform=True,
                    show=self.show
                )

                logging.info(f"✅ Completed individual evaluation for {label}")
                
        elif mode=="Unified":
            # 2. Unified multi-model plots (all labels together)

            # Combined loss curves
            if self.log_files:
                logging.info("📉 Plotting combined loss curves")
                plot_loss_curve_multi(self.log_files, output_dir=self.output_dir, show=self.show)

            # Combined density distribution
            compare_density_distribution_multi(
                y_true=self.gt_cube,
                model_outputs=self.model_outputs,
                save_dir=self.output_dir,
                index=self.index,
                show=self.show
            )

            # Combined projection comparison
            plot_projection_comparison_multi(
                input_cube=self.input_cube,
                gt_cube=self.gt_cube,
                model_outputs=self.model_outputs,
                axis=0,
                index=self.index,
                save_dir=self.output_dir,
                show=self.show
            )

            # Combined pixel scatter
            plot_pixel_scatter_multi(
                y_true=self.gt_cube,
                model_outputs=self.model_outputs,
                index=self.index,
                save_dir=self.output_dir,
                log_transform=True,
                show=self.show
            )

            # Unified nbodykit analysis over all models
            multi_analyzer = MultiDensityFieldAnalyzer(
                true_density=self.gt_cube,
                pred_dict=self.model_outputs,
                box_size=self.box_size,
                output_dir=self.output_dir
            )
            multi_analyzer.run_all()

            # Combined Gaussianity
            analyze_gaussianity_multi(
                y_true=self.gt_cube,
                model_outputs=self.model_outputs,
                index=self.index,
                save_dir=self.output_dir,
                log_transform=True,
                show=self.show
            )

            # Combined PDF
            compare_voxel_pdf_multi(
                y_true=self.gt_cube,
                model_outputs=self.model_outputs,
                index=self.index,
                save_dir=self.output_dir,
                log_transform=True,
                show=self.show
            )

            logging.info(f"📁 All evaluation results saved under {self.output_dir}")
            
        else:
            print("wrong mode!")