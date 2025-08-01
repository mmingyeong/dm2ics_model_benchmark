import os
import re
import h5py
import numpy as np
import logging

from nbodykit_analyzer import DensityFieldAnalyzer
from field_analysis_utils import (
    plot_loss_curve,
    compare_density_distribution,
    plot_projection_comparison,
    plot_pixel_scatter,
    analyze_gaussianity,
    compare_voxel_pdf
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class Evaluator:
    """
    Unified evaluation class for 3D density field predictions.

    Parameters
    ----------
    input_file : str
        Path to HDF5 file containing evolved input subcubes.
    target_file : str
        Path to HDF5 file containing ground truth initial condition subcubes.
    prediction_file : str
        Path to predicted subcubes (HDF5 format). Must include model/sample/epoch.
    log_file : str or None
        Path to training log CSV (optional).
    index : int
        Subcube index to evaluate.
    box_size : float
        Size of the subcube in Mpc/h.
    show : bool
        If True, show all plots with plt.show().
    """

    def __init__(self, input_file, target_file, prediction_file, log_file=None,
                 index=0, box_size=50.0, show=False):
        self.input_file = input_file
        self.target_file = target_file
        self.prediction_file = prediction_file
        self.log_file = log_file
        self.index = index
        self.box_size = box_size
        self.show = show  # <-- 추가됨

        # Extract model/sample/epoch from prediction path
        self.model_name, self.sample_name, self.epoch = self._parse_prediction_path(prediction_file)

        # Output directory
        self.output_dir = f"{self.model_name}_{self.sample_name}_{self.epoch}_results"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data
        self.input_cube = self._load_cube(self.input_file, self.index)
        self.gt_cube = self._load_cube(self.target_file, self.index)
        self.pred_cube = self._load_cube(self.prediction_file, self.index, pred=True)

    def _parse_prediction_path(self, path):
        parts = path.split("/")
        model_name = parts[-3]
        sample_match = re.search(r"(Sample\d+)", path)
        epoch_match = re.search(r"epoch(\d+)", path)
        sample = sample_match.group(1) if sample_match else "SampleUnknown"
        epoch = f"epoch{epoch_match.group(1)}" if epoch_match else "epochUnknown"
        return model_name, sample, epoch

    def _load_cube(self, file_path, index, pred=False):
        with h5py.File(file_path, "r") as f:
            key = list(f.keys())[0]
            data = f[key][index]
            if pred and data.ndim == 4:
                return np.array(data[0])
            return np.array(data)

    def evaluate_all(self):
        logging.info("🔍 Starting full evaluation...")

        if self.log_file and os.path.exists(self.log_file):
            logging.info("📉 Plotting loss curve...")
            plot_loss_curve(self.log_file, output_dir=self.output_dir)

        # (1+δ) 분포 비교
        compare_density_distribution(
            y_true=self.gt_cube,
            y_pred=self.pred_cube,
            index=self.index,
            model_name=self.model_name,
            save_dir=self.output_dir,
            show=self.show,
            bin_range=(0.5, 20),
            bins=np.logspace(np.log10(0.5), np.log10(20), 40),
            plot_mean_only=True
        )

        plot_projection_comparison(
            input_cube=self.input_cube,
            gt_cube=self.gt_cube,
            pred_cube=self.pred_cube,
            axis=0,
            index=self.index,
            model_name=self.model_name,
            save_dir=self.output_dir,
            show=self.show
        )

        plot_pixel_scatter(
            y_true=self.gt_cube,
            y_pred=self.pred_cube,
            index=self.index,
            model_name=self.model_name,
            save_dir=self.output_dir,
            log_transform=True,
            show=self.show
        )

        analyzer = DensityFieldAnalyzer(
            true_density=self.gt_cube,
            pred_density=self.pred_cube,
            box_size=self.box_size,
            output_dir=self.output_dir
        )
        analyzer.run_all()

        analyze_gaussianity(
            y_true=self.gt_cube,
            y_pred=self.pred_cube,
            index=self.index,
            model_name=self.model_name,
            save_dir=self.output_dir,
            log_transform=True,
            show=self.show
        )

        compare_voxel_pdf(
            y_true=self.gt_cube,
            y_pred=self.pred_cube,
            index=self.index,
            model_name=self.model_name,
            save_dir=self.output_dir,
            log_transform=True,
            show=self.show
        )

        logging.info(f"✅ All evaluations completed and saved to {self.output_dir}")
