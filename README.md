
# 🧠 dm2ics_model_benchmark

**Benchmarking Deep Learning Models for Mapping Dark Matter to Initial Conditions**

This repository provides a modular benchmarking framework for evaluating various deep learning models that reconstruct cosmological initial conditions from the evolved dark matter density field.

The initial version focuses on two representative architectures:
- **U-Net (V-Net style, PyTorch-based)**
- **Fourier Neural Operator (FNO)**

The repository is designed to be scalable and extendable to additional models such as ViT and cGAN in the future.

---

## 📁 Directory Structure

```

dm2ics_model_benchmark/
├── scripts/              # Shell scripts for training/evaluation automation
│   ├── train_all.sh
│   └── evaluate_all.sh
│
├── evaluation/           # Post-training evaluation tools
│   ├── compute_metrics.py
│   ├── plot_power_spectrum.py
│   └── compare_outputs.py
│
├── models/               # Model-specific training and inference logic
│   ├── unet/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   └── fno/
│       ├── model.py
│       ├── train.py
│       └── predict.py
│
├── results/              # Saved checkpoints, predictions, and logs
│   ├── unet/
│   └── fno/
│
├── shared/               # Common utilities used across all models
│   ├── data_loader.py    # HDF5 loader and preprocessing
│   ├── metrics.py        # MSE, PSNR, power spectrum, etc.
│   ├── losses.py         # Loss functions (e.g., MSE, spectral loss)
│   └── logger.py         # Logging utility
│
├── environment.yml       # Conda environment file
├── requirements.txt      # Python dependencies (for pip)
└── README.md             # Project overview and instructions

```

---

## 📦 Installation

Create a virtual environment using `conda`:

```bash
conda env create -f environment.yml
conda activate dm2ics
```

Or use `pip`:

```bash
pip install -r requirements.txt
```

---

## 🧪 Quick Start

### Training

```bash
bash scripts/train_all.sh
```

Or train a single model manually:

```bash
python models/unet/train.py      # For U-Net
python models/fno/train.py       # For FNO
```

### Inference

```bash
python models/unet/predict.py
```

### Evaluation

```bash
bash scripts/evaluate_all.sh
# or run individual evaluation modules
python evaluation/compute_metrics.py
python evaluation/plot_power_spectrum.py
```

---

## 📚 Dataset

This project assumes an HDF5 dataset containing 3D cube pairs of evolved density fields and corresponding initial conditions:

* `input`: Dark matter overdensity map (e.g., shape `[N, 1, 60, 60, 60]`)
* `target`: Initial density map (e.g., shape `[N, 1, 60, 60, 60]`)

Make sure `shared/data_loader.py` is configured to match your dataset structure.

---

## 🧠 Models

| Model | Architecture               | Purpose                                            |
| ----- | -------------------------- | -------------------------------------------------- |
| U-Net | 3D V-Net-style CNN         | Local receptive field, strong spatial localization |
| FNO   | Spectral operator learning | Long-range interactions via Fourier domain         |

---

## 📈 Evaluation Metrics

* Mean Squared Error (MSE)
* Power Spectrum Recovery
* Peak Signal-to-Noise Ratio (PSNR)

See `evaluation/` for implementations.

---

## ✏️ License

This repository is currently under active development. Please contact the author before redistribution or use for publication.

---

## 👩‍💻 Author

Mingyeong Yang (양민경)
PhD Student, Astronomy & Cosmology
UST – Korea Astronomy and Space Science Institute (KASI)

```

---