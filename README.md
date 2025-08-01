
# 🧠 dm2ics_model_benchmark

**Benchmarking Deep Learning Models for Mapping Dark Matter to Initial Conditions**

This repository provides a modular benchmarking framework for evaluating various deep learning models that reconstruct cosmological initial conditions from the evolved dark matter density field.

The benchmark currently includes four representative models:
- **U-Net (V-Net style 3D CNN)**
- **Fourier Neural Operator (FNO)**
- **Vision Transformer (ViT) for voxel-to-voxel regression**
- **Conditional Generative Adversarial Network (cGAN)**

The repository is designed to be scalable and extendable to additional models such as ViT and cGAN in the future.

---

## 📁 Directory Structure

```

dm2ics_model_benchmark/
├── evaluation/ # Evaluation utilities (loss curves, power spectrum, etc.)
├── models/ # Model implementations and training/inference logic
│ ├── unet/
│ ├── fno/
│ ├── vit/
│ └── cgan/
├── pretrain/ # Lightweight pretraining/testing scripts
├── results/ # Logs, checkpoints, predictions
│ ├── unet/
│ ├── fno/
│ ├── vit_test/
│ └── ...
├── scripts/ # Shell scripts to automate experiments
├── shared/ # Common utilities (data loading, logging, loss, metrics)
├── tests/ # Jupyter notebooks or unit tests for sanity checks
├── tuning/ # Hyperparameter tuning setup (e.g., Optuna)
├── .gitignore
└── README.md
```

---

## 📚 Dataset

This project assumes an HDF5 dataset containing 3D cube pairs of evolved density fields and corresponding initial conditions:

* `input`: Dark matter overdensity map (e.g., shape `[N, 1, 60, 60, 60]`)
* `target`: Initial density map (e.g., shape `[N, 1, 60, 60, 60]`)

Make sure `shared/data_loader.py` is configured to match your dataset structure.

---

## 🧠 Models

| Model | Architecture              | Characteristics                                        |
| ----- | ------------------------- | ------------------------------------------------------ |
| U-Net | 3D V-Net-style CNN        | Local convolution, spatial detail preservation         |
| FNO   | Fourier-based operator    | Global receptive field, frequency-domain processing    |
| ViT   | 3D Vision Transformer     | Global attention across voxels, long-range correlation |
| cGAN  | Generator + Discriminator | Adversarial training for sharper outputs               |

---

## 🔍 Tuning & Pretraining
Lightweight sanity-check notebooks are available in pretrain/.

Hyperparameter tuning configurations are maintained in tuning/, potentially using Optuna or similar frameworks.

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