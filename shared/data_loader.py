"""
shared/data_loader.py

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-06-10
Description: Custom PyTorch Dataset class for loading multiple HDF5 files 
             containing 3D input/output subcubes for deep learning.

Reference:
- https://github.com/redeostm/ML_LocalEnv/blob/main/generatorSingle.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import h5py
import torch
from glob import glob
from torch.utils.data import Dataset, Subset, DataLoader

from shared.logger import get_logger

logger = get_logger("data_loader", log_dir="logs")

class HDF5Dataset(Dataset):
    """
    Custom dataset for loading 3D input/output subcubes from multiple HDF5 files.
    
    Each HDF5 file must contain a dataset named 'subcubes'.
    The same index is used for corresponding input and output files.
    """

    def __init__(self, input_files, output_files):
        self.input_files = input_files
        self.output_files = output_files

        assert len(self.input_files) == len(self.output_files), \
            "The number of input and output files must match."

        logger.info(f"🔍 Initializing dataset with {len(self.input_files)} file pairs.")

        # Precompute number of samples per file
        self.sample_counts = []
        for i_file in self.input_files:
            with h5py.File(i_file, 'r') as f:
                n = f['subcubes'].shape[0]
                self.sample_counts.append(n)

        self.cumulative_counts = [0]
        for count in self.sample_counts:
            self.cumulative_counts.append(self.cumulative_counts[-1] + count)

        self.total_samples = self.cumulative_counts[-1]
        logger.info(f"📦 Total samples across all files: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = next(i for i in range(len(self.cumulative_counts) - 1)
                        if self.cumulative_counts[i] <= idx < self.cumulative_counts[i + 1])
        local_idx = idx - self.cumulative_counts[file_idx]

        input_file = self.input_files[file_idx]
        output_file = self.output_files[file_idx]

        with h5py.File(input_file, 'r') as fin, h5py.File(output_file, 'r') as fout:
            x = torch.from_numpy(fin['subcubes'][local_idx]).float().unsqueeze(0)
            y = torch.from_numpy(fout['subcubes'][local_idx]).float().unsqueeze(0)

        return x, y


def get_dataloader(input_dir, output_dir, batch_size, split="train", shuffle=True,
                   sample_fraction=1.0, num_workers=0):
    """
    Returns a DataLoader for the specified split by selecting appropriate HDF5 files.

    Parameters
    ----------
    input_dir : str
        Directory containing input HDF5 files.
    output_dir : str
        Directory containing corresponding output HDF5 files.
    batch_size : int
        Batch size for DataLoader.
    split : str
        One of ["train", "val", "test"].
    shuffle : bool
        Whether to shuffle the data.
    sample_fraction : float, optional
        Fraction of the dataset to sample (applies only to 'train' split).
    num_workers : int, optional
        Number of subprocesses to use for data loading. Default is 0 (main process).

    Returns
    -------
    DataLoader
        PyTorch DataLoader instance for the specified split.
    """
    all_input_files = sorted(glob(os.path.join(input_dir, "*.h5")))
    all_output_files = sorted(glob(os.path.join(output_dir, "*.h5")))

    assert len(all_input_files) == len(all_output_files), \
        "Mismatch between number of input and output HDF5 files."

    if split == "train":
        selected_inputs = all_input_files[:8]
        selected_outputs = all_output_files[:8]
    elif split == "val":
        selected_inputs = all_input_files[8:10]
        selected_outputs = all_output_files[8:10]
    elif split == "test":
        selected_inputs = all_input_files[10:12]
        selected_outputs = all_output_files[10:12]
    else:
        raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test'].")

    logger.info(f"📂 Split: {split} | Files: {len(selected_inputs)}")
    dataset = HDF5Dataset(selected_inputs, selected_outputs)

    logger.info(f"📂 Split: {split} | Files: {len(selected_inputs)}")
    dataset = HDF5Dataset(selected_inputs, selected_outputs)

    # 샘플링 적용 (모든 split에 대해 적용 가능)
    if 0.0 < sample_fraction < 1.0:
        total_len = len(dataset)
        sample_size = int(sample_fraction * total_len)
        sampled_indices = np.random.choice(total_len, size=sample_size, replace=False)
        dataset = Subset(dataset, sampled_indices)
        logger.info(f"🔎 Sampled {sample_size} / {total_len} {split} samples ({sample_fraction*100:.1f}%)")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
