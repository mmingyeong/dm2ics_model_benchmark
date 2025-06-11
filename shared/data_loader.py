"""
shared/data_loader.py

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-06-10
Description: Custom PyTorch Dataset class for loading multiple HDF5 files 
             containing 3D input/output subcubes for deep learning.

Reference:
- https://github.com/redeostm/ML_LocalEnv/blob/main/generatorSingle.py
"""

import os
import h5py
import torch
from torch.utils.data import Dataset
from shared.logger import get_logger

logger = get_logger("data_loader", log_dir="logs")

class HDF5Dataset(Dataset):
    """
    Custom dataset for loading 3D input/output subcubes from multiple HDF5 files.
    
    Each HDF5 file must contain a dataset named 'subcubes'.
    The same index is used for corresponding input and output files.
    
    Parameters
    ----------
    input_files : list of str
        List of paths to input HDF5 files.
    output_files : list of str
        List of paths to output HDF5 files.
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
        # Find the file index and local index
        file_idx = next(i for i in range(len(self.cumulative_counts) - 1)
                        if self.cumulative_counts[i] <= idx < self.cumulative_counts[i + 1])
        local_idx = idx - self.cumulative_counts[file_idx]

        input_file = self.input_files[file_idx]
        output_file = self.output_files[file_idx]

        with h5py.File(input_file, 'r') as fin, h5py.File(output_file, 'r') as fout:
            x = torch.from_numpy(fin['subcubes'][local_idx]).float().unsqueeze(0)
            y = torch.from_numpy(fout['subcubes'][local_idx]).float().unsqueeze(0)

        return x, y
