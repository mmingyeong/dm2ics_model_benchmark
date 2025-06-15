"""
model.py

Description:
    Definition of the Fourier Neural Operator (FNO) model in PyTorch.
    This model maps input functions to output functions using Fourier layers,
    and is suitable for solving PDE-related inverse problems and scientific data modeling.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-13

Reference:
    Adapted from the original FNO implementation:
    https://github.com/abelsr/Fourier-Neural-Operator/blob/main/FNO/PyTorch/fno.py
"""

"""
This model is a simplified variant of the original Fourier Neural Operator (FNO). 
It preserves the fundamental structure and principles of FNO—including the use of spectral convolution layers 
and coordinate-aware lifting—while omitting certain components like periodic padding and modular abstractions 
to enhance code clarity and experimental flexibility.
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.spectral_convolution import SpectralConvolution
from shared.logger import get_logger

logger = get_logger(__name__)


class FNO(nn.Module):
    """
    Fourier Neural Operator (FNO) model for 3D data.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 1 for scalar fields).
    out_channels : int
        Number of output channels.
    modes1, modes2, modes3 : int
        Number of Fourier modes to keep in each spatial dimension.
    width : int
        Width (number of channels) of the Fourier layers.
    lifting_channels : int, optional
        Number of hidden channels in the input lifting MLP (if used).
    add_grid : bool, optional
        Whether to concatenate coordinate grids to the input.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3,
                 width, activation: nn.Module, lifting_channels=None, add_grid=True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.lifting_channels = lifting_channels
        self.mid_channels = width
        self.activation = activation
        self.add_grid = add_grid

        # Lifting layers
        if lifting_channels is not None:
            self.p1 = nn.Linear(in_channels + 3, lifting_channels)  # +3 for coordinates
            self.p2 = nn.Linear(lifting_channels, self.mid_channels)
        else:
            self.p1 = nn.Linear(in_channels + 3, self.mid_channels)

        # Fourier layers
        self.fourier_blocks = nn.ModuleList([
            SpectralConvolution(self.mid_channels, self.mid_channels,
                                modes=[self.modes1, self.modes2, self.modes3])
            for _ in range(4)
        ])

        # Projection layers
        self.q1 = nn.Linear(self.mid_channels, self.mid_channels)
        self.final = nn.Linear(self.mid_channels, out_channels)
        self.activation = nn.GELU()

        logger.info("✅ FNO model initialized successfully.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C_in, D1, D2, D3]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, C_out, D1, D2, D3]
        """
        batch, _, *sizes = x.shape
        logger.info(f"🚀 FNO forward pass started. Input shape: {x.shape}")

        if self.add_grid:
            grid = self.set_grid(x)  # [B, 3, D1, D2, D3]
            x = torch.cat((x, grid), dim=1)  # [B, C+3, D1, D2, D3]
            logger.info(f"🔗 Added grid to input. New shape: {x.shape}")

        # Reorder: [B, C, D1, D2, D3] → [B, D1, D2, D3, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(-1, x.shape[-1])  # Flatten for linear

        if self.lifting_channels is not None:
            x = self.p1(x)
            x = F.gelu(x)
            x = self.p2(x)
        else:
            x = self.p1(x)

        # Restore shape: [B, D1, D2, D3, mid_channels]
        x = x.view(batch, *sizes, self.mid_channels)

        # Reorder: [B, D1, D2, D3, C] → [B, C, D1, D2, D3]
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Fourier layers
        for i, layer in enumerate(self.fourier_blocks):
            x = layer(x)
            logger.info(f"🔁 Passed through Fourier layer {i + 1}/{len(self.fourier_blocks)}")

        # Projection
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.q1(x)
        #x = self.activation(x)
        x = self.final(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        logger.info(f"✅ Forward pass completed. Output shape: {x.shape}")
        return x

    def set_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate normalized coordinate grid [0, 1] for each spatial dimension.

        Returns
        -------
        torch.Tensor : [B, 3, D1, D2, D3]
        """
        batch, _, D1, D2, D3 = x.shape
        device = x.device

        logger.info(f"🌐 Generating coordinate grid with shape: [{D1}, {D2}, {D3}]")

        grids = torch.meshgrid(
            torch.linspace(0, 1, D1, device=device),
            torch.linspace(0, 1, D2, device=device),
            torch.linspace(0, 1, D3, device=device),
            indexing='ij'
        )
        grid = torch.stack(grids, dim=0).unsqueeze(0).expand(batch, -1, -1, -1, -1)  # [B, 3, D1, D2, D3]

        logger.info("✅ Coordinate grid generated.")
        return grid
