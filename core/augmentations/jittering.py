"""
Jittering augmentation for time-series data.

Based on the paper: "TS-OOD: Evaluating Time-Series Out-of-Distribution Detection"
Reference: Sarkar et al. 2021

Jittering adds random noise (Gaussian, Poisson, or Exponential) to the time series.
"""

import torch
import torch.nn as nn


class Jittering(nn.Module):
    """
    Add Gaussian noise to time-series data.

    Args:
        sigma: Standard deviation of the noise (default: 0.05)
        noise_type: Type of noise - 'gaussian', 'uniform' (default: 'gaussian')
    """

    def __init__(self, sigma: float = 0.05, noise_type: str = 'gaussian'):
        super().__init__()
        self.sigma = sigma
        self.noise_type = noise_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply jittering to time-series batch.

        Args:
            x: Time-series tensor (batch_size, channels, length)

        Returns:
            Jittered time-series (batch_size, channels, length)
        """
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(x) * self.sigma
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(x) - 0.5) * 2 * self.sigma
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        return x + noise
