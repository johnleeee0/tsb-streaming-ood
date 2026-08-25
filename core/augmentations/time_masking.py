"""
Time Masking augmentation for time-series data.

Based on the paper: "TS-OOD: Evaluating Time-Series Out-of-Distribution Detection"
Reference: Han et al. 2021

Time Masking drops out certain observations within the time series to augment the data.
"""

import torch
import torch.nn as nn


class TimeMasking(nn.Module):
    """
    Time Masking augmentation by zeroing out random time steps.

    Args:
        mask_ratio: Ratio of time steps to mask (default: 0.1)
        mask_value: Value to use for masked positions (default: 0.0)
    """

    def __init__(self, mask_ratio: float = 0.1, mask_value: float = 0.0):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking to time-series batch.

        Args:
            x: Time-series tensor (batch_size, channels, length)

        Returns:
            Masked time-series (batch_size, channels, length)
        """
        batch_size, num_channels, seq_len = x.shape
        device = x.device

        # Number of time steps to mask
        num_masked = int(seq_len * self.mask_ratio)

        if num_masked == 0:
            return x

        # Create masked version
        x_masked = x.clone()

        # Apply masking independently for each sample
        for i in range(batch_size):
            # Randomly select time steps to mask
            mask_indices = torch.randperm(seq_len, device=device)[:num_masked]

            # Mask all channels at these time steps
            x_masked[i, :, mask_indices] = self.mask_value

        return x_masked
