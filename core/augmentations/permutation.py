"""
Permutation augmentation for time-series data.

Based on the paper: "TS-OOD: Evaluating Time-Series Out-of-Distribution Detection"
Reference: Jiang et al. 2021

Permutation consists of:
1. Segmentation: Divide the series into multiple subsequences
2. Permuting: Randomly rearrange the subsequences

Paper finding: Second-best augmentation with avg AUROC of 0.619
"""

import torch
import torch.nn as nn


class Permutation(nn.Module):
    """
    Permutation augmentation by segmenting and shuffling.

    Args:
        num_segments: Number of segments to divide the series into (default: 4)
    """

    def __init__(self, num_segments: int = 4):
        super().__init__()
        self.num_segments = num_segments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply permutation to time-series batch.

        Args:
            x: Time-series tensor (batch_size, channels, length)

        Returns:
            Permuted time-series (batch_size, channels, length)
        """
        batch_size, num_channels, seq_len = x.shape

        # Calculate segment length
        segment_len = seq_len // self.num_segments

        # If seq_len is not perfectly divisible, we'll handle the remainder
        if seq_len % self.num_segments != 0:
            # Trim to make it divisible
            effective_len = segment_len * self.num_segments
            x_trimmed = x[:, :, :effective_len]
            remainder = x[:, :, effective_len:]
        else:
            x_trimmed = x
            remainder = None

        # Reshape to (batch_size, channels, num_segments, segment_len)
        x_segmented = x_trimmed.reshape(batch_size, num_channels, self.num_segments, segment_len)

        # Permute segments for each sample in the batch independently
        permuted_batch = []
        for i in range(batch_size):
            # Generate random permutation for this sample
            perm = torch.randperm(self.num_segments, device=x.device)

            # Apply permutation
            permuted = x_segmented[i, :, perm, :]  # (channels, num_segments, segment_len)
            permuted_batch.append(permuted)

        # Stack back to batch
        permuted_batch = torch.stack(permuted_batch, dim=0)

        # Reshape back to (batch_size, channels, length)
        permuted = permuted_batch.reshape(batch_size, num_channels, -1)

        # Add back remainder if it exists
        if remainder is not None:
            permuted = torch.cat([permuted, remainder], dim=2)

        return permuted
