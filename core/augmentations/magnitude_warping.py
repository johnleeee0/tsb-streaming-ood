"""
Magnitude Warping augmentation for time-series data.

Based on the paper: "TS-OOD: Evaluating Time-Series Out-of-Distribution Detection"
Reference: Um et al. 2017 - "Data augmentation of wearable sensor data for
parkinson's disease monitoring using convolutional neural networks"

Magnitude Warping changes the magnitude of each sample in a time-series dataset
by multiplication with a smooth curve generated using cubic spline interpolation.

Paper finding: Best-performing augmentation with avg AUROC of 0.634
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import CubicSpline


class MagnitudeWarping(nn.Module):
    """
    Magnitude Warping augmentation using cubic spline curves.

    Args:
        sigma: Standard deviation of the warping magnitude (default: 0.2)
        num_knots: Number of control points for cubic spline (default: 4)
    """

    def __init__(self, sigma: float = 0.2, num_knots: int = 4):
        super().__init__()
        self.sigma = sigma
        self.num_knots = num_knots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping to time-series batch.

        Args:
            x: Time-series tensor (batch_size, channels, length)

        Returns:
            Warped time-series (batch_size, channels, length)
        """
        device = x.device
        dtype = x.dtype
        batch_size, num_channels, seq_len = x.shape

        # Convert to numpy for scipy interpolation
        x_np = x.detach().cpu().numpy()

        # Apply warping to each sample in the batch
        warped_batch = []
        for i in range(batch_size):
            # Generate random knots for cubic spline
            knot_positions = np.linspace(0, seq_len - 1, self.num_knots)
            knot_values = np.random.normal(loc=1.0, scale=self.sigma, size=self.num_knots)

            # Create cubic spline curve
            cs = CubicSpline(knot_positions, knot_values)

            # Evaluate spline at all time steps
            time_steps = np.arange(seq_len)
            warping_curve = cs(time_steps)

            # Apply warping curve to all channels
            # Shape: (channels, length)
            warped_sample = x_np[i] * warping_curve[np.newaxis, :]

            warped_batch.append(warped_sample)

        # Convert back to tensor
        warped_np = np.stack(warped_batch, axis=0)
        warped = torch.from_numpy(warped_np).to(dtype).to(device)

        return warped


class MagnitudeWarpingFast(nn.Module):
    """
    Faster version of Magnitude Warping using PyTorch interpolation.

    This version uses linear interpolation instead of cubic spline for speed,
    which may be sufficient for many applications.

    Args:
        sigma: Standard deviation of the warping magnitude (default: 0.2)
        num_knots: Number of control points for interpolation (default: 4)
    """

    def __init__(self, sigma: float = 0.2, num_knots: int = 4):
        super().__init__()
        self.sigma = sigma
        self.num_knots = num_knots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping to time-series batch (fast version).

        Args:
            x: Time-series tensor (batch_size, channels, length)

        Returns:
            Warped time-series (batch_size, channels, length)
        """
        batch_size, num_channels, seq_len = x.shape
        device = x.device

        # Generate random knots
        knot_positions = torch.linspace(0, seq_len - 1, self.num_knots, device=device)
        knot_values = torch.normal(
            mean=1.0,
            std=self.sigma,
            size=(batch_size, self.num_knots),
            device=device,
        )

        # Linear interpolation to create warping curve
        time_steps = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Simple linear interpolation for each batch
        warping_curves = []
        for i in range(batch_size):
            # Use PyTorch's interpolate (requires specific shape)
            curve = torch.nn.functional.interpolate(
                knot_values[i].unsqueeze(0).unsqueeze(0),  # (1, 1, num_knots)
                size=seq_len,
                mode='linear',
                align_corners=True,
            ).squeeze()  # (seq_len,)

            warping_curves.append(curve)

        warping_curves = torch.stack(warping_curves, dim=0)  # (batch_size, seq_len)

        # Apply warping: multiply each channel by the curve
        # x: (batch_size, channels, length)
        # warping_curves: (batch_size, length)
        # Need to broadcast: (batch_size, 1, length)
        warping_curves = warping_curves.unsqueeze(1)

        warped = x * warping_curves

        return warped
