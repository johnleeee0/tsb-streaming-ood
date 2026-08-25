"""
Cross-Entropy Loss wrapper for consistency with MPC loss interface.
"""

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy loss for classification.

    This is a simple wrapper around torch.nn.CrossEntropyLoss
    to maintain a consistent interface with MPC loss.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)

        Returns:
            Cross-entropy loss
        """
        return self.criterion(logits, labels)
