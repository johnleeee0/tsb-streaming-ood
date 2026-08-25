"""
Multi-Positive Contrastive (MPC) Loss

Based on the paper: "TS-OOD: Evaluating Time-Series Out-of-Distribution Detection"
Reference: Tian et al. 2024 - "StableRep: Synthetic images from text-to-image models
make strong visual representation learners"

MPC is a contrastive learning loss where:
- Anchor sample a is compared against a set of candidates {b1, b2, ..., bK}
- Multiple candidates can be "positive" (match the anchor, same class)
- Loss encourages anchor to be similar to all matching candidates

Formula (from paper):
1. Compute similarity-based probabilities:
   p_i = exp(a · b_i / τ) / Σ_j exp(a · b_j / τ)

2. Ground-truth distribution:
   y_i = 1_match(a,b_i) / Σ_j 1_match(a,b_j)

3. MPC loss = CrossEntropy(y, p) = -Σ y_i * log(p_i)

where τ is temperature, and all vectors are L2-normalized.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPositiveContrastiveLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss for time-series representation learning.

    Args:
        temperature: Temperature parameter τ for softmax (default: 0.1)
        normalize: Whether to L2-normalize embeddings (default: True, required by paper)
    """

    def __init__(self, temperature: float = 0.1, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(
        self,
        anchor: torch.Tensor,
        candidates: torch.Tensor,
        anchor_labels: torch.Tensor,
        candidate_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MPC loss.

        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            candidates: Candidate embeddings (batch_size * num_candidates, embedding_dim)
                        For each anchor, there are multiple candidates (augmented views + other samples)
            anchor_labels: Class labels for anchors (batch_size,)
            candidate_labels: Class labels for candidates (batch_size * num_candidates,)

        Returns:
            MPC loss (scalar)
        """
        batch_size = anchor.size(0)

        # L2-normalize embeddings (required by paper)
        if self.normalize:
            anchor = F.normalize(anchor, p=2, dim=1)
            candidates = F.normalize(candidates, p=2, dim=1)

        # Compute similarity matrix: anchor · candidates^T
        # Shape: (batch_size, num_candidates)
        similarity = torch.matmul(anchor, candidates.T) / self.temperature

        # Compute probabilities via softmax
        # p_i = exp(a · b_i / τ) / Σ_j exp(a · b_j / τ)
        probs = F.softmax(similarity, dim=1)

        # Create ground-truth distribution y
        # y_i = 1 if match(anchor, candidate_i), else 0
        # Then normalize: y_i = 1_match(a,b_i) / Σ_j 1_match(a,b_j)
        anchor_labels_expanded = anchor_labels.unsqueeze(1)  # (batch_size, 1)
        candidate_labels_expanded = candidate_labels.unsqueeze(0)  # (1, num_candidates)

        # Match matrix: 1 if labels match, 0 otherwise
        matches = (anchor_labels_expanded == candidate_labels_expanded).float()

        # Normalize to get ground-truth distribution
        # Avoid division by zero
        num_matches = matches.sum(dim=1, keepdim=True).clamp(min=1e-8)
        ground_truth = matches / num_matches

        # MPC loss = CrossEntropy(ground_truth, probs)
        # = -Σ y_i * log(p_i)
        loss = -(ground_truth * torch.log(probs + 1e-8)).sum(dim=1).mean()

        return loss


class MPCWithAugmentation(nn.Module):
    """
    MPC Loss with data augmentation for time-series.

    This wrapper applies augmentation to create positive pairs for contrastive learning.

    Args:
        temperature: Temperature for MPC loss
        augmentation: Augmentation function to apply (e.g., magnitude_warping)
        num_augmented: Number of augmented views per sample (default: 1)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        augmentation: Optional[Callable] = None,
        num_augmented: int = 1,
    ):
        super().__init__()
        self.mpc_loss = MultiPositiveContrastiveLoss(temperature=temperature)
        self.augmentation = augmentation
        self.num_augmented = num_augmented

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with augmentation.

        Args:
            model: Backbone model to extract embeddings
            x: Input time-series data (batch_size, channels, length)
            labels: Class labels (batch_size,)

        Returns:
            MPC loss
        """
        batch_size = x.size(0)

        # Original embeddings (anchors)
        anchor_embeddings = model(x)

        # Create augmented candidates
        if self.augmentation is not None:
            # Apply augmentation num_augmented times
            augmented_views = []
            for _ in range(self.num_augmented):
                x_aug = self.augmentation(x)
                augmented_views.append(model(x_aug))

            # Also include original samples as candidates (in-batch negatives)
            augmented_views.append(anchor_embeddings)

            # Concatenate all candidates
            candidates = torch.cat(augmented_views, dim=0)

            # Repeat labels for augmented views
            candidate_labels = torch.cat([labels] * (self.num_augmented + 1), dim=0)
        else:
            # No augmentation: use all batch samples as candidates
            candidates = anchor_embeddings
            candidate_labels = labels

        # Compute MPC loss
        loss = self.mpc_loss(
            anchor=anchor_embeddings,
            candidates=candidates,
            anchor_labels=labels,
            candidate_labels=candidate_labels,
        )

        return loss
