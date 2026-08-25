from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("mahalanobis")
class MahalanobisDetector(BaseOODDetector):
    """
    Mahalanobis Distance-based OOD Detection (MDS).

    Following the paper: "TS-OOD: Evaluating Time-Series Out-of-Distribution Detection"
    Reference: Lee et al. 2018 - "A simple unified framework for detecting
    out-of-distribution samples and adversarial attacks"

    Key approach:
    - Compute per-class Gaussian distributions with tied (pooled) covariance
    - OOD score = Mahalanobis distance to nearest class mean
    - Uses features from pre-logit layer

    Paper findings (Table 2):
    - Strong ID-OOD correlation: 0.690 (CE), 0.751 (MPC)
    - Second-best method after DFM variants
    """
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.class_means: Dict[Any, np.ndarray] = {}
        self.tied_cov_inv: Optional[np.ndarray] = None
        self.class_labels: Optional[np.ndarray] = None

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """
        Fit class-conditional Gaussians with tied covariance.

        Args:
            x_id: Training data (ID classes only)
            y_id: Class labels (REQUIRED)
        """
        if y_id is None:
            raise ValueError(
                "Mahalanobis detector requires class labels (y_id) "
                "to compute per-class means and tied covariance."
            )

        x_tensor = self._to_tensor(x_id)
        with torch.no_grad():
            feats = self._forward_features(x_tensor).detach().cpu().numpy()

        # Ensure y_id is numpy array
        if isinstance(y_id, torch.Tensor):
            y_id = y_id.cpu().numpy()
        elif not isinstance(y_id, np.ndarray):
            y_id = np.array(y_id)

        self.class_labels = np.unique(y_id)

        # Compute per-class means and accumulate WITHIN-CLASS deviations.
        all_feats_list = []
        for class_label in self.class_labels:
            mask = (y_id == class_label)
            feats_class = feats[mask]

            if len(feats_class) == 0:
                raise ValueError(f"Class {class_label} has no samples.")

            # Store class mean
            self.class_means[class_label] = feats_class.mean(axis=0)

            # Center each class's features on ITS OWN mean before pooling so the
            # tied covariance is the within-class scatter Sigma_W, not the total
            # covariance Sigma_T = Sigma_W + Sigma_B. This matches Lee et al. 2018
            # Eq. (1) and the official sample_estimator (lib_generation.py:112-114),
            # which pool class-centered features. Pooling raw features instead would
            # let EmpiricalCovariance center on the global mean, contaminating the
            # metric with between-class scatter (not rank-preserving).
            all_feats_list.append(feats_class - self.class_means[class_label])

        # Compute tied (pooled) within-class covariance matrix across all ID classes.
        # Deviations sum to zero per class, so the pooled matrix already has zero mean.
        all_feats_concat = np.concatenate(all_feats_list, axis=0)

        # Use sklearn's EmpiricalCovariance for robust estimation
        cov_estimator = EmpiricalCovariance().fit(all_feats_concat)
        cov = cov_estimator.covariance_

        # Add regularization to ensure invertibility
        cov += 1e-6 * np.eye(cov.shape[0])

        # Compute inverse covariance (precision matrix)
        try:
            self.tied_cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse if covariance is singular
            self.tied_cov_inv = np.linalg.pinv(cov)

    def score(self, x: Any) -> np.ndarray:
        """
        Compute OOD scores as minimum Mahalanobis distance to ID class means.

        For each test sample:
            1. Compute Mahalanobis distance to each ID class mean
            2. Return minimum distance (closest class)

        Mahalanobis distance: sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
        where μ is class mean and Σ is tied covariance

        Higher score = more OOD (farther from all class means)
        """
        if not self.class_means or self.tied_cov_inv is None:
            raise RuntimeError("Mahalanobis detector must be fit on ID data before scoring.")

        x_tensor = self._to_tensor(x)
        with torch.no_grad():
            feats = self._forward_features(x_tensor).detach().cpu().numpy()

        num_samples = feats.shape[0]
        ood_scores = np.zeros(num_samples)

        # For each test sample
        for i in range(num_samples):
            test_feat = feats[i]

            # Compute Mahalanobis distance to each ID class mean
            distances = []
            for class_label in self.class_labels:
                class_mean = self.class_means[class_label]

                # Compute delta: (x - μ)
                delta = test_feat - class_mean

                # Mahalanobis distance: sqrt(delta^T * Σ^-1 * delta)
                mahal_dist_squared = delta @ self.tied_cov_inv @ delta.T
                mahal_dist = np.sqrt(max(0, mahal_dist_squared))  # Ensure non-negative

                distances.append(mahal_dist)

            # OOD score = minimum Mahalanobis distance (nearest class).
            # Lee et al. 2018: score = min_c distance(x, mu_c).
            # ID sample: close to at least one class → small min → low score
            # OOD sample: far from ALL classes → large min → high score
            ood_scores[i] = min(distances)

        return ood_scores
