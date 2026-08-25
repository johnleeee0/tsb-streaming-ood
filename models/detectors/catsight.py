"""
CatSight: Common Spatial Pattern for Change Detection (Adapted).

Based on:
    Paper: https://doi.org/10.1007/s13042-023-01810-z
           "CatSight, a direct path to proper multi-variate time series change
            detection: perceiving a concept drift through common spatial pattern"
           (Int. J. Machine Learning and Cybernetics, 2023)

Adaptation for OOD detection on frozen backbone:
    - Original CatSight uses CSP on raw multivariate time series
    - CatSight-Lite applies CSP on frozen backbone features
    - CSP maximizes variance difference between ID classes
    - OOD samples don't match learned spatial patterns → high score

Learns Common Spatial Pattern (CSP) filters by solving a generalized eigenvalue
problem (C1·W = λ·C2·W) on per-class covariance matrices from ID data. Spatial
filters with extreme eigenvalues are selected to maximize variance differences
between classes. Test samples are projected through these filters and scored
based on their deviation from reference variance patterns.

Parameters:
    n_components: Number of CSP components (default: 6)
    reg: Regularization for covariance (default: 1e-4)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy import linalg

from core.registry import register_ood
from core.base_ood import BaseOODDetector


# CSP Spatial Filter Computation

def compute_csp_filters(
    X1: np.ndarray,
    X2: np.ndarray,
    n_components: int = 6,
    reg: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Common Spatial Pattern (CSP) filters.

    CSP finds spatial filters that maximize variance difference between
    two classes by solving a generalized eigenvalue problem.

    Args:
        X1: (N1, feat_dim) features from class 1
        X2: (N2, feat_dim) features from class 2
        n_components: number of components to extract (half from top, half from bottom)
        reg: regularization for covariance matrices

    Returns:
        W: (feat_dim, n_components) CSP spatial filters
        eigenvalues: (n_components,) corresponding eigenvalues
    """
    # Compute covariance matrices
    C1 = np.cov(X1.T) + reg * np.eye(X1.shape[1])  # (feat_dim, feat_dim)
    C2 = np.cov(X2.T) + reg * np.eye(X2.shape[1])  # (feat_dim, feat_dim)

    # Generalized eigenvalue decomposition: C1·W = λ·C2·W
    # This finds W such that var(W.T @ X1) / var(W.T @ X2) is maximized/minimized
    eigenvalues, eigenvectors = linalg.eigh(C1, C1 + C2)

    # Sort by eigenvalues
    ix = np.argsort(eigenvalues)[::-1]  # Descending order
    eigenvalues = eigenvalues[ix]
    eigenvectors = eigenvectors[:, ix]

    # Select top and bottom components (extreme eigenvalues = max variance ratio)
    n_top = n_components // 2
    n_bottom = n_components - n_top

    top_ix = ix[:n_top]
    bottom_ix = ix[-n_bottom:]
    selected_ix = np.concatenate([top_ix, bottom_ix])

    W = eigenvectors[:, selected_ix]  # (feat_dim, n_components)
    eigs = eigenvalues[selected_ix]  # (n_components,)

    return W, eigs


# ---------------------------------------------------------------------------
# CatSight-Lite Detector
# ---------------------------------------------------------------------------

@register_ood("catsight")
class CatSightDetector(BaseOODDetector):
    """CatSight-Lite: CSP-based OOD detection.

    Config keys (all optional):
        n_components (int, 6)       – number of CSP components
        reg          (float, 1e-4)  – covariance regularization
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.n_components: int = int(self.config.get("n_components", 6))
        self.reg: float = float(self.config.get("reg", 1e-4))

        self.csp_filters: Optional[np.ndarray] = None  # (feat_dim, n_components)
        self.id_mean_proj: Optional[np.ndarray] = None  # (n_components,) ID centroid in CSP space
        self.id_std_proj: Optional[np.ndarray] = None  # (n_components,) ID std in CSP space

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """Learn CSP spatial filters from ID data.

        Args:
            x_id: ID training data, shape (N, C, T).
            y_id: Class labels (N,), optional for multi-class CSP.
        """
        x_tensor = self._to_tensor(x_id)  # (N, C, T)

        # Extract features from frozen backbone
        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        N, feat_dim = feats.shape
        feats_np = feats.cpu().numpy()

        # If labels available, use CSP between classes
        if y_id is not None and len(np.unique(y_id)) >= 2:
            y_np = np.array(y_id)
            unique_classes = np.unique(y_np)

            # Use first two classes for CSP (binary CSP)
            class1 = unique_classes[0]
            class2 = unique_classes[1]

            X1 = feats_np[y_np == class1]  # (N1, feat_dim)
            X2 = feats_np[y_np == class2]  # (N2, feat_dim)

        else:
            # Unsupervised: split temporally (first half vs second half)
            mid = N // 2
            X1 = feats_np[:mid]
            X2 = feats_np[mid:]

        # Compute CSP filters
        self.csp_filters, _ = compute_csp_filters(
            X1, X2,
            n_components=min(self.n_components, feat_dim),
            reg=self.reg,
        )

        # Compute reference statistics for ID data in CSP space
        feats_projected = feats_np @ self.csp_filters  # (N, n_components)
        self.id_mean_proj = np.mean(feats_projected, axis=0)  # (n_components,)
        self.id_std_proj = np.std(feats_projected, axis=0) + 1e-6  # (n_components,)

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, x: Any) -> np.ndarray:
        """Return OOD scores for x.  Higher = more OOD.

        Method: Project features via CSP filters, compute normalized distance to ID centroid.
        - ID samples: close to centroid in CSP space → low score
        - OOD samples: far from centroid in CSP space → high score

        Args:
            x: Input data (N, C, T).

        Returns:
            ood_scores: (N,) float array; higher value → more OOD.
        """
        if self.csp_filters is None or self.id_mean_proj is None:
            raise RuntimeError("CatSight must be fit before scoring.")

        x_tensor = self._to_tensor(x)

        # Extract features
        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        feats_np = feats.cpu().numpy()

        # Project via CSP filters
        feats_projected = feats_np @ self.csp_filters  # (N, n_components)

        # Compute normalized distance to the ID centroid in CSP space.
        # Consistent with the CSP distance direction: ID samples project close to
        # the learned ID centroid (small distance), while OOD samples deviate from
        # the learned spatial patterns (large distance).  Higher distance = more OOD.
        normalized_diff = (feats_projected - self.id_mean_proj) / self.id_std_proj
        ood_scores = np.linalg.norm(normalized_diff, axis=1)  # (N,)

        return ood_scores.astype(np.float64)
