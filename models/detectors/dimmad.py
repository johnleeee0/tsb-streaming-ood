"""Corrected DiMMAD detector (flattened, canonical).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

DiMMAD: Distance Multi-Metric Anomaly Detection (Adapted). Based on Chaini, Bianco &
Mahabal, "In Search of the Unknown Unknowns: A Multi-Metric Distance Ensemble for Out of
Distribution Anomaly Detection in Astronomical Surveys" (NeurIPS ML4PS 2025);
code https://github.com/sidchaini/distclassipy (DistClassiPy).

This file is SELF-CONTAINED: the base multi-metric machinery (formerly
models/ood_methods/dimmad.py) is inlined here as ``_DiMMADBase`` and the corrected
variant (formerly methods/dimmad/dimmad_enh) is the canonical, registered
``DiMMADDetector``. Behaviour is identical to the dimmad_enh variant used in the
benchmark method_set.

The corrected variant:
  * Drops scipy's boolean hamming/dice metrics (ill-defined on continuous features).
  * Restores Jaccard as its CONTINUOUS form — the Ružička / weighted-Jaccard distance
    (d = 1 − Σ min(uᵢ,vᵢ) / Σ max(uᵢ,vᵢ)) — matching DistClassiPy / the paper's Fig. 1.
  * Uses the MEDIAN class-centroid central statistic (DistClassiPy default).
Aggregation (class_agg='min', metric_agg='median') and orientation are unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import (
    euclidean, cityblock, chebyshev, minkowski,
    cosine, correlation, canberra, braycurtis,
    mahalanobis, hamming, jaccard, dice
)
# Note: kulsinski, rogerstanimoto, russellrao deprecated in scipy 1.9+

from core.registry import register_ood
from core.base_ood import BaseOODDetector


# ---------------------------------------------------------------------------
# Distance Metric Functions
# ---------------------------------------------------------------------------

def standardized_euclidean(u: np.ndarray, v: np.ndarray, V: Optional[np.ndarray] = None) -> float:
    """Standardized Euclidean distance: sqrt(sum((u - v)^2 / var))."""
    if V is None:
        V = np.ones_like(u)
    diff = u - v
    return np.sqrt(np.sum(diff ** 2 / (V + 1e-10)))


def mahalanobis_wrapper(u: np.ndarray, v: np.ndarray, VI: Optional[np.ndarray] = None) -> float:
    """Mahalanobis distance with covariance inverse."""
    if VI is None:
        VI = np.eye(len(u))
    return mahalanobis(u, v, VI)


# Distance metric registry: (name, function, needs_covariance)
DISTANCE_METRICS: List[Tuple[str, Callable, bool]] = [
    ("euclidean", euclidean, False),
    ("manhattan", cityblock, False),
    ("chebyshev", chebyshev, False),
    ("minkowski", lambda u, v: minkowski(u, v, p=3), False),  # p=3 for variety
    ("cosine", cosine, False),
    ("correlation", correlation, False),
    ("canberra", canberra, False),
    ("braycurtis", braycurtis, False),
    ("mahalanobis", mahalanobis_wrapper, True),  # Needs covariance
    ("standardized_euclidean", standardized_euclidean, True),  # Needs variance
    ("hamming", hamming, False),
    ("jaccard", jaccard, False),
    ("dice", dice, False),
]


# ---------------------------------------------------------------------------
# Base DiMMAD machinery (inlined from models/ood_methods/dimmad.py)
# ---------------------------------------------------------------------------

class _DiMMADBase(BaseOODDetector):
    """DiMMAD-Lite base: multi-metric distance ensemble for OOD detection."""

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.class_agg: str = self.config.get("class_agg", "min")
        self.metric_agg: str = self.config.get("metric_agg", "median")
        self.metric_names: Optional[List[str]] = self.config.get("metrics", None)

        # Filter metrics if specified
        if self.metric_names is None:
            self.metrics = DISTANCE_METRICS
        else:
            self.metrics = [m for m in DISTANCE_METRICS if m[0] in self.metric_names]

        self.class_centroids: Optional[np.ndarray] = None  # (n_classes, feat_dim)
        self.class_labels: Optional[np.ndarray] = None  # (n_classes,)
        self.inv_cov: Optional[np.ndarray] = None  # For Mahalanobis
        self.variance: Optional[np.ndarray] = None  # For standardized Euclidean

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """Compute class centroids from ID data (mean central statistic)."""
        x_tensor = self._to_tensor(x_id)  # (N, C, T)

        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        N, feat_dim = feats.shape
        feats_np = feats.cpu().numpy()

        if y_id is not None:
            y_np = np.array(y_id)
            unique_classes = np.unique(y_np)
            centroids = []
            for c in unique_classes:
                class_feats = feats_np[y_np == c]
                centroid = np.mean(class_feats, axis=0)
                centroids.append(centroid)
            self.class_centroids = np.array(centroids)  # (n_classes, feat_dim)
            self.class_labels = unique_classes
        else:
            self.class_centroids = np.mean(feats_np, axis=0, keepdims=True)  # (1, feat_dim)
            self.class_labels = np.array([0])

        try:
            cov = np.cov(feats_np.T)
            self.inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(feat_dim))
        except np.linalg.LinAlgError:
            self.inv_cov = np.linalg.pinv(np.cov(feats_np.T) + 1e-6 * np.eye(feat_dim))

        self.variance = np.var(feats_np, axis=0) + 1e-10

    def score(self, x: Any) -> np.ndarray:
        """Multi-metric distance ensemble. Higher = more OOD."""
        if self.class_centroids is None:
            raise RuntimeError("DiMMAD must be fit before scoring.")

        x_tensor = self._to_tensor(x)

        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        N = len(feats)
        feats_np = feats.cpu().numpy()  # (N, feat_dim)

        metric_scores = []  # List of (N,) arrays, one per metric

        for metric_name, metric_func, needs_cov in self.metrics:
            distances = []  # (N, n_classes)
            for i in range(N):
                feat_i = feats_np[i]
                dists_to_centroids = []
                for centroid in self.class_centroids:
                    if needs_cov:
                        if metric_name == "mahalanobis":
                            dist = metric_func(feat_i, centroid, self.inv_cov)
                        elif metric_name == "standardized_euclidean":
                            dist = metric_func(feat_i, centroid, self.variance)
                        else:
                            dist = metric_func(feat_i, centroid)
                    else:
                        try:
                            dist = metric_func(feat_i, centroid)
                        except Exception:
                            dist = euclidean(feat_i, centroid)
                    dists_to_centroids.append(dist)
                distances.append(dists_to_centroids)

            distances = np.array(distances)  # (N, n_classes)

            if self.class_agg == "min":
                single_metric_score = np.min(distances, axis=1)  # (N,)
            elif self.class_agg == "median":
                single_metric_score = np.median(distances, axis=1)  # (N,)
            else:
                single_metric_score = np.mean(distances, axis=1)  # fallback

            metric_scores.append(single_metric_score)

        metric_scores = np.array(metric_scores)  # (n_metrics, N)

        if self.metric_agg == "median":
            ood_scores = np.median(metric_scores, axis=0)  # (N,)
        elif self.metric_agg == "mean":
            ood_scores = np.mean(metric_scores, axis=0)  # (N,)
        else:
            ood_scores = np.median(metric_scores, axis=0)  # default

        return ood_scores.astype(np.float64)


# ---------------------------------------------------------------------------
# Corrected (canonical) variant
# ---------------------------------------------------------------------------

# Continuous metrics retained from the base ensemble (scipy's boolean hamming/jaccard/dice excluded).
_CONTINUOUS = {
    "euclidean", "manhattan", "chebyshev", "minkowski", "cosine", "correlation",
    "canberra", "braycurtis", "mahalanobis", "standardized_euclidean",
}


def ruzicka(u: np.ndarray, v: np.ndarray) -> float:
    """Continuous Jaccard distance (Ružička / weighted Jaccard).

    d(u, v) = 1 − Σ min(uᵢ, vᵢ) / Σ max(uᵢ, vᵢ).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    min_sum = np.minimum(u, v).sum()
    max_sum = np.maximum(u, v).sum()
    if not np.isfinite(max_sum) or abs(max_sum) < 1e-12:
        return 0.0
    return float(1.0 - min_sum / max_sum)


# Faithful ensemble: continuous base metrics + continuous Jaccard (Ružička).
_ENH_METRICS: List[Tuple[str, Callable, bool]] = (
    [m for m in DISTANCE_METRICS if m[0] in _CONTINUOUS]
    + [("jaccard", ruzicka, False)]
)


@register_ood("dimmad")
class DiMMADDetector(_DiMMADBase):
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.metrics = _ENH_METRICS

    # fit — identical to the base except class centroids use the *median* central
    # statistic (DistClassiPy default) instead of the mean.
    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        x_tensor = self._to_tensor(x_id)  # (N, C, T)

        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        _, feat_dim = feats.shape
        feats_np = feats.cpu().numpy()

        if y_id is not None:
            y_np = np.array(y_id)
            unique_classes = np.unique(y_np)
            centroids = [np.median(feats_np[y_np == c], axis=0) for c in unique_classes]
            self.class_centroids = np.array(centroids)  # (n_classes, feat_dim)
            self.class_labels = unique_classes
        else:
            self.class_centroids = np.median(feats_np, axis=0, keepdims=True)  # (1, feat_dim)
            self.class_labels = np.array([0])

        try:
            cov = np.cov(feats_np.T)
            self.inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(feat_dim))
        except np.linalg.LinAlgError:
            self.inv_cov = np.linalg.pinv(np.cov(feats_np.T) + 1e-6 * np.eye(feat_dim))

        self.variance = np.var(feats_np, axis=0) + 1e-10
