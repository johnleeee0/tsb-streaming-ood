"""DriftLens (Class-D appendix build) — BATCH-LEVEL Fréchet drift detector.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Faithful re-build of the OFFICIAL DriftLens granularity (window/batch-level),
per methods/_validation/CLASS_D_DECISIONS.md and BUILD_PLAN_CLASS_D.md §4.

Official DriftLens (Greco et al., arXiv:2406.17813; repo grecosalvatore/drift-lens)
computes, per monitoring window, a distribution-to-distribution **Fréchet
(Wasserstein-2)** distance between the window's embedding distribution and an ID
baseline in PCA space. There is NO per-sample score in the paper — a per-sample
proxy simply duplicates the `mahalanobis` detector (median Spearman 0.999, see
methods/driftlens/VERIFICATION.md §3A), so this build exposes ONLY a batch-level
scorer, exactly as the reference `driftlens.py::_compute_frechet_distribution_distances`
and `distribution_distances/frechet_drift_distance.py::frechet_distance` do.

  baseline (offline) : PCA on ID embeddings -> (mu_b, Sigma_b) in PCA space
  score_batch(batch) : embed windows -> PCA transform -> (mu_w, Sigma_w)
                       -> Fréchet(mu_b, mu_w, Sigma_b, Sigma_w)   (higher = more OOD)

The scorer runs on the SHARED FROZEN backbone (no training), so it does not break
the backbone anchor — but its native granularity is the batch, so it is evaluated
with batch-level AUROC (batches of B consecutive windows, batch label = frac OOD >= tau).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA


def frechet_distance(mu_x: np.ndarray, mu_y: np.ndarray,
                     sigma_x: np.ndarray, sigma_y: np.ndarray) -> float:
    """Fréchet (Wasserstein-2) distance between two multivariate Gaussians.

    Matches the official drift-lens formulation
    (reference/driftlens/distribution_distances/frechet_drift_distance.py:53):

        ||mu_x - mu_y|| + trace(Sigma_x + Sigma_y - 2 * sqrtm(Sigma_x @ Sigma_y))

    A tiny diagonal is added before the matrix square root for numerical stability
    on near-singular batch covariances (a documented risk in BUILD_PLAN §4.5); the
    imaginary residue of sqrtm on a PSD product is discarded.
    """
    mu_x = np.asarray(mu_x, dtype=np.float64).ravel()
    mu_y = np.asarray(mu_y, dtype=np.float64).ravel()
    sigma_x = np.atleast_2d(np.asarray(sigma_x, dtype=np.float64))
    sigma_y = np.atleast_2d(np.asarray(sigma_y, dtype=np.float64))

    diff = float(np.linalg.norm(mu_x - mu_y))

    d = sigma_x.shape[0]
    eps = 1e-6 * np.eye(d)
    covmean = scipy.linalg.sqrtm((sigma_x + eps) @ (sigma_y + eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr = float(np.trace(sigma_x + sigma_y - 2.0 * covmean))
    val = diff + tr
    if not np.isfinite(val):
        return float("nan")
    # Fréchet distance is non-negative; clamp tiny negative round-off.
    return max(val, 0.0)


class DriftLensClassD:
    """Batch-level Fréchet drift detector on shared frozen-backbone embeddings.

    Interface:
      fit(id_windows)        -> sets the ID baseline (mu_b, Sigma_b) in PCA space
      score_batch(batch)     -> float, higher = more OOD  (usable by batch_level_auroc)

    Params (config):
      n_pc : number of PCA components for the baseline / batch distributions.
             Automatically capped to keep the per-batch covariance non-singular
             (n_pc <= batch_size - 2), so pass the intended monitoring batch size
             as `batch_size` in the config to have it enforced at fit time.
    """

    EVAL_MODE = "batch_level"

    def __init__(self, backbone: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.bb = backbone
        self.config = config or {}
        self.n_pc = int(self.config.get("n_pc", 20))
        self.batch_size = self.config.get("batch_size")  # optional, caps n_pc
        self.pca: Optional[PCA] = None
        self.mu_b: Optional[np.ndarray] = None
        self.cov_b: Optional[np.ndarray] = None

    # -- embedding via the shared frozen backbone -------------------------------
    def _embed(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 4:  # (M, B, C, T) -> flatten batch dim; caller passes (B,C,T)
            arr = arr.reshape(-1, *arr.shape[2:])
        return np.asarray(self.bb.embed(arr), dtype=np.float64)

    def fit(self, id_windows: np.ndarray) -> "DriftLensClassD":
        E = self._embed(id_windows)                       # (N, D)
        n = E.shape[0]
        n_pc = min(self.n_pc, E.shape[1], max(1, n - 1))
        if self.batch_size is not None:
            n_pc = min(n_pc, max(1, int(self.batch_size) - 2))
        n_pc = max(1, n_pc)
        self.pca = PCA(n_components=n_pc, random_state=42).fit(E)
        Er = self.pca.transform(E)                        # (N, n_pc)
        self.mu_b = Er.mean(axis=0)
        self.cov_b = np.cov(Er, rowvar=False)
        self.cov_b = np.atleast_2d(self.cov_b)
        return self

    def score_batch(self, batch: np.ndarray) -> float:
        """batch: (B, C, T) -> scalar Fréchet distance to the ID baseline."""
        if self.pca is None:
            raise RuntimeError("DriftLensClassD must be fit before scoring.")
        E = self._embed(batch)                            # (B, D)
        if len(E) < 2:
            return float("nan")
        Er = self.pca.transform(E)                        # (B, n_pc)
        mu_w = Er.mean(axis=0)
        cov_w = np.atleast_2d(np.cov(Er, rowvar=False))
        return frechet_distance(self.mu_b, mu_w, self.cov_b, cov_w)
