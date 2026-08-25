"""TD-IVDM-inspired (Class-D appendix build) — ordered multi-scale density.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Caption (binding): "TD-IVDM-inspired (unverifiable — paper paywalled, no code)".

TD-IVDM ("Time Dependency – Inter Variable Dependency"; Wang et al.,
Neurocomputing 2025, doi:10.1016/j.neucom.2025.131120) is described by its
authors as combining an improved TS2Vec representation branch (time dependencies)
with a multi-dimensional KDE (inter-variable dependencies), applied MULTI-SCALE
over smaller time frames and variable subsets. The paper is paywalled (HTTP 403
on ScienceDirect and ResearchGate) and no public code exists, so a faithful
reproduction cannot be verified — see methods/tdivdm/VERIFICATION.md. This build
is therefore an honest reconstruction of the method's SHAPE, not the method:

  * time-dependency representation  -> the shared frozen backbone embedding stands
    in for the (unavailable) improved-TS2Vec branch (disclosed substitution);
  * inter-variable KDE             -> Gaussian KDE (multi-dimensional) on the
    embedding, exactly the pillar the production impl already had;
  * MULTI-SCALE (the title contribution, absent from the production impl)
    -> the density is evaluated in several PCA subspaces of increasing budget
       (coarse -> fine), and the per-window negative log-densities are
       standardised per scale and aggregated. This gives a genuinely multi-scale
       density score while remaining self-contained.

Ordered stream -> per-window negative log-density -> per_sample_auroc directly
(higher = more OOD).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


class TDIVDMClassD:
    """Multi-scale KDE density detector on shared frozen-backbone embeddings.

    Interface:
      fit(id_windows)          -> fit one PCA+KDE per scale on ID embeddings
      score_stream(stream_x)   -> per-window aggregated neg-log-density (higher = OOD)

    Params (config):
      scales    : list of PCA component budgets (default [5, 10, 20]); each is a
                  "scale" (coarse -> fine) at which the density is measured.
      bandwidth : 'scott' | 'silverman' | float (default 'scott')
    """

    EVAL_MODE = "ordered_per_window"

    def __init__(self, backbone: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.bb = backbone
        cfg = config or {}
        self.scales: List[int] = list(cfg.get("scales", [5, 10, 20]))
        self.bandwidth = cfg.get("bandwidth", "scott")
        self.seed = int(cfg.get("seed", 42))
        self._models: List[Dict[str, Any]] = []   # per-scale {pca, kde, mu, sd}

    def _embed(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.bb.embed(np.asarray(x, dtype=np.float32)), dtype=np.float64)

    @staticmethod
    def _bw(bandwidth: Any, n: int, d: int) -> float:
        """Resolve a scalar KDE bandwidth (KernelDensity needs a float)."""
        if isinstance(bandwidth, (int, float)):
            return float(bandwidth)
        n = max(n, 2)
        if bandwidth == "silverman":
            return float(n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
        # 'scott'
        return float(n) ** (-1.0 / (d + 4))

    def fit(self, id_windows: np.ndarray) -> "TDIVDMClassD":
        E = self._embed(id_windows)                         # (N, D)
        n = E.shape[0]
        self._models = []
        for k in self.scales:
            n_pc = max(1, min(int(k), E.shape[1], max(1, n - 1)))
            pca = PCA(n_components=n_pc, whiten=True, random_state=self.seed).fit(E)
            Er = pca.transform(E)                            # (N, n_pc)
            bw = self._bw(self.bandwidth, n, n_pc)
            kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(Er)
            # per-scale standardisation of the training neg-log-density, so scales
            # of different dimensionality are comparable before aggregation.
            nld = -kde.score_samples(Er)
            mu = float(nld.mean())
            sd = float(nld.std()) + 1e-6
            self._models.append({"pca": pca, "kde": kde, "mu": mu, "sd": sd})
        return self

    def score_stream(self, stream_x: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("TDIVDMClassD must be fit before scoring.")
        E = self._embed(stream_x)                            # (N, D)
        acc = np.zeros(len(E), dtype=np.float64)
        for m in self._models:
            Er = m["pca"].transform(E)
            nld = -m["kde"].score_samples(Er)                # (N,) neg log-density
            acc += (nld - m["mu"]) / m["sd"]                 # standardised, aggregated
        return acc / max(len(self._models), 1)               # higher = more OOD

    # convenience alias so batch/per-window callers share one entry point
    def score(self, x: np.ndarray) -> np.ndarray:
        return self.score_stream(x)
