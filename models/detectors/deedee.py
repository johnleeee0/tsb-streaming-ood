"""Faithful DEEDEE detector (flattened, canonical).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

DEEDEE: Fast and Scalable Out-of-Distribution Dynamics Detection (adapted). Based on
Aljaafari et al., arXiv:2510.21638 (2025).

This file is SELF-CONTAINED: the corrected variant (formerly methods/deedee/deedee_fix)
is the canonical, registered ``DEEDEEDetector``, with the small base ``__init__``
(formerly models/ood_methods/deedee.py) inlined. Behaviour is identical to the
deedee_fix variant used in the benchmark method_set.

The corrected variant computes the two DEEDEE statistics over the window's real time
axis: the episodewise mean (mean over time, per channel) for global shift, and an RBF
self-similarity over consecutive timesteps for local dynamics. An isolation forest is
fitted on these per-window trajectory statistics from the in-distribution windows; the
(negated) isolation-forest score is the OOD score (higher = more OOD). Operates on the
raw windows, as in the paper, not on backbone features.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("deedee")
class DEEDEEDetector(BaseOODDetector):
    """DEEDEE (trajectory-statistic + isolation forest) OOD detection.

    Config keys (all optional):
        window_size     (int, 10)      – trajectory window size (unused by the
                                         time-axis statistics; kept for compatibility)
        sigma           (float, 1.0)   – RBF kernel bandwidth
        s               (float, 1.5)   – RBF scaling factor
        n_estimators    (int, 100)     – isolation forest trees
        contamination   (float, 0.1)   – expected outlier fraction
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.window_size: int = int(self.config.get("window_size", 10))
        self.sigma: float = float(self.config.get("sigma", 1.0))
        self.s: float = float(self.config.get("s", 1.5))
        self.n_estimators: int = int(self.config.get("n_estimators", 100))
        self.contamination: float = float(self.config.get("contamination", 0.1))

        self.feat_dim: Optional[int] = None
        self._iforest: Optional[IsolationForest] = None

    def _traj_stats(self, x: np.ndarray) -> np.ndarray:
        """x: (N, C, T) raw windows -> (N, 2C) trajectory statistics.
        For each channel: episodewise mean over time, and mean RBF self-similarity
        between consecutive timesteps (local dynamics)."""
        N, C, T = x.shape
        mean_stat = x.mean(axis=2)  # (N, C) global level over the trajectory
        if T > 1:
            d = (x[:, :, 1:] - x[:, :, :-1]) ** 2  # squared step differences (N, C, T-1)
            rbf = (self.s * np.exp(-d / (self.sigma ** 2))).mean(axis=2)  # (N, C) local dynamics
        else:
            rbf = np.full((N, C), self.s, dtype=np.float64)
        return np.concatenate([mean_stat, rbf], axis=1)  # (N, 2C)

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        x = np.asarray(x_id["x"] if isinstance(x_id, dict) else x_id, dtype=np.float64)
        feats = self._traj_stats(x)
        self.feat_dim = feats.shape[1]
        self._iforest = IsolationForest(
            n_estimators=self.n_estimators, contamination=self.contamination, random_state=42
        ).fit(feats)

    def score(self, x: Any) -> np.ndarray:
        if self._iforest is None:
            raise RuntimeError("DEEDEE must be fit before scoring.")
        xx = np.asarray(x["x"] if isinstance(x, dict) else x, dtype=np.float64)
        feats = self._traj_stats(xx)
        # score_samples: higher = more normal; negate so higher = more OOD
        return (-self._iforest.score_samples(feats)).astype(np.float64)
