from __future__ import annotations

"""Online / incremental-covariance Mahalanobis detector (true streaming).

Author: Stylianos Giannoulis - AUTH MSc Data and Web Science - Supervisor: John Paparrizos

Motivation
----------
The batch Mahalanobis detector (models/detectors/mahalanobis.py) fits a mean +
tied covariance ONCE on the in-distribution (Source-1) training windows and then
scores every test window against that frozen estimate. This module tests the
thesis's headline recommendation ("deploy Mahalanobis") under a GENUINE streaming
protocol: the estimate is warm-started on the ID training windows and then updated
INCREMENTALLY as the detector consumes the temporally-ordered evaluation stream,
one window (or mini-batch) at a time.

Protocol (score-then-update)
----------------------------
For each streamed window (or mini-batch), in temporal order:
  1. SCORE the window against the CURRENT running estimate (Mahalanobis distance
     to the running ID mean; higher = more OOD). This is the deployment-time
     decision, made before the detector has "seen the answer".
  2. UPDATE the running mean + covariance with that window using a numerically
     stable incremental rule.

This ordering guarantees no window is ever scored against an estimate that already
absorbed it (no look-ahead leakage).

Incremental estimator (decayed sufficient statistics)
-----------------------------------------------------
We maintain exponentially-decayed weighted sums (a single running Gaussian):

    w      = decay * w      + 1
    s1     = decay * s1     + f
    s2     = decay * s2     + outer(f, f)

with running mean  mu  = s1 / w  and covariance
    Sigma = s2 / w - outer(mu, mu).

  * decay == 1.0  -> pure incremental / growing estimate (Welford-equivalent, no
                     forgetting): every window ever seen contributes equally.
  * decay  < 1.0  -> exponential forgetting with effective window ~ 1/(1-decay),
                     so the estimate TRACKS SLOW DRIFT in the ID distribution.

The default (decay=0.999, effective window ~1000 windows) adapts slowly enough to
be robust to short anomaly bursts yet fast enough to follow gradual drift. A
sliding-window mode ("window") is also provided. Covariance is regularised with a
ridge (matching the batch detector's 1e-6) plus an optional shrinkage toward the
scaled identity for stability while the effective count is small.

Single running Gaussian vs per-class tied covariance
----------------------------------------------------
The batch detector keeps per-pseudo-class means with a tied within-class
covariance and scores min-distance over classes. The eval STREAM carries no class
labels, so the streaming analogue is a SINGLE running ID Gaussian (one running
mean + one running covariance). The warm-start therefore uses the pooled ID
training features (global mean + tied within-class covariance), and streaming
updates a single Gaussian. This is documented as an intentional, faithful
streaming reduction of the batch method.

Unsupervised update (anomaly contamination)
-------------------------------------------
IMPORTANT: at deployment the detector does NOT know which streamed windows are
anomalies, so the running estimate is updated from ALL streamed windows
(unsupervised streaming), NOT only the ID ones. This is the honest streaming
setting: contaminating the estimate with the occasional anomaly is a real cost of
online adaptation, and the decay/window forgetting is what keeps that cost bounded.
Filtering updates by the (unknown) ground-truth labels would leak the answer.

Feature path
------------
Features are the pre-logit backbone embeddings via the shared BaseOODDetector
._forward_features path -- identical to the batch Mahalanobis detector, so any
batch-vs-online difference is due to the estimator, not the representation.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from core.base_ood import BaseOODDetector
from core.registry import register_ood


@register_ood("online_mahalanobis")
class OnlineMahalanobisDetector(BaseOODDetector):
    """Streaming Mahalanobis with incremental mean + covariance.

    Config keys (all optional):
      decay      : forgetting factor in (0, 1].  1.0 = pure incremental (no
                   forgetting); <1.0 = exponential forgetting (default 0.999).
      mode       : "decay" (exponential forgetting, default) or "window"
                   (sliding window of the last `window_len` streamed features).
      window_len : sliding-window length when mode == "window" (default 500).
      ridge      : diagonal ridge added to the covariance (default 1e-6, matches
                   the batch detector).
      shrinkage  : shrinkage toward the scaled identity in [0, 1) (default 0.02),
                   blended as (1-s)*Sigma + s*mean(diag(Sigma))*I for stability
                   while the effective count is small.
      mini_batch : number of consecutive windows scored against the current
                   estimate before a single joint update (default 1 = per-window
                   true streaming).
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.decay: float = float(self.config.get("decay", 0.999))
        self.mode: str = str(self.config.get("mode", "decay"))
        self.window_len: int = int(self.config.get("window_len", 500))
        self.ridge: float = float(self.config.get("ridge", 1e-6))
        self.shrinkage: float = float(self.config.get("shrinkage", 0.02))
        self.mini_batch: int = int(self.config.get("mini_batch", 1))
        self._feat_batch: int = int(self.config.get("feat_batch", 256))

        # Running decayed sufficient statistics (initialised in fit()).
        self._w: float = 0.0                       # decayed weight (effective count)
        self._s1: Optional[np.ndarray] = None      # decayed sum of features (D,)
        self._s2: Optional[np.ndarray] = None      # decayed sum of outer products (D,D)
        self._dim: Optional[int] = None
        # Sliding-window buffer (mode == "window").
        self._buffer: list = []
        # Warm-start snapshot (so score() on a fresh input reuses the fitted state).
        self._warm: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Feature extraction (shared backbone path, chunked)
    # ------------------------------------------------------------------
    def _features(self, x: Any) -> np.ndarray:
        x_tensor = self._to_tensor(x)
        feats = []
        n = x_tensor.shape[0]
        with torch.no_grad():
            for s in range(0, n, self._feat_batch):
                chunk = x_tensor[s:s + self._feat_batch]
                f = self._forward_features(chunk).detach().cpu().numpy()
                feats.append(np.asarray(f, dtype=np.float64))
        return np.concatenate(feats, axis=0) if feats else np.empty((0, 0))

    # ------------------------------------------------------------------
    # Warm-start on the ID training windows
    # ------------------------------------------------------------------
    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """Warm-start the running mean + covariance from the ID training windows.

        The warm-start reproduces the batch detector's ID statistics: the running
        mean is the pooled ID mean and the running covariance is the tied
        within-class covariance (features centred on their pseudo-class mean before
        pooling, exactly as the batch detector) when class labels are supplied;
        otherwise the pooled empirical covariance about the global mean.
        """
        feats = self._features(x_id)
        if feats.ndim != 2 or feats.shape[0] == 0:
            raise ValueError("OnlineMahalanobis.fit received no usable features.")
        n, d = feats.shape
        self._dim = d

        global_mean = feats.mean(axis=0)

        # Tied within-class scatter (matches batch mahalanobis) if labels given.
        if y_id is not None:
            if isinstance(y_id, torch.Tensor):
                y_id = y_id.cpu().numpy()
            y_id = np.asarray(y_id)
            centred = np.empty_like(feats)
            for c in np.unique(y_id):
                m = (y_id == c)
                centred[m] = feats[m] - feats[m].mean(axis=0)
            cov = (centred.T @ centred) / max(n, 1)
        else:
            dev = feats - global_mean
            cov = (dev.T @ dev) / max(n, 1)

        # Seed the decayed sufficient statistics so that, at t=0, mu == global_mean
        # and Sigma == cov (i.e. the running estimate equals the batch estimate).
        self._w = float(n)
        self._s1 = global_mean * n
        # s2 = w*(cov + outer(mu, mu))  ->  s2/w - outer(mu, mu) == cov
        self._s2 = n * (cov + np.outer(global_mean, global_mean))

        if self.mode == "window":
            # Seed the sliding buffer with the ID features (kept bounded).
            self._buffer = list(feats[-self.window_len:])

        self._warm = {
            "w": self._w,
            "s1": self._s1.copy(),
            "s2": self._s2.copy(),
            "buffer": list(self._buffer),
        }

    # ------------------------------------------------------------------
    # Running-estimate helpers
    # ------------------------------------------------------------------
    def _current_mean_precision(self):
        """Return (mu, precision) from the current running statistics."""
        if self.mode == "window":
            buf = np.asarray(self._buffer, dtype=np.float64)
            mu = buf.mean(axis=0)
            dev = buf - mu
            cov = (dev.T @ dev) / max(len(buf), 1)
        else:
            mu = self._s1 / max(self._w, 1e-12)
            cov = self._s2 / max(self._w, 1e-12) - np.outer(mu, mu)

        cov = 0.5 * (cov + cov.T)  # symmetrise (guards FP drift)
        if self.shrinkage > 0.0:
            diag_mean = float(np.mean(np.diag(cov)))
            cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag_mean * np.eye(self._dim)
        cov = cov + self.ridge * np.eye(self._dim)
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov)
        return mu, precision

    def _update(self, f: np.ndarray) -> None:
        """Fold one streamed feature vector into the running statistics."""
        if self.mode == "window":
            self._buffer.append(f)
            if len(self._buffer) > self.window_len:
                self._buffer = self._buffer[-self.window_len:]
        else:
            g = self.decay
            self._w = g * self._w + 1.0
            self._s1 = g * self._s1 + f
            self._s2 = g * self._s2 + np.outer(f, f)

    @staticmethod
    def _mahal(feats: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
        delta = feats - mu
        d2 = np.einsum("ij,jk,ik->i", delta, precision, delta)
        return np.sqrt(np.maximum(d2, 0.0))

    # ------------------------------------------------------------------
    # Streaming scorer (score-then-update, temporal order preserved)
    # ------------------------------------------------------------------
    def score_stream(self, stream_x: Any) -> np.ndarray:
        """Score a temporally-ordered window stream, updating after each step.

        Returns per-window Mahalanobis scores (higher = more OOD) in input order.
        The running estimate is warm-started (fit) and then updated from EVERY
        streamed window (unsupervised) after that window has been scored.
        """
        if self._s1 is None and self.mode != "window":
            raise RuntimeError("OnlineMahalanobis must be fit before scoring.")
        feats = self._features(stream_x)
        n = feats.shape[0]
        scores = np.empty(n, dtype=np.float64)

        mb = max(1, self.mini_batch)
        for s in range(0, n, mb):
            e = min(s + mb, n)
            mu, precision = self._current_mean_precision()
            scores[s:e] = self._mahal(feats[s:e], mu, precision)
            for i in range(s, e):
                self._update(feats[i])
        return scores

    def reset_to_warm_start(self) -> None:
        """Restore the running estimate to the post-fit warm-start snapshot."""
        if self._warm is None:
            raise RuntimeError("reset_to_warm_start called before fit().")
        self._w = self._warm["w"]
        self._s1 = self._warm["s1"].copy()
        self._s2 = self._warm["s2"].copy()
        self._buffer = list(self._warm["buffer"])

    # ------------------------------------------------------------------
    # BaseOODDetector API: score() runs a fresh stream pass from the warm start.
    # ------------------------------------------------------------------
    def score(self, x: Any) -> np.ndarray:
        """Score `x` as a stream from the warm-start state (non-mutating).

        Provided for BaseOODDetector compatibility. Each call resets to the
        post-fit warm start first, so repeated score() calls are deterministic.
        """
        self.reset_to_warm_start()
        return self.score_stream(x)
