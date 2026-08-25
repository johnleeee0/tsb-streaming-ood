"""
CODiT: Conformal out-of-distribution detection for time series.

Paper: Kaur, Sridhar, Jha, Roy, Sokolsky & Lee, *CODiT: Conformal Out-of-Distribution
       Detection in Time-series Data for Cyber-Physical Systems*, ICCPS 2022
       (arXiv:2207.11769)
Code:  https://github.com/kaustubhsridhar/time-series-OOD
       (local clone: methods/codit/reference/ours/check_OOD_carla.py)

Faithful reproduction of the official per-window scoring pipeline, adapted to the
frozen-backbone / per-window benchmark protocol:

  1. A K-way transformation classifier (linear head on frozen backbone features)
     is trained to predict which of the five time-series transformations
     (identity, reverse, shuffle, periodic, speed) was applied to a window.
  2. Calibration nonconformity scores are built from a held-out ID split: for each
     of ``eval_n`` draws, every calibration window receives a *randomly sampled*
     transform and the cross-entropy loss against that transform's label is
     recorded (``calc_cal_ce_loss``, reference lines 129-168).
  3. At test time each window is scored with ``eval_n`` independent random-transform
     draws. Each draw yields a conformal p-value
     ``(#{cal_loss >= test_loss} + 1) / (n_cal + 1)`` (``calc_p_value``, ref 170-184).
  4. The ``eval_n`` p-values are multiplied into a single product and combined with
     the Fisher statistic ``F = prod * sum_{i=0}^{eval_n-1} (-log prod)^i / i!``
     (``calc_fisher_value`` / ``calc_fisher_batch``, ref 267-283). The term count
     matches the number of p-values combined.
  5. Orientation follows the official ``getAUROC`` (ref 315-324): ID = 1, OOD = 0,
     Fisher passed unnegated, so **higher Fisher = more in-distribution**. This
     detector's public contract is "higher = more OOD", so the returned score is
     ``-Fisher``.

Divergences from the official code, disclosed:
  * The transform classifier is a linear head on frozen backbone features rather
    than an end-to-end ``r3d_regressor`` over (orig, transformed) clip pairs.
  * The benchmark scores independent per-window inputs, so the official run-length
    detection over an ordered trace has no analogue and is omitted.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.registry import register_ood
from core.base_ood import BaseOODDetector


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

# Order matches the official default list
# (check_OOD_carla.py:49 -> ["speed","shuffle","reverse","periodic","identity"]).
# The set is identical; ordering only fixes the integer class labels.
_TRANSFORM_NAMES = ["identity", "reverse", "shuffle", "periodic", "speed"]


def _apply_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    """Apply a single named transformation to a batch of time-series.

    Args:
        x: (B, C, T) or (C, T) float tensor.
        name: One of the five transformation names.

    Returns:
        Transformed tensor with the same shape as x.
    """
    single = x.dim() == 2
    if single:
        x = x.unsqueeze(0)

    B, C, T = x.shape

    if name == "identity":
        out = x.clone()

    elif name == "reverse":
        out = torch.flip(x, dims=[-1])

    elif name == "shuffle":
        idx = torch.randperm(T, device=x.device)
        out = x[:, :, idx]

    elif name == "periodic":
        # Circular shift by a random offset
        offset = torch.randint(1, max(2, T), (1,)).item()
        out = torch.roll(x, shifts=int(offset), dims=-1)

    elif name == "speed":
        # Subsample by factor 2, then upsample back to original length
        # Simulates a speed-doubled (shorter effective window) signal
        x_f = x.float()
        # (B, C, T) → treat (B*C, 1, T) for F.interpolate
        bc = B * C
        x_r = x_f.reshape(bc, 1, T)
        down = F.interpolate(x_r, size=max(2, T // 2), mode="linear", align_corners=False)
        up = F.interpolate(down, size=T, mode="linear", align_corners=False)
        out = up.reshape(B, C, T)

    else:
        raise ValueError(f"Unknown transformation: {name}")

    return out.squeeze(0) if single else out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@register_ood("codit")
class CODiTDetector(BaseOODDetector):
    """CODiT OOD detector.

    Config keys (all optional):
        n_epochs   (int, 30)    – epochs for training the transform head.
        lr         (float, 1e-3) – learning rate.
        batch_size (int, 64)    – training / scoring batch size.
        eval_n     (int, 20)    – number of random-transform draws combined by
                                  Fisher (official ``--n`` default = 20).
        cal_frac   (float, 0.2) – fraction of ID train held out for calibration.
        seed       (int, 42)    – seed for the calibration split and the random
                                  transform draws (official uses manual_seed(42)).
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.n_transforms: int = len(_TRANSFORM_NAMES)
        self.n_epochs: int = int(self.config.get("n_epochs", 30))
        self.lr: float = float(self.config.get("lr", 1e-3))
        self.batch_size: int = int(self.config.get("batch_size", 64))
        self.eval_n: int = int(self.config.get("eval_n", 20))
        self.cal_frac: float = float(self.config.get("cal_frac", 0.2))
        self.seed: int = int(self.config.get("seed", 42))

        self.transform_head: Optional[nn.Linear] = None
        # Calibration nonconformity losses, shape (eval_n, n_cal).
        self.cal_losses: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # internal: one random-transform draw over a set of windows
    # ------------------------------------------------------------------

    def _random_transform_ce(
        self, x_tensor: torch.Tensor, np_rng: np.random.Generator
    ) -> np.ndarray:
        """Assign each window a random transformation, apply it, and return the
        per-window cross-entropy loss against that transform's label.

        Mirrors the official nonconformity: CE against the *actually applied*
        (randomly sampled) transform (reference lines 109, 145).

        Args:
            x_tensor: (M, C, T) windows on ``self.device``.
            np_rng:   numpy RNG driving the per-window transform assignment.

        Returns:
            (M,) float64 array of CE losses.
        """
        M = len(x_tensor)
        assign = np_rng.integers(0, self.n_transforms, size=M)
        losses = np.empty(M, dtype=np.float64)

        self.transform_head.eval()
        with torch.no_grad():
            for t_idx, t_name in enumerate(_TRANSFORM_NAMES):
                sel = np.nonzero(assign == t_idx)[0]
                if len(sel) == 0:
                    continue
                sel_t = torch.from_numpy(sel).to(self.device)
                sub = x_tensor.index_select(0, sel_t)
                x_t = _apply_transform(sub, t_name)
                for i in range(0, len(x_t), self.batch_size):
                    xb = x_t[i : i + self.batch_size]
                    feats = self._forward_features(xb)
                    logits = self.transform_head(feats)
                    targets = torch.full(
                        (len(xb),), t_idx, dtype=torch.long, device=self.device
                    )
                    ce = F.cross_entropy(logits, targets, reduction="none")
                    losses[sel[i : i + len(xb)]] = ce.cpu().numpy()
        return losses

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """Train the transformation head and build the calibration nonconformity set.

        Args:
            x_id: ID training data, shape (N, C, T).
            y_id: Ignored (transformation labels are self-generated).
        """
        x_tensor = self._to_tensor(x_id)  # (N, C, T)
        N = len(x_tensor)

        # Infer embedding dimension from backbone
        with torch.no_grad():
            sample_feat = self._forward_features(x_tensor[:2])
        feat_dim: int = sample_feat.shape[1]

        # Build transformation head
        self.transform_head = nn.Linear(feat_dim, self.n_transforms).to(self.device)
        optimizer = torch.optim.Adam(self.transform_head.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # Deterministic calibration split (official: manual_seed(42), ref line 192).
        split_gen = torch.Generator().manual_seed(self.seed)
        n_cal = max(1, int(N * self.cal_frac))
        n_train = max(1, N - n_cal)
        n_cal = N - n_train
        if n_cal < 1:  # tiny-N fallback: reuse train data for calibration
            n_train, n_cal = N, N
            perm = torch.randperm(N, generator=split_gen)
            idx_train = perm
            idx_cal = perm
        else:
            perm = torch.randperm(N, generator=split_gen)
            idx_train = perm[:n_train]
            idx_cal = perm[n_train:]

        x_train = x_tensor[idx_train]
        x_cal = x_tensor[idx_cal]

        # Seed torch so the augmented-set transforms and training shuffle are
        # reproducible run-to-run.
        torch.manual_seed(self.seed)

        # Build augmented training set: every train sample × every transformation.
        x_aug_list: List[torch.Tensor] = []
        y_aug_list: List[int] = []
        for t_idx, t_name in enumerate(_TRANSFORM_NAMES):
            x_t = _apply_transform(x_train, t_name)
            x_aug_list.append(x_t)
            y_aug_list.extend([t_idx] * len(x_train))

        x_aug = torch.cat(x_aug_list, dim=0)
        y_aug = torch.tensor(y_aug_list, dtype=torch.long, device=self.device)

        ds = TensorDataset(x_aug, y_aug)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Train transformation head (backbone frozen)
        self.model.eval()
        for _ in range(self.n_epochs):
            self.transform_head.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                with torch.no_grad():
                    feats = self._forward_features(xb)
                logits = self.transform_head(feats)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Build calibration nonconformity: eval_n independent random-transform
        # draws over the calibration split (official calc_cal_ce_loss, ref 145-168).
        self.transform_head.eval()
        cal_rng = np.random.default_rng(self.seed)
        cal_losses = np.empty((self.eval_n, len(x_cal)), dtype=np.float64)
        for k in range(self.eval_n):
            torch.manual_seed(self.seed + 101 * (k + 1))  # transform-internal RNG
            cal_losses[k] = self._random_transform_ce(x_cal, cal_rng)
        self.cal_losses = cal_losses

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, x: Any) -> np.ndarray:
        """Return OOD scores for x.  Higher = more OOD.

        For each window: draw ``eval_n`` random transforms → one conformal p-value
        each → multiply into a product → Fisher combine over ``eval_n`` terms →
        negate (official orientation is higher-Fisher = more ID).

        Args:
            x: Input data (N, C, T).

        Returns:
            ood_scores: (N,) float array; higher value → more OOD.
        """
        if self.transform_head is None or self.cal_losses is None:
            raise RuntimeError("CODiT must be fit before scoring.")

        x_tensor = self._to_tensor(x)
        N = len(x_tensor)
        n_cal = self.cal_losses.shape[1]

        self.transform_head.eval()
        test_rng = np.random.default_rng(self.seed + 999)

        # Product of the eval_n per-draw conformal p-values (per test window).
        p_prod = np.ones(N, dtype=np.float64)
        for k in range(self.eval_n):
            torch.manual_seed(self.seed + 999 + 101 * (k + 1))  # transform-internal RNG
            test_losses_k = self._random_transform_ce(x_tensor, test_rng)  # (N,)
            cal_k = self.cal_losses[k]  # (n_cal,)
            # p = (#{test_loss <= cal_loss} + 1) / (n_cal + 1)  (ref 179-181)
            compare = test_losses_k[:, None] <= cal_k[None, :]
            p_k = (np.sum(compare, axis=1) + 1.0) / (n_cal + 1.0)
            p_prod *= p_k

        # Fisher combination over eval_n terms (ref calc_fisher_value, 267-271).
        fisher_vals = self._calc_fisher_value(p_prod, self.eval_n)  # (N,)

        # Official orientation: higher Fisher = more ID. This detector returns
        # higher = more OOD, so negate.
        return -fisher_vals

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_fisher_value(t_value: np.ndarray, eval_n: int) -> np.ndarray:
        """Fisher combination statistic (vectorised over test windows).

        F(t) = t * sum_{i=0}^{eval_n-1} (-log t)^i / i!

        This is the survival-based combination used by the official
        ``calc_fisher_value`` (reference lines 267-271), where ``t`` is the
        product of ``eval_n`` p-values and the term count equals ``eval_n``.

        Args:
            t_value: (N,) array of p-value products, each in (0, 1].
            eval_n:  number of p-values combined (Fisher term count).

        Returns:
            (N,) array of Fisher values.
        """
        t = np.clip(np.asarray(t_value, dtype=np.float64), 1e-300, 1.0)
        log_t = -np.log(t)
        summation = np.zeros_like(t)
        power = np.ones_like(t)  # (-log t)^0
        for i in range(eval_n):
            summation += power / math.factorial(i)
            power *= log_t
        return t * summation
