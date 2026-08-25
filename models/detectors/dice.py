"""Faithful DICE detector (dice_enh).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Faithful reproduction of DICE (Sun & Li, *DICE: Leveraging Sparsification for
Out-of-Distribution Detection*, ECCV 2022, arXiv:2111.09805), verified against the
official repository (deeplearning-wisc/dice, `models/route.py::RouteDICE`).

DICE's defining mechanism is **directed sparsification of the classification-head
weights**:

1. Precompute a single ID-mean feature vector `info` over the in-distribution
   training set (official `precompute.py:88`: `feat_log.mean(0)`).
2. Form the *input-independent* contribution matrix `contrib = info[None, :] * W`
   for the head weights `W` (C, D), rank the **signed** contributions (the official
   `np.abs` is explicitly commented out, `route.py:18`), threshold at a **global**
   percentile `p` over the whole `(C, D)` matrix, and multiply the resulting binary
   mask into the **weights** to obtain a static sparsified head `masked_w`
   (`route.py:16-24`).
3. Score every test input through the *same* sparsified head with the energy score
   (`util/score.py:93`: `logsumexp`).

This is built here in `fit(x_id, y_id)` (auto-called by the runner before `score()`),
and `score()` simply applies the static sparsified head and returns energy.

Differences from the previous `dice_enh`, which was NOT the method: it computed the
contribution per *test sample*, sparsified nothing (recomputed logits per sample),
ranked by **absolute** magnitude (admitting the large-negative units the official mask
excludes), used a fixed per-class `k=20` instead of a global percentile, and never
implemented `fit()` (used no ID statistics at all). See VERIFICATION.md.

Orientation follows the project convention (higher = OOD) via `_energy` returning
`-logsumexp`; this is a rank-invariant sign flip of the paper's `+logsumexp`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("dice")
class DICEEnhDetector(BaseOODDetector):
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        # Sparsity level as a global percentile over the contribution matrix.
        # Paper/official default p = 90 (route.py:8). `top_k` is retained only for
        # backward-compatible config parsing and is NOT used by the faithful path.
        self.p = float(self.config.get("percentile", self.config.get("p", 90.0)))
        self.top_k = int(self.config.get("top_k", 20))
        self.temperature = float(self.config.get("temperature", 1.0))
        # Static sparsified head, built once in fit().
        self._masked_w: Optional[torch.Tensor] = None  # (C, D)
        self._bias: Optional[torch.Tensor] = None       # (C,)

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """Build the STATIC ID-mean contribution mask and sparsify the head WEIGHTS.

        contrib = info[None, :] * W   (signed, input-independent)
        thresh  = percentile(contrib, p)   (single global threshold over (C, D))
        mask    = contrib > thresh
        masked_w = W * mask
        """
        clf = self._classifier()
        if clf is None or not hasattr(clf, "weight"):
            # No head available: nothing to sparsify; score() falls back to energy
            # on raw features (documented fallback).
            self._masked_w = None
            self._bias = None
            return None

        x_tensor = self._to_tensor(x_id)
        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, D)
            info = feats.mean(dim=0)                   # (D,) ID-mean feature vector
            weight = clf.weight.detach()              # (C, D)
            info_np = info.detach().cpu().numpy().reshape(1, -1)  # (1, D)
            w_np = weight.cpu().numpy()               # (C, D)

            contrib = info_np * w_np                  # (C, D) signed, input-independent
            thresh = np.percentile(contrib, self.p)   # global percentile threshold
            mask = (contrib > thresh).astype(np.float32)  # one-sided, signed

            masked_w = torch.from_numpy(w_np * mask).to(
                device=weight.device, dtype=weight.dtype
            )
            self._masked_w = masked_w
            if clf.bias is not None:
                self._bias = clf.bias.detach().to(weight.device)
            else:
                self._bias = torch.zeros(
                    weight.size(0), device=weight.device, dtype=weight.dtype
                )
        return None

    def score(self, x: Any) -> np.ndarray:
        x_tensor = self._to_tensor(x)
        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (B, D)

            if self._masked_w is None:
                clf = self._classifier()
                if clf is None or not hasattr(clf, "weight"):
                    # No head: energy on raw features as a documented fallback.
                    return (-torch.logsumexp(feats / self.temperature, dim=-1)).detach().cpu().numpy()
                # fit() was not run (e.g. no ID data) but a head exists: build the
                # mask lazily from the current batch's mean so the detector is still
                # usable. This is a safety fallback, not the intended path.
                self.fit(x)

            masked_w = self._masked_w.to(device=feats.device, dtype=feats.dtype)  # (C, D)
            bias = self._bias.to(device=feats.device, dtype=feats.dtype)          # (C,)
            logits = feats @ masked_w.t() + bias.unsqueeze(0)                     # (B, C)
            scores = self._energy(logits, temperature=self.temperature)          # higher = OOD
        return scores.detach().cpu().numpy()
