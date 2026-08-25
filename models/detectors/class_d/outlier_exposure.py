"""Outlier Exposure (Class-D appendix build) — FAITHFUL OE, both fine-tune arms.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Faithful re-build of Outlier Exposure (Hendrycks, Mazeika & Dietterich, ICLR 2019;
repo hendrycks/outlier-exposure, CIFAR/oe_tune.py), per
methods/_validation/CLASS_D_DECISIONS.md and BUILD_PLAN_CLASS_D.md §1.

OE fine-tunes the classifier with

    L = CE(f(x_id), y_id) + lambda * CE_to_uniform(f(x_out))

where the OE term is  -(mean_k z_k - logsumexp_k z_k)  on the AUXILIARY-outlier
half of each batch (reference/CIFAR/oe_tune.py:172-177; lambda = 0.5). After a few
epochs the detector scores test windows with the ENERGY score -logsumexp(z/T) on
the FINE-TUNED net (higher = more OOD).

This module is a THIN ENERGY SCORER over an already-fine-tuned (backbone, head).
The fine-tuning itself is done by the appendix runner's `finetune()` helper
(experiments/run_class_d.py) which:
  * deep-copies the shared frozen backbone (so the 17 production methods' anchor
    is NEVER mutated), and
  * runs BOTH arms — 'head_only' (frozen ResNet, least-unfair vs the frozen-backbone
    17) and 'full_net' (paper-faithful) — reporting the pair.

So `OutlierExposureClassD.fit()` accepts the ALREADY-fine-tuned (backbone, head)
passed in via the config and is a no-op; `score()` returns the per-window energy.
This is an appendix-only, fair-comparison-BREAKING study (OE updates weights the 17
methods do not have) — never a row in the 17-method leaderboard.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn


def energy_scores(
    model: nn.Module,
    head: nn.Module,
    x: np.ndarray,
    temperature: float = 1.0,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Per-window energy score  -logsumexp(head(model(x)) / T)  (higher = more OOD).

    Runs the fine-tuned (backbone, head) in eval mode, batched, no grad.
    """
    model.eval()
    head.eval()
    model.to(device)
    head.to(device)
    arr = np.asarray(x, dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(arr), batch_size):
            xb = torch.from_numpy(arr[i:i + batch_size]).float().to(device)
            logits = head(model(xb))
            e = -torch.logsumexp(logits / temperature, dim=-1)
            out.append(e.detach().cpu().numpy())
    if not out:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(out).astype(np.float64)


class OutlierExposureClassD:
    """Energy scorer over an OE-fine-tuned (backbone, head).

    Interface (matches the appendix runner's per-sample fine-tune path):
      __init__(backbone, config)  -- backbone is the fine-tuned deep copy;
                                     config["classifier"] is the fine-tuned head.
      fit(x_id=None, y_id=None)   -- no-op (fine-tuning already done by the runner).
      score(x) -> np.ndarray      -- per-window energy, higher = more OOD.
    """

    EVAL_MODE = "per_sample_finetune"
    SCORE_TYPE = "energy"
    ARMS = ("head_only", "full_net")

    def __init__(self, backbone: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.bb = backbone
        self.config = config or {}
        self.head: Optional[nn.Module] = self.config.get("classifier")
        self.temperature = float(self.config.get("temperature", 1.0))
        self.device = self.config.get("device", "cpu")
        if self.head is None:
            raise ValueError(
                "OutlierExposureClassD needs the fine-tuned head in config['classifier']."
            )

    def fit(self, x_id: Any = None, y_id: Any = None) -> "OutlierExposureClassD":
        # The (backbone, head) handed in are ALREADY OE-fine-tuned by the runner.
        return self

    def score(self, x: Any) -> np.ndarray:
        return energy_scores(
            self.bb.model, self.head, x,
            temperature=self.temperature, device=self.device,
        )
