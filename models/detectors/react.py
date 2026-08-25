"""Corrected ReAct detector (react_enh).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

We identified and corrected an inconsistency between the original implementation and the procedure
described in Sun, Guo & Li (2021), confirmed against the official repository
(deeplearning-wisc/react). ReAct rectifies (clips) penultimate activations at a percentile
threshold estimated on in-distribution data and then computes the ENERGY score on the resulting
logits. The original implementation clipped correctly but scored with the maximum softmax
probability instead of energy. react_enh restores the energy score after clipping.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("react")
class ReACTEnhDetector(BaseOODDetector):
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.percentile = float(self.config.get("percentile", 90.0))
        self.temperature = float(self.config.get("temperature", 1.0))
        self.threshold: Optional[float] = None

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        x_tensor = self._to_tensor(x_id)
        with torch.no_grad():
            feats = self._forward_features(x_tensor)
        self.threshold = float(np.percentile(feats.detach().cpu().numpy(), self.percentile))

    def score(self, x: Any) -> np.ndarray:
        x_tensor = self._to_tensor(x)
        with torch.no_grad():
            feats = self._forward_features(x_tensor)
            if self.threshold is None:
                self.threshold = float(np.percentile(feats.detach().cpu().numpy(), self.percentile))
            feats = torch.clamp(feats, max=self.threshold)
            clf = self._classifier()
            if clf is None:
                # No head: fall back to clipped-feature energy proxy.
                return (-torch.logsumexp(feats / self.temperature, dim=-1)).detach().cpu().numpy()
            logits = clf(feats)
            scores = self._energy(logits, temperature=self.temperature)  # -logsumexp -> higher = OOD
        return scores.detach().cpu().numpy()
