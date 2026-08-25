"""Corrected SCALE detector (scale_enh).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

We identified and corrected an inconsistency between the original implementation and the procedure
described in Xu et al. (2024), confirmed against the official repository (kai422/SCALE,
openood/networks/scale_net.py). SCALE rescales the PENULTIMATE-LAYER activations by a per-sample
factor exp(s1/s2), where s1 is the sum of all activations and s2 the sum after pruning all but the
top-k activations (k = n - round(n * percentile / 100)); the scaled features are passed through the
classification layer and scored with ENERGY. The original implementation instead z-standardised the
LOGITS and computed energy, which neither scales activations nor operates at the penultimate layer.
scale_enh implements the published activation-scaling procedure. Because SCALE assumes non-negative
(post-ReLU) activations and the project backbone exposes a linear embedding, we apply a ReLU to the
features before scaling and document this adaptation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("scale")
class SCALEEnhDetector(BaseOODDetector):
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.percentile = float(self.config.get("percentile", 85.0))
        self.temperature = float(self.config.get("temperature", 1.0))

    def score(self, x: Any) -> np.ndarray:
        x_tensor = self._to_tensor(x)
        with torch.no_grad():
            feats = self._forward_features(x_tensor)
            feats = torch.relu(feats)  # SCALE assumes non-negative activations
            clf = self._classifier()
            b, d = feats.shape
            s1 = feats.sum(dim=1)
            k = max(1, d - int(np.round(d * self.percentile / 100.0)))
            v, i = torch.topk(feats, k, dim=1)
            pruned = torch.zeros_like(feats).scatter_(1, i, v)
            s2 = pruned.sum(dim=1) + 1e-6
            scale = (s1 / s2).clamp(max=50.0)  # guard against overflow
            feats_scaled = feats * torch.exp(scale)[:, None]
            if clf is None:
                return (-torch.logsumexp(feats_scaled / self.temperature, dim=-1)).detach().cpu().numpy()
            logits = clf(feats_scaled)
            scores = self._energy(logits, temperature=self.temperature)  # higher = OOD
        return scores.detach().cpu().numpy()
