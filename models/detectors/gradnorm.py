"""Corrected GradNorm detector (gradnorm_enh).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

We identified and corrected an inconsistency between the original implementation and the
procedure described in Huang, Geng & Li (2021), "On the Importance of Gradients for Detecting
Distributional Shifts in the Wild", confirmed against the official repository
(deeplearning-wisc/gradnorm_ood, test_ood.py:124-145). The original implementation backpropagated
a cross-entropy loss to the network INPUT and reported the L2 norm of the input gradient as an OOD
score (higher = OOD). The published method instead backpropagates the KL divergence between the
temperature-scaled softmax and the uniform distribution, and reports the L1 norm of the gradient
with respect to the LAST classification-layer weights, where in-distribution samples yield a LARGER
gradient norm than out-of-distribution samples. We therefore (i) use the KL-to-uniform objective,
(ii) take the gradient with respect to the classifier weights, (iii) use the L1 norm, and
(iv) negate the norm so that, consistent with the project convention, a higher score denotes a
more out-of-distribution input.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("gradnorm")
class GradNormEnhDetector(BaseOODDetector):
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.temperature = float(self.config.get("temperature", 1.0))

    def score(self, x: Any) -> np.ndarray:
        x_tensor = self._to_tensor(x)
        clf = self._classifier()
        if clf is None or not hasattr(clf, "weight"):
            # No head available: fall back to feature-gradient L1 (documented degenerate case).
            with torch.no_grad():
                feats = self._forward_features(x_tensor)
            return (-feats.abs().sum(dim=-1)).detach().cpu().numpy()

        num_classes = clf.weight.shape[0]
        scores = np.empty(x_tensor.shape[0], dtype=np.float64)
        logsoftmax = torch.nn.LogSoftmax(dim=-1)
        for i in range(x_tensor.shape[0]):
            xi = x_tensor[i : i + 1]
            with torch.no_grad():
                feats = self._forward_features(xi)
            feats = feats.detach().clone().requires_grad_(False)
            if clf.weight.grad is not None:
                clf.weight.grad = None
            logits = clf(feats) / self.temperature
            targets = torch.ones((1, num_classes), device=logits.device)
            loss = torch.mean(torch.sum(-targets * logsoftmax(logits), dim=-1))
            loss.backward()
            l1 = clf.weight.grad.detach().abs().sum().item()
            scores[i] = -l1  # ID -> large L1 -> low (more negative) -> not OOD; OOD -> small L1 -> higher score
        if clf.weight.grad is not None:
            clf.weight.grad = None
        return scores
