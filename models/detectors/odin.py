from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("odin")
class ODINDetector(BaseOODDetector):
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.temperature = float(self.config.get("temperature", 1000.0))
        self.epsilon = float(self.config.get("epsilon", 0.001))

    def score(self, x: Any) -> np.ndarray:
        logits, x_tensor = self._logits_and_input(x, require_grad=True)
        pred = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits / self.temperature, pred)
        loss.backward()
        grad_sign = x_tensor.grad.detach().sign()
        perturbed = x_tensor - self.epsilon * grad_sign

        with torch.no_grad():
            logits_pert = self._forward_logits(perturbed)
            scores = 1.0 - self._softmax_max(logits_pert, temperature=self.temperature)
        return scores.detach().cpu().numpy()
