from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("msp")
class MSPDetector(BaseOODDetector):
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.temperature = float(self.config.get("temperature", 1.0))

    def score(self, x: Any) -> np.ndarray:
        logits, _ = self._logits_and_input(x, require_grad=False)
        scores = 1.0 - self._softmax_max(logits, temperature=self.temperature)
        return scores.detach().cpu().numpy()
