"""Energy-based OOD detector (EBO baseline).

Clean energy baseline used in the benchmark under the label "energy". Equivalent to
the OutlierExposure energy path (base_ood._energy): the score is the negative
log-sum-exp of the class logits, -logsumexp(logits / T). Higher = more OOD.

Reference: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("energy")
class EnergyDetector(BaseOODDetector):
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.temperature = float(self.config.get("temperature", 1.0))

    def score(self, x: Any) -> np.ndarray:
        logits, _ = self._logits_and_input(x, require_grad=False)
        scores = self._energy(logits, temperature=self.temperature)
        return scores.detach().cpu().numpy()
