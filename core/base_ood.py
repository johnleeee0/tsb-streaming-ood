from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


class BaseOODDetector(ABC):
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.model = model
        self.config = config or {}
        self.device = self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self._set_eval_mode()

    def _set_eval_mode(self) -> None:
        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()
        elif hasattr(self.model, "model") and isinstance(self.model.model, nn.Module):
            self.model.model.to(self.device)
            self.model.model.eval()

    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, dict) and "x" in x:
            x = x["x"]
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            x_tensor = x.float()
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        return x_tensor.to(self.device)

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, nn.Module):
            out = self.model(x)
        elif hasattr(self.model, "model"):
            out = self.model.model(x)
        else:
            out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        # Chain through classification head if available so callers
        # get actual class logits (num_classes-dim) not raw embeddings.
        clf = self._classifier()
        if clf is not None:
            out = clf(out)
        return out

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw backbone embeddings – intentionally skips the
        classification head so callers receive feature vectors, not logits."""
        if hasattr(self.model, "embed"):
            feats = self.model.embed(x.detach().cpu().numpy())
            return torch.from_numpy(feats).to(self.device)
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(x)
        # Direct backbone call without chaining the classifier head.
        if isinstance(self.model, nn.Module):
            out = self.model(x)
        elif hasattr(self.model, "model"):
            out = self.model.model(x)
        else:
            out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        return out

    def _classifier(self) -> Optional[nn.Module]:
        clf = self.config.get("classifier")
        if isinstance(clf, nn.Module):
            clf.to(self.device)
            clf.eval()
        return clf

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        return None

    @abstractmethod
    def score(self, x: Any) -> np.ndarray:
        raise NotImplementedError

    def _softmax_max(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs.max(dim=-1).values

    def _energy(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return -torch.logsumexp(logits / temperature, dim=-1)

    def _logits_and_input(self, x: Any, require_grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, dict) and "x" in x:
            x = x["x"]
        x_tensor = self._to_tensor(x)
        x_tensor.requires_grad_(require_grad)
        logits = self._forward_logits(x_tensor)
        return logits, x_tensor
