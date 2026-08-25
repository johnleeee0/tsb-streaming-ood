from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from core.registry import register_backbone


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length) -> (batch, length, channels)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.proj(pooled)


@register_backbone("lstm")
class LSTMBackbone:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        ).to(self.device)

    def fit(self, dataset: Dict[str, Any]) -> None:
        return None

    def embed(self, x: np.ndarray) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float()
        x_tensor = x_tensor.to(self.device)
        self.model.eval()
        with torch.no_grad():
            feats = self.model(x_tensor)
        return feats.cpu().numpy()
