from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from core.registry import register_backbone


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        layers: Optional[List[int]] = None,
        embedding_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = [2, 2, 2]

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        channels = base_channels
        self.layer1 = self._make_layer(channels, channels, layers[0], stride=1)
        self.layer2 = self._make_layer(channels, channels * 2, layers[1], stride=2)
        channels *= 2
        self.layer3 = self._make_layer(channels, channels * 2, layers[2], stride=2)
        channels *= 2

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(channels, embedding_dim)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.proj(x)


@register_backbone("resnet")
@register_backbone("resnet1d")
class ResNetBackbone:
    def __init__(
        self,
        input_dim: int,
        base_channels: int = 64,
        layers: Optional[List[int]] = None,
        embedding_dim: int = 128,
        dropout: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet1D(
            in_channels=input_dim,
            base_channels=base_channels,
            layers=layers,
            embedding_dim=embedding_dim,
            dropout=dropout,
        ).to(self.device)

    def fit(self, dataset: Dict[str, Any]) -> None:
        # Placeholder for future training loops.
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
