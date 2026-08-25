"""
M2N2: When Model Meets New Normals (Adapted).

Based on:
    Paper: https://doi.org/10.1609/aaai.v38i12.29210
           "When Model Meets New Normals: Test-time Adaptation for
            Unsupervised Time-series Anomaly Detection" (AAAI 2024)
    Code:  https://github.com/carrtesy/M2N2
    ArXiv: https://arxiv.org/abs/2312.11976

Adaptation for frozen backbone architecture:
    - Original M2N2 trains autoencoders on raw time series with test-time adaptation
    - M2N2-Lite trains a lightweight autoencoder on frozen backbone features
    - Implements EMA-based trend estimation for detrending
    - Supports optional test-time adaptation (update on predicted normals)

Trains a lightweight autoencoder to reconstruct normal frozen backbone features.
Test samples are detrended using exponential moving average (EMA) before
reconstruction. The reconstruction error serves as the OOD score. Supports
optional test-time adaptation to update the model on predicted-normal samples
during inference.

Parameters:
    n_epochs: Training epochs (default: 30)
    hidden_dim: Autoencoder hidden dimension (default: 64)
    gamma: EMA update rate for trend (default: 0.9)
    eta: Test-time learning rate (default: 0.01)
    adapt_test_time: Enable online adaptation (default: False for simplicity)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.registry import register_ood
from core.base_ood import BaseOODDetector


# Lightweight Autoencoder for Feature Reconstruction

class FeatureAutoencoder(nn.Module):
    """Simple MLP-based autoencoder for feature reconstruction."""

    def __init__(self, feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input features.

        Args:
            x: (B, feat_dim) input features

        Returns:
            x_recon: (B, feat_dim) reconstructed features
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# ---------------------------------------------------------------------------
# M2N2-Lite Detector
# ---------------------------------------------------------------------------

@register_ood("m2n2")
class M2N2Detector(BaseOODDetector):
    """M2N2-Lite: Test-time adaptive OOD detection with trend estimation.

    Config keys (all optional):
        n_epochs         (int, 30)       – training epochs for autoencoder
        hidden_dim       (int, 64)       – autoencoder hidden dimension
        gamma            (float, 0.995)  – EMA update rate for trend (0-1, higher=more stable)
        eta              (float, 0.01)   – test-time learning rate
        adapt_test_time  (bool, False)   – enable online adaptation during scoring
        batch_size       (int, 64)       – training batch size
        lr               (float, 1e-3)   – training learning rate
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.n_epochs: int = int(self.config.get("n_epochs", 30))
        self.hidden_dim: int = int(self.config.get("hidden_dim", 64))
        self.gamma: float = float(self.config.get("gamma", 0.995))
        self.eta: float = float(self.config.get("eta", 0.01))
        self.adapt_test_time: bool = bool(self.config.get("adapt_test_time", False))
        self.batch_size: int = int(self.config.get("batch_size", 64))
        self.lr: float = float(self.config.get("lr", 1e-3))

        self.autoencoder: Optional[FeatureAutoencoder] = None
        self.trend_mean: Optional[torch.Tensor] = None  # EMA trend estimate

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """Train autoencoder on ID features.

        Args:
            x_id: ID training data, shape (N, C, T).
            y_id: Ignored (unsupervised reconstruction).
        """
        x_tensor = self._to_tensor(x_id)  # (N, C, T)

        # Extract features from frozen backbone
        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        N, feat_dim = feats.shape

        # Initialize autoencoder
        self.autoencoder = FeatureAutoencoder(
            feat_dim=feat_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        # Initialize trend estimate (mean of training features)
        self.trend_mean = feats.mean(dim=0, keepdim=True).detach()  # (1, feat_dim)
        # Snapshot the fit-time trend so every score() call starts from the same
        # state (the runner scores val then test; without this reset the EMA from
        # the val pass leaks into the test pass, making test scores order-dependent
        # on an unrelated split). Fix per methods/_validation/FIX_PLAN.md (2026-08-20).
        self._trend_init = self.trend_mean.clone()

        # Training
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        dataset = TensorDataset(feats)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.autoencoder.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for (batch_feats,) in loader:
                batch_feats = batch_feats.to(self.device)
                B = len(batch_feats)

                # Detrend: subtract trend mean
                detrended = batch_feats - self.trend_mean

                # Reconstruct
                recon = self.autoencoder(detrended)

                # Reconstruction loss
                loss = F.mse_loss(recon, detrended)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B

            epoch_loss /= N

        self.autoencoder.eval()

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, x: Any) -> np.ndarray:
        """Return OOD scores for x.  Higher = more OOD.

        Method: Reconstruction error on detrended features.
        - Detrend using EMA trend estimate
        - Reconstruct via autoencoder
        - Compute MSE as OOD score
        - Optionally: online adapt on predicted normals (test-time adaptation)

        Args:
            x: Input data (N, C, T).

        Returns:
            ood_scores: (N,) float array; higher value → more OOD.
        """
        if self.autoencoder is None or self.trend_mean is None:
            raise RuntimeError("M2N2 must be fit before scoring.")

        # Reset the EMA trend to its fit-time value so each score() call (val, test)
        # is independent and reproducible. See FIX_PLAN.md (2026-08-20).
        self.trend_mean = self._trend_init.clone()

        x_tensor = self._to_tensor(x)

        # Extract features
        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        N = len(feats)
        ood_scores = []

        # Process sequentially for EMA trend update (simulates streaming)
        for i in range(N):
            feat_i = feats[i:i+1]  # (1, feat_dim)

            # Update EMA trend estimate
            empirical_mean = feat_i
            self.trend_mean = self.gamma * self.trend_mean + (1 - self.gamma) * empirical_mean

            # Detrend
            detrended = feat_i - self.trend_mean

            # Reconstruct
            with torch.no_grad():
                recon = self.autoencoder(detrended)

            # Reconstruction error
            error = F.mse_loss(recon, detrended, reduction="none").mean().item()
            ood_scores.append(error)

            # Optional: Test-time adaptation (update model on predicted normals)
            if self.adapt_test_time:
                # Simple heuristic: if error is below median, consider it normal
                # (In practice, would use a learned threshold)
                threshold = np.median(ood_scores) if len(ood_scores) > 10 else float('inf')

                if error < threshold:
                    # Predicted as normal → update autoencoder
                    self.autoencoder.train()
                    optimizer = torch.optim.SGD(self.autoencoder.parameters(), lr=self.eta)

                    recon_adapt = self.autoencoder(detrended)
                    loss = F.mse_loss(recon_adapt, detrended)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    self.autoencoder.eval()

        # Higher reconstruction error = more OOD
        return np.array(ood_scores, dtype=np.float64)
