"""
InvAD: Detecting Both Seen and Unseen Anomalies in Time Series (Adapted).

Based on:
    Paper: https://dl.acm.org/doi/10.1145/3717071
           "Detecting Both Seen and Unseen Anomalies in Time Series"
           (ACM TKDD, 2025)
    Code:  https://github.com/fly-orange/InvAD

Adaptation for frozen backbone architecture:
    - Original InvAD uses a full Invertible Neural Network (INN) on raw time series
    - InvAD-Lite decomposes frozen backbone features into ID and OOD components
    - Uses a lightweight invertible coupling layer for feature decomposition
    - Detects ID anomalies via classifier, OOD anomalies via reconstruction error

Trains an invertible network to decompose frozen backbone features into ID
features (z_id) and OOD (residual) features (z_ood). A classifier trained on
z_id recognizes ID patterns, while a *deliberately lossy* reconstruction
detects OOD samples.

Faithful reconstruction mechanism (mirrors fly-orange/InvAD
``model.py:51`` and ``get_rec_scores`` ``:82-88``): at reconstruction time the
residual branch ``z_ood`` is REPLACED BY A CONSTANT before the inverse pass, so
``reconstruct([z_id, const])`` must rebuild the features from ``z_id`` alone.
This makes the reconstruction lossy and its error informative — high for
samples whose residual information cannot be recovered from ``z_id`` (OOD).
Feeding the exact forward output ``[z_id, z_ood]`` back (the previous behaviour)
made ``reconstruct`` the exact inverse of the forward pass, so the error was
float round-off (~1e-14) and carried no signal.

Two design elements make the invertible network actually decompose:
    - A fixed half-permutation is applied BETWEEN coupling layers so the first
      half (z_id) is mixed with the second half. Without it every coupling
      layer leaves the first half untouched (z1 = x1), leaving z_id bit-
      identical to the raw features (no learned decomposition).
    - The reconstruction score adds the official residual-deviation term
      ``MSE(z_ood, const)`` alongside ``MSE(x_recon, x)``.

The final anomaly score combines the reconstruction error (OOD signal) with the
classifier's low-confidence signal on z_id (ID-anomaly signal); higher = OOD.

Parameters:
    n_epochs: Training epochs (default: 30)
    hidden_dim: Hidden dimension of invertible layers (default: 128)
    lr: Learning rate (default: 1e-3)
    lambda_recon: Reconstruction loss weight (default: 0.5)
    res_const: Constant substituted for z_ood before the inverse pass
               (default: 0.0; cf. official ``res_const``)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.registry import register_ood
from core.base_ood import BaseOODDetector


# Invertible Coupling Layer (Affine Coupling)

class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for invertible transformation.

    Split input x into x1, x2.
    Forward:  z1 = x1
              z2 = x2 * exp(s(x1)) + t(x1)
    Inverse:  x1 = z1
              x2 = (z2 - t(z1)) * exp(-s(z1))
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.split_dim = feat_dim // 2

        # Scale and translation networks
        self.s_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim - self.split_dim),
            nn.Tanh(),  # Bounded scale for stability
        )

        self.t_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim - self.split_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> z."""
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        s = self.s_net(x1)
        t = self.t_net(x1)
        z1 = x1
        z2 = x2 * torch.exp(s) + t
        return torch.cat([z1, z2], dim=1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass: z -> x."""
        z1, z2 = z[:, :self.split_dim], z[:, self.split_dim:]
        s = self.s_net(z1)
        t = self.t_net(z1)
        x1 = z1
        x2 = (z2 - t) * torch.exp(-s)
        return torch.cat([x1, x2], dim=1)


# ---------------------------------------------------------------------------
# InvAD Decomposition Network
# ---------------------------------------------------------------------------

class InvADNetwork(nn.Module):
    """Invertible network for feature decomposition + classifier."""

    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.split_dim = feat_dim // 2

        # Stack of invertible coupling layers
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(feat_dim, hidden_dim) for _ in range(n_layers)
        ])

        # Fixed permutations applied BETWEEN consecutive coupling layers so that
        # information crosses the split boundary. Each affine coupling leaves its
        # first half untouched (z1 = x1); without a permutation the first half
        # would pass through every layer unchanged, so z_id would be bit-
        # identical to feats[:, :D//2] (no learned decomposition). A fixed
        # (seeded) permutation is a valid invertible operation whose inverse is
        # the argsort index. Permutations are registered as buffers so they move
        # with the module (.to(device)) and are not trained.
        self.n_perms = max(0, n_layers - 1)
        gen = torch.Generator().manual_seed(0)
        for i in range(self.n_perms):
            perm = torch.randperm(feat_dim, generator=gen)
            self.register_buffer(f"perm_{i}", perm)
            self.register_buffer(f"inv_perm_{i}", torch.argsort(perm))

        # Classifier on ID features (first half of decomposed features)
        id_dim = feat_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(id_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose features into ID and OOD components.

        Args:
            x: (B, feat_dim) original features

        Returns:
            z_id: (B, feat_dim//2) ID features
            z_ood: (B, feat_dim - feat_dim//2) OOD features
        """
        z = x
        for i, layer in enumerate(self.coupling_layers):
            z = layer(z)
            if i < self.n_perms:
                z = z[:, getattr(self, f"perm_{i}")]

        z_id = z[:, :self.split_dim]
        z_ood = z[:, self.split_dim:]
        return z_id, z_ood

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct original features from a decomposed representation.

        Inverts the forward pass (coupling layers + inter-layer permutations) in
        reverse order. NOTE: when called with the residual branch replaced by a
        constant (``[z_id, const]``), this is intentionally a *lossy* inverse —
        its error is the OOD reconstruction signal.

        Args:
            z: (B, feat_dim) representation [z_id, z_ood-or-const]

        Returns:
            x_recon: (B, feat_dim) reconstructed features
        """
        x = z
        for i in reversed(range(len(self.coupling_layers))):
            if i < self.n_perms:
                x = x[:, getattr(self, f"inv_perm_{i}")]
            x = self.coupling_layers[i].inverse(x)
        return x

    def classify(self, z_id: torch.Tensor) -> torch.Tensor:
        """Classify based on ID features.

        Args:
            z_id: (B, id_dim) ID features

        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(z_id)


# ---------------------------------------------------------------------------
# InvAD-Lite Detector
# ---------------------------------------------------------------------------

@register_ood("invad")
class InvADDetector(BaseOODDetector):
    """InvAD-Lite: Invertible decomposition for OOD detection.

    Config keys (all optional):
        n_epochs       (int, 30)     – training epochs
        hidden_dim     (int, 128)    – hidden dimension of invertible layers
        n_layers       (int, 2)      – number of coupling layers
        lr             (float, 1e-3) – learning rate
        lambda_recon   (float, 0.5)  – reconstruction loss weight
        lambda_cls     (float, 1.0)  – classification loss weight
        batch_size     (int, 64)     – training batch size
        res_const      (float, 0.0)  – constant substituted for z_ood before the
                                       inverse pass (official ``res_const``)
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.n_epochs: int = int(self.config.get("n_epochs", 30))
        self.hidden_dim: int = int(self.config.get("hidden_dim", 128))
        self.n_layers: int = int(self.config.get("n_layers", 2))
        self.lr: float = float(self.config.get("lr", 1e-3))
        self.lambda_recon: float = float(self.config.get("lambda_recon", 0.5))
        self.lambda_cls: float = float(self.config.get("lambda_cls", 1.0))
        self.batch_size: int = int(self.config.get("batch_size", 64))
        self.res_const: float = float(self.config.get("res_const", 0.0))

        self.invad_net: Optional[InvADNetwork] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """Train InvAD network: invertible decomposition + classifier.

        Args:
            x_id: ID training data, shape (N, C, T).
            y_id: Class labels (N,), optional for supervised training.
        """
        x_tensor = self._to_tensor(x_id)  # (N, C, T)

        # Extract features from frozen backbone
        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        N, feat_dim = feats.shape

        # Get labels (use for supervised classification if available)
        if y_id is not None:
            y_tensor = torch.from_numpy(np.array(y_id)).long().to(self.device)
            num_classes = int(y_tensor.max().item()) + 1
        else:
            # Unsupervised: dummy labels (single class)
            y_tensor = torch.zeros(N, dtype=torch.long, device=self.device)
            num_classes = 1

        # Initialize InvAD network
        self.invad_net = InvADNetwork(
            feat_dim=feat_dim,
            num_classes=num_classes,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        ).to(self.device)

        # Training
        optimizer = torch.optim.Adam(self.invad_net.parameters(), lr=self.lr)
        dataset = TensorDataset(feats, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.invad_net.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for batch_feats, batch_labels in loader:
                batch_feats = batch_feats.to(self.device)
                batch_labels = batch_labels.to(self.device)
                B = len(batch_feats)

                # Decompose features
                z_id, z_ood = self.invad_net.decompose(batch_feats)

                # Reconstruction loss (LOSSY inverse): replace the residual
                # branch z_ood with a constant before inverting, mirroring the
                # official model.py:51. This forces the network to encode the
                # reconstructable information into z_id and to push z_ood -> const
                # for ID data, so reconstruction error becomes an informative
                # OOD signal (rather than the ~0 exact-inverse error).
                z_const = torch.cat(
                    [z_id, torch.full_like(z_ood, self.res_const)], dim=1
                )
                x_recon = self.invad_net.reconstruct(z_const)
                loss_recon = F.mse_loss(x_recon, batch_feats) + F.mse_loss(
                    z_ood, torch.full_like(z_ood, self.res_const)
                )

                # Classification loss (on ID features)
                logits = self.invad_net.classify(z_id)
                if num_classes > 1:
                    loss_cls = F.cross_entropy(logits, batch_labels)
                else:
                    # Unsupervised: no classification loss
                    loss_cls = 0.0

                # Total loss
                loss = self.lambda_recon * loss_recon + self.lambda_cls * loss_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B

            epoch_loss /= N

        self.invad_net.eval()

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, x: Any) -> np.ndarray:
        """Return OOD scores for x.  Higher = more OOD.

        Method: Combine two signals:
        1. ID anomaly: low classifier confidence (seen but anomalous)
        2. OOD anomaly: high reconstruction error (unseen)

        Args:
            x: Input data (N, C, T).

        Returns:
            ood_scores: (N,) float array; higher value → more OOD.
        """
        if self.invad_net is None:
            raise RuntimeError("InvAD must be fit before scoring.")

        x_tensor = self._to_tensor(x)

        # Extract features
        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        N = len(feats)

        # Decompose and score
        self.invad_net.eval()
        with torch.no_grad():
            # Decompose
            z_id, z_ood = self.invad_net.decompose(feats)

            # Reconstruction error (OOD signal): replace the residual branch
            # z_ood with a constant before inverting so the reconstruction is
            # deliberately lossy (official model.py:51). The score is the
            # feature-reconstruction error PLUS the residual-deviation term
            # MSE(z_ood, const) (official get_rec_scores, model.py:82-88).
            z_const = torch.cat(
                [z_id, torch.full_like(z_ood, self.res_const)], dim=1
            )
            x_recon = self.invad_net.reconstruct(z_const)
            recon_error = F.mse_loss(x_recon, feats, reduction="none").mean(dim=1)  # (N,)
            res_dev = F.mse_loss(
                z_ood, torch.full_like(z_ood, self.res_const), reduction="none"
            ).mean(dim=1)  # (N,)
            recon_error = recon_error + res_dev

            # Classification confidence (ID signal)
            logits = self.invad_net.classify(z_id)
            probs = F.softmax(logits, dim=1)
            max_prob = probs.max(dim=1).values  # (N,)

            # ID anomaly score: 1 - max_prob (low confidence = anomalous)
            id_anomaly_score = 1.0 - max_prob

            # Final OOD score: combine reconstruction error + low confidence
            # Higher reconstruction error OR low confidence → more OOD
            ood_scores = 0.6 * recon_error + 0.4 * id_anomaly_score

        return ood_scores.cpu().numpy()
