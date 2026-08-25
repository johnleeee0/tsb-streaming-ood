"""Faithful DiffAD detector (flattened, canonical).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

DiffAD-Lite: diffusion-based anomaly detection on frozen backbone features. Based on
Xiao et al., "Imputation-based Time-Series Anomaly Detection with Conditional
Weight-Incremental Diffusion Models" (KDD 2023); code https://github.com/ChunjingXiao/DiffAD.

This file is SELF-CONTAINED: the base DDPM machinery (formerly
models/ood_methods/diffad.py) is inlined as ``_DiffADBase`` and the corrected variant
(formerly methods/diffad/diffad_fix) is the canonical, registered ``DiffADDetector``.
Behaviour is identical to the diffad_fix variant used in the benchmark method_set.

The corrected variant implements the published imputation-based reverse process: the
input is partially noised to an intermediate diffusion step t0 and then denoised back,
so the reconstruction is conditioned on the input; the (non-negated) reconstruction
error is the OOD score. The denoising network and noise schedule are trained exactly as
in the base fit().
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


# ---------------------------------------------------------------------------
# Simplified Denoising Network (U-Net style for features)
# ---------------------------------------------------------------------------

class DenoisingNetwork(nn.Module):
    """Lightweight MLP-based denoising network for feature vectors."""

    def __init__(self, feat_dim: int, hidden_dim: int = 128, n_steps: int = 50):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_steps = n_steps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Denoising network
        self.net = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_norm = t.float() / self.n_steps  # Normalize to [0, 1]
        t_emb = self.time_embed(t_norm)  # (B, hidden_dim)
        x_t = torch.cat([x, t_emb], dim=-1)  # (B, feat_dim + hidden_dim)
        noise_pred = self.net(x_t)  # (B, feat_dim)
        return noise_pred


# ---------------------------------------------------------------------------
# Diffusion utilities (linear beta schedule, forward/reverse process)
# ---------------------------------------------------------------------------

def linear_beta_schedule(n_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Linear noise schedule."""
    return torch.linspace(beta_start, beta_end, n_steps)


def get_diffusion_params(betas: torch.Tensor):
    """Compute alpha, alpha_bar, etc. for DDPM."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }


# ---------------------------------------------------------------------------
# Base DiffAD machinery (inlined from models/ood_methods/diffad.py)
# ---------------------------------------------------------------------------

class _DiffADBase(BaseOODDetector):
    """DiffAD-Lite base: diffusion-based OOD detection on frozen backbone features."""

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.n_steps: int = int(self.config.get("n_steps", 50))
        self.n_epochs: int = int(self.config.get("n_epochs", 20))
        self.batch_size: int = int(self.config.get("batch_size", 64))
        self.hidden_dim: int = int(self.config.get("hidden_dim", 128))
        self.lr: float = float(self.config.get("lr", 1e-3))
        self.recon_samples: int = int(self.config.get("recon_samples", 5))

        self.denoiser: Optional[DenoisingNetwork] = None
        self.diffusion_params: Optional[Dict[str, torch.Tensor]] = None

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """Train the denoising diffusion model on ID training features."""
        x_tensor = self._to_tensor(x_id)  # (N, C, T)

        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        feat_dim = feats.shape[1]
        N = len(feats)

        self.denoiser = DenoisingNetwork(
            feat_dim=feat_dim,
            hidden_dim=self.hidden_dim,
            n_steps=self.n_steps,
        ).to(self.device)

        betas = linear_beta_schedule(self.n_steps).to(self.device)
        self.diffusion_params = get_diffusion_params(betas)

        optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=self.lr)
        dataset = TensorDataset(feats)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.denoiser.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for (batch_feats,) in loader:
                batch_feats = batch_feats.to(self.device)
                B = len(batch_feats)

                t = torch.randint(0, self.n_steps, (B,), device=self.device)

                noise = torch.randn_like(batch_feats)
                sqrt_alpha_bar = self.diffusion_params["sqrt_alphas_cumprod"][t].view(B, 1)
                sqrt_one_minus_alpha_bar = self.diffusion_params["sqrt_one_minus_alphas_cumprod"][t].view(B, 1)
                noisy_feats = sqrt_alpha_bar * batch_feats + sqrt_one_minus_alpha_bar * noise

                noise_pred = self.denoiser(noisy_feats, t)

                loss = F.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B

            epoch_loss /= N

        self.denoiser.eval()

    def score(self, x: Any) -> np.ndarray:
        """Base (from-noise) reverse process. Overridden by the canonical variant."""
        if self.denoiser is None or self.diffusion_params is None:
            raise RuntimeError("DiffAD must be fit before scoring.")

        x_tensor = self._to_tensor(x)

        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, feat_dim)

        N, feat_dim = feats.shape

        recon_errors = []
        self.denoiser.eval()
        with torch.no_grad():
            for i in range(N):
                feat_i = feats[i:i+1]  # (1, feat_dim)
                errors_i = []
                for _ in range(self.recon_samples):
                    x_t = torch.randn_like(feat_i)  # Start from noise
                    for t_idx in reversed(range(self.n_steps)):
                        t = torch.tensor([t_idx], device=self.device)
                        noise_pred = self.denoiser(x_t, t)
                        alpha = self.diffusion_params["alphas"][t_idx]
                        alpha_bar = self.diffusion_params["alphas_cumprod"][t_idx]
                        beta = self.diffusion_params["betas"][t_idx]
                        if t_idx > 0:
                            noise_sample = torch.randn_like(x_t)
                        else:
                            noise_sample = 0.0
                        x_t = (1.0 / torch.sqrt(alpha)) * (
                            x_t - (beta / torch.sqrt(1.0 - alpha_bar)) * noise_pred
                        ) + torch.sqrt(beta) * noise_sample
                    recon = x_t
                    error = F.mse_loss(recon, feat_i, reduction="none").mean().item()
                    errors_i.append(error)
                recon_errors.append(np.mean(errors_i))

        return -np.array(recon_errors, dtype=np.float64)


# ---------------------------------------------------------------------------
# Corrected (canonical) variant — input-conditioned imputation reverse process
# ---------------------------------------------------------------------------

@register_ood("diffad")
class DiffADDetector(_DiffADBase):
    def __init__(self, model, config=None) -> None:
        super().__init__(model, config)
        # Partial-noising level: noise the input to t0 then denoise back (imputation-style).
        self.t0 = int(self.config.get("t0", max(1, self.n_steps // 2)))

    def score(self, x: Any) -> np.ndarray:
        if self.denoiser is None or self.diffusion_params is None:
            raise RuntimeError("DiffAD must be fit before scoring.")
        dp = self.diffusion_params
        x_tensor = self._to_tensor(x)
        with torch.no_grad():
            feats = self._forward_features(x_tensor)  # (N, D)
        N = feats.shape[0]
        errs = np.empty(N, dtype=np.float64)
        self.denoiser.eval()
        with torch.no_grad():
            for i in range(N):
                x0 = feats[i:i + 1]
                sample_errs = []
                for _ in range(self.recon_samples):
                    t0 = min(self.t0, self.n_steps - 1)
                    # forward: noise the INPUT to step t0
                    noise = torch.randn_like(x0)
                    xt = (dp["sqrt_alphas_cumprod"][t0] * x0
                          + dp["sqrt_one_minus_alphas_cumprod"][t0] * noise)
                    # reverse: denoise from t0 back to 0 (conditioned on the noised input)
                    for t_idx in reversed(range(t0 + 1)):
                        t = torch.tensor([t_idx], device=xt.device)
                        eps = self.denoiser(xt, t)
                        alpha = dp["alphas"][t_idx]
                        alpha_bar = dp["alphas_cumprod"][t_idx]
                        beta = dp["betas"][t_idx]
                        z = torch.randn_like(xt) if t_idx > 0 else 0.0
                        xt = (1.0 / torch.sqrt(alpha)) * (
                            xt - (beta / torch.sqrt(1.0 - alpha_bar)) * eps
                        ) + torch.sqrt(beta) * z
                    sample_errs.append(F.mse_loss(xt, x0).item())
                errs[i] = float(np.mean(sample_errs))
        # higher reconstruction error = more OOD (no negation)
        return errs
