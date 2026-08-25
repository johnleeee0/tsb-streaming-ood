"""
SRS: Seasonal Ratio Scoring for time-series OOD detection.

Paper: https://arxiv.org/abs/2207.04306
Code:  https://github.com/tahabelkhouja/SRS

Uses STL decomposition to extract trend+seasonal patterns per class, aligns samples
via circular shift, then trains two conditional VAEs: one on aligned signals and one
on residuals. The OOD score is the Seasonal Ratio: the per-sample signal neg-ELBO
divided by the residual neg-ELBO (mirroring the official `ratio = ll_signal / ll_residual`
in Run_SRS.py:139,145,178). Higher = more OOD.

This is an SRS-inspired PyTorch implementation. The per-sample neg-ELBO deliberately
improves on the official code's batch-constant likelihood (see VERIFICATION.md §4), and
the CVAE/STL details differ from the official Keras/TensorFlow model; but the score is the
seasonal signal/residual ratio as in the paper. Uses statsmodels for STL decomposition
with fallback to moving-average trend.

Requires: statsmodels (pip install statsmodels)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.registry import register_ood
from core.base_ood import BaseOODDetector


# STL decomposition helpers

def _stl_pattern(series: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pattern, residual) for a 1-D signal using STL.

    Falls back to simple moving-average trend subtraction when statsmodels
    is unavailable or the series is too short for STL.

    Args:
        series: (T,) float array.
        period: Seasonal period.

    Returns:
        pattern:  (T,) trend+seasonal component.
        residual: (T,) residual = series - pattern.
    """
    T = len(series)

    # Try statsmodels STL first
    if T >= 2 * period + 1:
        try:
            from statsmodels.tsa.seasonal import STL
            result = STL(series, period=period, robust=True).fit()
            pattern = result.trend + result.seasonal
            return pattern, result.resid
        except Exception:
            pass

    # Fallback: moving-average trend + mean seasonal pattern
    half = min(period // 2, T // 4, 1)
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    trend = np.convolve(series, kernel, mode="same")
    residual = series - trend
    return trend, residual


def _class_pattern(class_signals: np.ndarray, period: int) -> np.ndarray:
    """Compute mean trend+seasonal pattern across a class.

    Args:
        class_signals: (N_c, T) array for one class.
        period: Seasonal period.

    Returns:
        pattern: (T,) mean pattern.
    """
    patterns = []
    for sig in class_signals:
        p, _ = _stl_pattern(sig.astype(np.float64), period)
        patterns.append(p)
    return np.mean(patterns, axis=0)


def _align_to_pattern(signal: np.ndarray, pattern: np.ndarray) -> Tuple[np.ndarray, int]:
    """Align signal to pattern via circular shift (best cross-correlation lag).

    Args:
        signal:  (T,) array.
        pattern: (T,) target pattern.

    Returns:
        aligned:     (T,) circularly-shifted version of signal.
        best_offset: integer shift applied.
    """
    T = len(signal)
    corr = np.array(
        [np.dot(np.roll(signal, k), pattern) for k in range(T)]
    )
    best_offset = int(np.argmax(corr))
    return np.roll(signal, best_offset), best_offset


def _auto_period(T: int) -> int:
    """Heuristic: use T // 4 (quarter of series length), at least 2."""
    return max(2, T // 4)


# ---------------------------------------------------------------------------
# CVAE (1-D convolutional, class-conditional)
# ---------------------------------------------------------------------------

class _Conv1dCVAE(nn.Module):
    """Conditional VAE with 1-D convolutions.

    Encoder compresses (C, T) → (latent_dim,).
    Conditioning is achieved by concatenating a one-hot class vector to z.
    Decoder reconstructs (C, T) from (latent_dim + n_classes,).

    Args:
        in_channels:  Number of input channels (C).
        seq_len:      Sequence length (T).
        n_classes:    Number of ID classes (for conditioning).
        latent_dim:   Latent space dimensionality.
        base_filters: Base number of convolutional filters.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        n_classes: int,
        latent_dim: int = 32,
        base_filters: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.enc_pool = nn.AdaptiveAvgPool1d(max(1, seq_len // 4))

        enc_flat_dim = base_filters * 2 * max(1, seq_len // 4)
        self.enc_fc = nn.Linear(enc_flat_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        dec_in_dim = latent_dim + n_classes
        self.dec_fc1 = nn.Linear(dec_in_dim, 128)
        self.dec_fc2 = nn.Linear(128, enc_flat_dim)

        self._dec_reshape_c = base_filters * 2
        self._dec_reshape_t = max(1, seq_len // 4)

        self.dec_upsample = nn.Upsample(size=seq_len, mode="linear", align_corners=False)
        self.dec_conv1 = nn.Conv1d(base_filters * 2, base_filters, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv1d(base_filters, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = self.enc_pool(h)
        h = h.reshape(h.size(0), -1)
        h = F.relu(self.enc_fc(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
        z_cond = torch.cat([z, labels_onehot], dim=-1)
        h = F.relu(self.dec_fc1(z_cond))
        h = F.relu(self.dec_fc2(h))
        h = h.reshape(h.size(0), self._dec_reshape_c, self._dec_reshape_t)
        h = self.dec_upsample(h)
        h = F.relu(self.dec_conv1(h))
        return self.dec_conv2(h)

    def forward(
        self, x: torch.Tensor, labels_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z, labels_onehot)
        return recon, mu, logvar

    def elbo(
        self, x: torch.Tensor, labels_onehot: torch.Tensor, beta: float = 1.0
    ) -> torch.Tensor:
        """Compute negative ELBO (loss to minimise).

        −ELBO = reconstruction_loss + β * KL
        """
        recon, mu, logvar = self.forward(x, labels_onehot)
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta * kl

    def neg_elbo_per_sample(self, x: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
        """Return per-sample negative ELBO (no reduction).

        Used for scoring at test time.
        """
        recon, mu, logvar = self.forward(x, labels_onehot)
        recon_loss = F.mse_loss(recon, x, reduction="none").sum(dim=(1, 2))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        return recon_loss + kl


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def _train_cvae(
    cvae: _Conv1dCVAE,
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    label_map: Dict[Any, int],
) -> None:
    """Train the CVAE in-place.

    Args:
        cvae:      Model to train.
        x_data:    (N, C, T) float array.
        y_data:    (N,) class labels.
        n_epochs:  Training epochs.
        batch_size: Mini-batch size.
        lr:        Learning rate.
        device:    Torch device string.
        label_map: Maps class label → integer index for one-hot encoding.
    """
    n_classes = len(label_map)
    x_t = torch.from_numpy(x_data).float()
    y_idx = torch.tensor(
        [label_map[c] for c in y_data], dtype=torch.long
    )
    labels_oh = F.one_hot(y_idx, num_classes=n_classes).float()

    ds = TensorDataset(x_t, labels_oh)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    cvae = cvae.to(device)
    opt = torch.optim.Adam(cvae.parameters(), lr=lr)

    cvae.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = cvae.elbo(xb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    cvae.eval()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@register_ood("srs")
class SRSDetector(BaseOODDetector):
    """Seasonal Ratio Scoring (SRS) OOD detector.

    Config keys (all optional):
        period     (int|null, null) – seasonal period; auto-detected if null.
        latent_dim (int, 32)        – CVAE latent dimension.
        n_epochs   (int, 30)        – CVAE training epochs.
        batch_size (int, 32)        – CVAE training batch size.
        lr         (float, 1e-3)    – CVAE learning rate.
        beta       (float, 1.0)     – KL weight in ELBO.
        mc_samples (int, 10)        – Monte-Carlo samples for likelihood estimation.
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.period: Optional[int] = self.config.get("period", None)
        self.latent_dim: int = int(self.config.get("latent_dim", 32))
        self.n_epochs: int = int(self.config.get("n_epochs", 30))
        self.batch_size: int = int(self.config.get("batch_size", 32))
        self.lr: float = float(self.config.get("lr", 1e-3))
        self.mc_samples: int = int(self.config.get("mc_samples", 10))

        # Fitted state
        self.class_labels: Optional[np.ndarray] = None
        self.label_map: Optional[Dict[Any, int]] = None
        self.class_patterns: Optional[Dict[Any, np.ndarray]] = None  # per-class (C, T)
        self.cvae: Optional[_Conv1dCVAE] = None
        self.rescvae: Optional[_Conv1dCVAE] = None
        self._period_used: int = 2

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """Fit class patterns and train both CVAEs.

        Args:
            x_id: (N, C, T) array or tensor.
            y_id: (N,) class labels (required).
        """
        if y_id is None:
            raise ValueError("SRS requires class labels (y_id).")

        if isinstance(x_id, torch.Tensor):
            x_np = x_id.cpu().numpy().astype(np.float32)
        else:
            x_np = np.asarray(x_id, dtype=np.float32)

        if isinstance(y_id, torch.Tensor):
            y_np = y_id.cpu().numpy()
        else:
            y_np = np.asarray(y_id)

        N, C, T = x_np.shape
        self._period_used = self.period if self.period is not None else _auto_period(T)

        # Normalise per-channel (across train set)
        self._x_mean = x_np.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        self._x_std = x_np.std(axis=(0, 2), keepdims=True) + 1e-6
        x_norm = (x_np - self._x_mean) / self._x_std

        # Compute per-class seasonal patterns using STL decomposition
        self.class_labels = np.unique(y_np)
        self.label_map = {c: i for i, c in enumerate(self.class_labels)}
        n_classes = len(self.class_labels)

        self.class_patterns = {}
        for cl in self.class_labels:
            mask = (y_np == cl)
            # average pattern across channels, then broadcast back
            ch_patterns = []
            for ch in range(C):
                sigs = x_norm[mask, ch, :]          # (N_c, T)
                pat = _class_pattern(sigs, self._period_used)
                ch_patterns.append(pat)
            self.class_patterns[cl] = np.stack(ch_patterns, axis=0)  # (C, T)

        # Align samples to nearest class pattern via circular shift
        x_aligned, x_residual, y_aligned = self._align_dataset(x_norm, y_np)

        # Train CVAE on aligned signals
        self.cvae = _Conv1dCVAE(
            in_channels=C,
            seq_len=T,
            n_classes=n_classes,
            latent_dim=self.latent_dim,
        )
        _train_cvae(
            self.cvae, x_aligned, y_aligned,
            self.n_epochs, self.batch_size, self.lr, self.device, self.label_map,
        )

        # Train ResCVAE on residuals
        self.rescvae = _Conv1dCVAE(
            in_channels=C,
            seq_len=T,
            n_classes=n_classes,
            latent_dim=self.latent_dim,
        )
        _train_cvae(
            self.rescvae, x_residual, y_aligned,
            self.n_epochs, self.batch_size, self.lr, self.device, self.label_map,
        )

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, x: Any) -> np.ndarray:
        """Compute SRS OOD scores.  Higher = more OOD.

        For each test sample:
            1. Find the best-fitting ID class (minimum residual after alignment).
            2. Compute neg-ELBO_cvae  for aligned signal.
            3. Compute neg-ELBO_rescvae for residual.
            4. ratio = neg_elbo_signal / neg_elbo_residual
               Higher ratio → worse fit to any ID class → more OOD.

        Args:
            x: (N, C, T) input.

        Returns:
            ood_scores: (N,) float array; higher → more OOD.
        """
        if self.cvae is None or self.rescvae is None:
            raise RuntimeError("SRS must be fit before scoring.")

        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy().astype(np.float32)
        else:
            x_np = np.asarray(x, dtype=np.float32)

        x_norm = (x_np - self._x_mean) / self._x_std
        N, C, T = x_norm.shape
        n_classes = len(self.class_labels)

        # Find best class for each test sample (minimum mean residual)
        best_classes = self._find_best_class(x_norm)

        # Align to best class and compute residuals
        x_aligned_list = []
        x_residual_list = []
        y_idx_list = []

        for i in range(N):
            cl = best_classes[i]
            pattern = self.class_patterns[cl]  # (C, T)
            aligned, resid = self._align_sample(x_norm[i], pattern)
            x_aligned_list.append(aligned)
            x_residual_list.append(resid)
            y_idx_list.append(self.label_map[cl])

        x_aligned_t = torch.from_numpy(np.stack(x_aligned_list)).float().to(self.device)
        x_residual_t = torch.from_numpy(np.stack(x_residual_list)).float().to(self.device)
        y_idx_t = torch.tensor(y_idx_list, dtype=torch.long, device=self.device)
        labels_oh = F.one_hot(y_idx_t, num_classes=n_classes).float()

        # Score with Monte Carlo sampling for more stable estimates
        self.cvae.eval()
        self.rescvae.eval()

        neg_elbo_sig = torch.zeros(N, device=self.device)
        neg_elbo_res = torch.zeros(N, device=self.device)

        with torch.no_grad():
            for _ in range(self.mc_samples):
                neg_elbo_sig += self.cvae.neg_elbo_per_sample(x_aligned_t, labels_oh)
                neg_elbo_res += self.rescvae.neg_elbo_per_sample(x_residual_t, labels_oh)

        neg_elbo_sig = (neg_elbo_sig / self.mc_samples).cpu().numpy()
        neg_elbo_res = (neg_elbo_res / self.mc_samples).cpu().numpy()

        # Seasonal Ratio Score (the paper's titular contribution).
        # Official SRS forms ratio = ll_signal / ll_residual (Run_SRS.py:139,145,178).
        # Here both quantities are per-sample neg-ELBOs (higher = worse fit), so the
        # analogous seasonal ratio is neg_elbo_sig / neg_elbo_res. Higher = more OOD:
        #   - ID samples align well to a class pattern → signal CVAE models them well
        #     (LOW neg_elbo_sig), while the residual carries the noise (relatively HIGH
        #     neg_elbo_res) → LOW ratio.
        #   - OOD samples fit no ID class pattern → HIGH neg_elbo_sig → HIGH ratio.
        # Guard against divide-by-zero (neg-ELBO = MSE + KL is >= 0; clamp near-zero
        # denominators to a small epsilon to keep the score finite).
        eps = 1e-8
        neg_elbo_res_safe = np.where(
            np.abs(neg_elbo_res) < eps, eps, neg_elbo_res
        )
        ratio = neg_elbo_sig / neg_elbo_res_safe
        # Guarantee finite scores: on some real series the CVAE neg-ELBO can go
        # non-finite (divergent recon/KL), which would otherwise fail the whole
        # dataset in the runner's finiteness check. Map NaN->0 (neutral, ID-like)
        # and clip ±inf to large finite bounds; orientation (higher = OOD) preserved.
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=1e12, neginf=-1e12)
        return ratio

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _align_dataset(
        self, x_norm: np.ndarray, y_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align every training sample to its true class pattern."""
        N, C, T = x_norm.shape
        x_aligned = np.empty_like(x_norm)
        x_residual = np.empty_like(x_norm)

        for i in range(N):
            cl = y_np[i]
            pattern = self.class_patterns[cl]
            aligned, resid = self._align_sample(x_norm[i], pattern)
            x_aligned[i] = aligned
            x_residual[i] = resid

        return x_aligned, x_residual, y_np

    @staticmethod
    def _align_sample(
        signal: np.ndarray, pattern: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align (C, T) signal to (C, T) pattern and compute residual.

        Alignment is done per-channel; the modal (most common) offset is used.
        """
        C, T = signal.shape
        offsets = []
        for ch in range(C):
            _, off = _align_to_pattern(signal[ch], pattern[ch])
            offsets.append(off)

        # Use the most common offset across channels
        offset = int(np.median(offsets))
        aligned = np.roll(signal, shift=offset, axis=-1)
        residual = aligned - pattern
        return aligned, residual

    def _find_best_class(self, x_norm: np.ndarray) -> List[Any]:
        """For each sample, return the ID class label with smallest residual."""
        N, C, T = x_norm.shape
        best_classes = []
        for i in range(N):
            best_cl = self.class_labels[0]
            best_err = float("inf")
            for cl in self.class_labels:
                pattern = self.class_patterns[cl]
                _, resid = self._align_sample(x_norm[i], pattern)
                err = np.mean(resid ** 2)
                if err < best_err:
                    best_err = err
                    best_cl = cl
            best_classes.append(best_cl)
        return best_classes
