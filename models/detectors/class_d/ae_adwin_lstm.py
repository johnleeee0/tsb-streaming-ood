"""AE-ADWIN-LSTM (Class-D appendix build) — ordered-stream temporal detector.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Faithful re-build per methods/_validation/CLASS_D_DECISIONS.md and
BUILD_PLAN_CLASS_D.md §6, of the method described in
"A Novel Concept Drift Detection Model for Handling Evolving Patterns in
Multivariate Time Series" (IEEE APCI 2025, doi:10.1109/APCI65531.2025.11136854).
The paper is paywalled and no public code exists; this build follows the paper's
named components as summarised from secondary sources and fixes every defect the
production stand-in was flagged for (methods/ae_adwin_lstm/VERIFICATION.md):

  1. ORDERED stream (load_tsb(ordered_eval=True)) so the LSTM history and the
     ADWIN error stream are temporally meaningful — the production version scored
     a randomly permuted eval set (ρ≈0.80 between two shuffles).
  2. REAL ADWIN (class `ADWIN` below): exponential-histogram buckets, ALL cut
     points tested at bucket boundaries, Hoeffding cut with δ' = δ/n. The
     production "ADWIN" was a single midpoint two-sample mean test.
  3. DRIFT-TRIGGERED INCREMENTAL UPDATE of the AE+LSTM when ADWIN fires — the
     paper's defining mechanism, entirely absent from the production version.
  4. ADWIN reset at the start of every score_stream (idempotent scoring).
  5. Positional artefact fixed: the first seq_len-1 windows use the normalised
     prediction-error mean (0.0) consistently, so composition is uniform.
  6. ORIENTATION: higher error = more OOD (NO negation). The production version
     negated on a false premise and scored 0.25 AUROC (0.75 flipped).

Native output is a per-window scalar in stream order -> per_sample_auroc directly.
A drift-detection-delay secondary metric (windows between true drift onset and the
first ADWIN alarm) is also exposed via `drift_delay`.

Works on SHARED FROZEN backbone embeddings (a disclosed adaptation: the AE is a
feature-space MLP-AE, matching the production adaptation domain).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ===========================================================================
# Real ADWIN (Bifet & Gavaldà 2007) — exponential histogram + Hoeffding cut
# ===========================================================================

class ADWIN:
    """Adaptive Windowing change detector with an exponential-histogram window.

    Faithful to the algorithm's defining properties (unlike the midpoint
    two-sample test the production stand-in called "ADWIN"):

      * the window is summarised by buckets whose sizes are powers of two, with
        at most `max_buckets` buckets of each size (the exponential histogram);
      * on each check EVERY cut point at a bucket boundary is tested;
      * the cut threshold is the Hoeffding bound
            eps_cut = sqrt( 0.5 * (1/n0 + 1/n1) * ln(4/delta') ),  delta' = delta/n
        and the older sub-window is dropped whenever |mean0 - mean1| > eps_cut.

    The residual stream fed in is z-scored (roughly unit scale), so the Hoeffding
    (bounded-range) form of the cut is used; this is a standard ADWIN variant.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5,
                 min_window_len: int = 10) -> None:
        self.delta = float(delta)
        self.max_buckets = int(max_buckets)
        self.min_window_len = int(min_window_len)
        self.reset()

    def reset(self) -> None:
        # buckets ordered OLDEST -> NEWEST; each is [total, count(=power of two)]
        self.bucket: List[List[float]] = []
        self.width = 0
        self.total = 0.0
        self.drift_detected = False

    # -- exponential-histogram maintenance -------------------------------------
    def _insert(self, value: float) -> None:
        self.bucket.append([float(value), 1])   # newest, size 1, at the end
        self.width += 1
        self.total += float(value)
        self._compress()

    def _compress(self) -> None:
        # Ensure at most max_buckets buckets of each size; merge the two OLDEST
        # buckets of an over-full size (they are contiguous) into a double bucket.
        while True:
            merged_any = False
            sizes = sorted({b[1] for b in self.bucket})
            for s in sizes:
                idxs = [i for i, b in enumerate(self.bucket) if b[1] == s]
                if len(idxs) > self.max_buckets:
                    i0, i1 = idxs[0], idxs[1]          # contiguous, oldest pair
                    b0, b1 = self.bucket[i0], self.bucket[i1]
                    self.bucket[i0] = [b0[0] + b1[0], b0[1] + b1[1]]
                    del self.bucket[i1]
                    merged_any = True
                    break
            if not merged_any:
                break

    # -- drift check over all bucket-boundary cut points -----------------------
    def _detect(self) -> bool:
        changed = False
        if self.width < self.min_window_len:
            return False
        shrunk = True
        while shrunk and len(self.bucket) >= 2:
            shrunk = False
            n0 = 0
            s0 = 0.0
            for i in range(len(self.bucket) - 1):      # cut between i and i+1
                n0 += self.bucket[i][1]
                s0 += self.bucket[i][0]
                n1 = self.width - n0
                s1 = self.total - s0
                if n0 < 1 or n1 < 1:
                    continue
                m0 = s0 / n0
                m1 = s1 / n1
                inv = 1.0 / n0 + 1.0 / n1
                delta_prime = self.delta / max(self.width, 1)
                eps = math.sqrt(0.5 * inv * math.log(4.0 / delta_prime))
                if abs(m0 - m1) > eps:
                    old = self.bucket.pop(0)           # drop the oldest bucket
                    self.width -= old[1]
                    self.total -= old[0]
                    changed = True
                    shrunk = True
                    break
        return changed

    def update(self, value: float) -> bool:
        """Add a value; return True iff a change was detected on this step."""
        self._insert(value)
        self.drift_detected = self._detect()
        return self.drift_detected


# ===========================================================================
# Feature autoencoder (spatial) + LSTM next-step predictor (temporal)
# ===========================================================================

class _FeatureAE(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _LSTMPredictor(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=0.1 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ===========================================================================
# Detector
# ===========================================================================

class AEADWINLSTMClassD:
    """Ordered-stream AE + real-ADWIN + LSTM detector on frozen-backbone features.

    Interface:
      fit(id_windows)          -> train AE + LSTM on ID (ordered) features
      score_stream(stream_x)   -> (scores, alarms); scores higher = more OOD
                                  (per-window, in temporal order)
    """

    EVAL_MODE = "ordered_per_window"

    def __init__(self, backbone: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.bb = backbone
        cfg = config or {}
        self.device = cfg.get("device", "cpu")
        self.n_epochs_ae = int(cfg.get("n_epochs_ae", 20))
        self.n_epochs_lstm = int(cfg.get("n_epochs_lstm", 20))
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        self.lstm_layers = int(cfg.get("lstm_layers", 1))
        self.seq_len = int(cfg.get("seq_len", 10))
        self.adwin_delta = float(cfg.get("adwin_delta", 0.002))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.lr = float(cfg.get("lr", 1e-3))
        self.incremental_update = bool(cfg.get("incremental_update", True))
        self.incr_steps = int(cfg.get("incr_steps", 1))
        self.seed = int(cfg.get("seed", 42))

        self.ae: Optional[_FeatureAE] = None
        self.lstm: Optional[_LSTMPredictor] = None
        self.opt_ae = None
        self.opt_lstm = None
        self.adwin: Optional[ADWIN] = None
        self.mu_r = 0.0
        self.sd_r = 1.0
        self.mu_p = 0.0
        self.sd_p = 1.0
        self.last_alarms: Optional[np.ndarray] = None

    def _embed(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.bb.embed(np.asarray(x, dtype=np.float32)), dtype=np.float32)

    def fit(self, id_windows: np.ndarray) -> "AEADWINLSTMClassD":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        feats = torch.from_numpy(self._embed(id_windows)).float().to(self.device)  # (N,D) ordered
        N, feat_dim = feats.shape

        # ---- autoencoder ----
        self.ae = _FeatureAE(feat_dim, self.hidden_dim).to(self.device)
        self.opt_ae = torch.optim.Adam(self.ae.parameters(), lr=self.lr)
        self.ae.train()
        idx = np.arange(N)
        for _ in range(self.n_epochs_ae):
            np.random.shuffle(idx)
            for s in range(0, N, self.batch_size):
                b = feats[idx[s:s + self.batch_size]]
                self.opt_ae.zero_grad()
                loss = F.mse_loss(self.ae(b), b)
                loss.backward()
                self.opt_ae.step()
        self.ae.eval()
        with torch.no_grad():
            errs = F.mse_loss(self.ae(feats), feats, reduction="none").mean(dim=1)
            self.mu_r = float(errs.mean())
            self.sd_r = float(errs.std()) + 1e-6

        # ---- LSTM next-step predictor (needs ordered sequences) ----
        if N > self.seq_len:
            seqs = torch.stack([feats[i:i + self.seq_len] for i in range(N - self.seq_len)])
            tgts = torch.stack([feats[i + self.seq_len] for i in range(N - self.seq_len)])
            self.lstm = _LSTMPredictor(feat_dim, self.hidden_dim, self.lstm_layers).to(self.device)
            self.opt_lstm = torch.optim.Adam(self.lstm.parameters(), lr=self.lr)
            self.lstm.train()
            sidx = np.arange(len(seqs))
            for _ in range(self.n_epochs_lstm):
                np.random.shuffle(sidx)
                for s in range(0, len(seqs), self.batch_size):
                    bb = sidx[s:s + self.batch_size]
                    self.opt_lstm.zero_grad()
                    loss = F.mse_loss(self.lstm(seqs[bb]), tgts[bb])
                    loss.backward()
                    self.opt_lstm.step()
            self.lstm.eval()
            with torch.no_grad():
                perr = F.mse_loss(self.lstm(seqs), tgts, reduction="none").mean(dim=1)
                self.mu_p = float(perr.mean())
                self.sd_p = float(perr.std()) + 1e-6
        else:
            self.lstm = None

        self.adwin = ADWIN(delta=self.adwin_delta)
        return self

    def _incremental_step(self, buffer: List[torch.Tensor]) -> None:
        """Drift-triggered incremental update — the paper's defining mechanism."""
        if not self.incremental_update or len(buffer) == 0:
            return
        batch = torch.stack(buffer).to(self.device)          # (k, D)
        self.ae.train()
        for _ in range(self.incr_steps):
            self.opt_ae.zero_grad()
            F.mse_loss(self.ae(batch), batch).backward()
            self.opt_ae.step()
        self.ae.eval()
        if self.lstm is not None and len(buffer) > self.seq_len:
            seqs = torch.stack([batch[i:i + self.seq_len] for i in range(len(buffer) - self.seq_len)])
            tgts = torch.stack([batch[i + self.seq_len] for i in range(len(buffer) - self.seq_len)])
            self.lstm.train()
            for _ in range(self.incr_steps):
                self.opt_lstm.zero_grad()
                F.mse_loss(self.lstm(seqs), tgts).backward()
                self.opt_lstm.step()
            self.lstm.eval()

    def score_stream(self, stream_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.ae is None or self.adwin is None:
            raise RuntimeError("AEADWINLSTMClassD must be fit before scoring.")
        feats = torch.from_numpy(self._embed(stream_x)).float().to(self.device)  # (N,D) ordered
        N = len(feats)
        self.adwin.reset()

        scores = np.empty(N, dtype=np.float64)
        alarms = np.zeros(N, dtype=bool)
        history: deque = deque(maxlen=self.seq_len)
        recent: deque = deque(maxlen=self.seq_len + 1)

        for i in range(N):
            e = feats[i:i + 1]                                # (1,D)
            with torch.no_grad():
                recon_err = float(F.mse_loss(self.ae(e), e))
            recon_norm = (recon_err - self.mu_r) / self.sd_r

            pred_norm = 0.0                                   # normalised mean for warm-up
            if self.lstm is not None and len(history) == self.seq_len:
                seq = torch.cat(list(history), dim=0).unsqueeze(0)  # (1,seq_len,D)
                with torch.no_grad():
                    pred_err = float(F.mse_loss(self.lstm(seq), e))
                pred_norm = (pred_err - self.mu_p) / self.sd_p

            combined = recon_norm + pred_norm                # residual stream -> ADWIN
            drift = self.adwin.update(combined)
            alarms[i] = drift

            # higher error = more OOD (NO negation — the corrected orientation)
            scores[i] = 0.5 * recon_norm + 0.5 * pred_norm

            history.append(e.detach())
            recent.append(feats[i].detach())
            if drift:
                self._incremental_step(list(recent))

        self.last_alarms = alarms
        return scores, alarms


def drift_delay(alarms: np.ndarray, stream_y: np.ndarray) -> float:
    """Secondary metric: windows between true drift onset and the first alarm.

    True onset = index of the first OOD window in the ordered stream. Returns the
    (signed) gap first_alarm_at_or_after_onset - onset, or NaN if there is no
    onset or no alarm at/after it.
    """
    y = np.asarray(stream_y).astype(int)
    a = np.asarray(alarms).astype(bool)
    ood = np.where(y == 1)[0]
    if len(ood) == 0:
        return float("nan")
    onset = int(ood[0])
    later = np.where(a)[0]
    later = later[later >= onset]
    if len(later) == 0:
        return float("nan")
    return float(int(later[0]) - onset)
