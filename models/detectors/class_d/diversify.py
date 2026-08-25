"""DIVERSIFY (Class-D appendix build) — from-scratch adversarial (GRL) extractor
for 1-D time-series windows, with an INVENTED OOD score.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Faithful-as-possible re-build per methods/_validation/CLASS_D_DECISIONS.md (§7) and
BUILD_PLAN_CLASS_D.md §7, of DIVERSIFY (Lu et al., "Out-of-Distribution Representation
Learning for Time Series Classification", ICLR 2023, arXiv:2209.07027; TPAMI
extension doi:10.1109/TPAMI.2024.3355212).

Why this is a from-scratch build and not the production benchmark1/models/ood_methods/
diversify.py: the production stand-in trains ONLY K centroid vectors on a permanently
FROZEN shared backbone (see methods/diversify/VERIFICATION.md). Official DIVERSIFY
*retrains the feature extractor* adversarially through a Gradient Reversal Layer (GRL)
so the representation itself captures worst-case latent domains. DIVERSIFY *is* the
retraining, so this build:

  1. Trains its OWN feature extractor from scratch on the ID train windows — it does
     NOT reuse the shared frozen ResNet backbone (that would defeat the method).
  2. Runs a minimal DANN-style adversarial loop adapted to 1-D windows:
       * a 1-D CNN feature extractor  F,
       * a class / pseudo-label head  C  (classifies ID pseudo-classes),
       * a latent-domain characterizer that (re)assigns latent-domain labels each
         epoch by k-means (COSINE distance on L2-normalised features — the paper's
         geometry), and
       * a domain-adversarial branch  D  through a GRL: D is trained to separate the
         latent domains (maximising latent-domain DIVERSITY), while the reversed
         gradient forces F to become domain-invariant given the class (the paper's
         class-invariance / distribution-matching objective).
  3. GUARDS the N < latent_domain_num crash the production version hit (three empty
     TSB-U result dirs, VERIFICATION §4): latent_domain_num is clamped to the data
     (>= 2 samples/cluster), and the adversarial branch is SKIPPED entirely when the
     stream is too small to form >= 2 domains (single-domain fallback, no crash).

INVENTED OOD SCORE (the paper defines NONE — it is a domain-generalisation
classifier, not an OOD detector):
  * primary   = ENERGY, -logsumexp of the class head's logits over the learned
                features (higher = more OOD);
  * secondary = COSINE-CENTROID, cosine distance to the nearest class centroid on
                L2-normalised learned features (higher = more OOD).
Both are exposed; score(x) defaults to energy, score(x, variant="cosine") gives the
alternate. Appendix-only, fair-comparison-BREAKING (learns its own representation);
captioned honestly "score added; paper defines none".
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


# ===========================================================================
# Gradient Reversal Layer (Ganin & Lempitsky 2015) — the DANN/DIVERSIFY device
# ===========================================================================

class _GradReverse(Function):
    """Identity forward; on backward, multiplies the gradient by -alpha.

    This is the mechanism the production stand-in lacks: it lets a single forward
    pass train the domain discriminator to SEPARATE latent domains while the
    extractor is pushed (reversed gradient) to CONFUSE them.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # noqa: D401
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # noqa: D401
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


# ===========================================================================
# From-scratch 1-D feature extractor + heads
# ===========================================================================

class _Extractor1D(nn.Module):
    """Compact 1-D CNN featurizer for TSB windows (C, T) -> feat_dim vector."""

    def __init__(self, in_channels: int, feat_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.proj(h)


class _Head(nn.Module):
    """Linear classifier head (feat_dim -> n_out)."""

    def __init__(self, feat_dim: int, n_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, max(2, int(n_out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ===========================================================================
# Detector
# ===========================================================================

class DiversifyClassD:
    """From-scratch DIVERSIFY with GRL adversarial latent-domain training + an
    invented OOD score.

    Interface (matches the appendix runner's self-train Group-III path):
      __init__(config)            -- trains its OWN extractor; no backbone needed.
                                     (An optional first positional arg is accepted
                                     and IGNORED so the shared _load_detector /
                                     cls(bb, cfg) convention still works.)
      fit(id_train_windows[, y])  -- trains extractor + class head + domain branch.
      score(x, variant="energy")  -- per-sample scores, higher = more OOD.
                                     variant in {"energy", "cosine"}.
    """

    EVAL_MODE = "per_sample_selftrain"
    SCORE_TYPE = "energy"
    SCORE_VARIANTS = ("energy", "cosine")

    def __init__(self, config: Optional[Dict[str, Any]] = None, *_ignored: Any) -> None:
        # Accept either DiversifyClassD(cfg) or DiversifyClassD(bb, cfg): if the
        # first arg is not a dict, treat the (ignored) backbone as first and read
        # the config from the trailing positional. This method NEVER uses a
        # shared backbone — it trains its own extractor.
        if config is not None and not isinstance(config, dict):
            cfg = _ignored[0] if _ignored and isinstance(_ignored[0], dict) else {}
        else:
            cfg = config or {}
        self.config = dict(cfg)

        self.latent_domain_num = int(self.config.get("latent_domain_num", 5))
        self.epochs = int(self.config.get("epochs", 30))
        self.alpha = float(self.config.get("alpha", 1.0))  # GRL strength
        self.lr = float(self.config.get("lr", 1e-3))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.feat_dim = int(self.config.get("feat_dim", 64))
        self.temperature = float(self.config.get("temperature", 1.0))
        self.device = self.config.get("device", "cpu")
        self.seed = int(self.config.get("seed", 42))
        self.min_per_domain = int(self.config.get("min_per_domain", 2))

        self.extractor: Optional[_Extractor1D] = None
        self.class_head: Optional[_Head] = None
        self.domain_head: Optional[_Head] = None
        self.n_classes: Optional[int] = None
        self.eff_domain_num: int = 1
        self.class_centroids: Optional[np.ndarray] = None  # L2-normalised (K, D)

    # -- latent-domain assignment: k-means on COSINE distance (paper geometry) --
    def _assign_domains(self, feats: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """k-means (cosine) on L2-normalised features -> domain labels in [0, k).

        Guarded: caller clamps k to the data; here we defend again and return all
        zeros (single domain) if k <= 1 or feats too small.
        """
        n = len(feats)
        if k <= 1 or n < 2:
            return np.zeros(n, dtype=np.int64)
        fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        # k-means++ style init on cosine (= euclidean on the unit sphere)
        centers = [fn[rng.integers(n)]]
        for _ in range(1, k):
            d = 1.0 - np.max(fn @ np.stack(centers).T, axis=1)  # cosine distance
            d = np.clip(d, 0, None)
            if d.sum() <= 1e-12:
                centers.append(fn[rng.integers(n)])
                continue
            probs = d / d.sum()
            centers.append(fn[rng.choice(n, p=probs)])
        C = np.stack(centers)
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(10):
            sims = fn @ C.T
            labels = np.argmax(sims, axis=1)
            newC = []
            for j in range(k):
                m = labels == j
                if m.any():
                    c = fn[m].mean(axis=0)
                    c = c / (np.linalg.norm(c) + 1e-8)
                else:
                    c = fn[rng.integers(n)]
                newC.append(c)
            newC = np.stack(newC)
            if np.allclose(newC, C, atol=1e-5):
                C = newC
                break
            C = newC
        return labels.astype(np.int64)

    def _clamp_domain_num(self, n: int) -> int:
        """CRITICAL guard against the N < latent_domain_num crash.

        Clamp so every domain can hold >= min_per_domain samples; a stream too
        small for >= 2 domains falls back to a single domain (adversarial branch
        skipped). Never returns a value that would sample more clusters than points.
        """
        k = min(self.latent_domain_num, max(1, n // max(1, self.min_per_domain)))
        return int(max(1, k))

    def _batches(self, n: int, rng: np.random.Generator):
        idx = rng.permutation(n)
        for s in range(0, n, self.batch_size):
            yield idx[s:s + self.batch_size]

    def fit(self, id_train_windows: np.ndarray, y: Optional[np.ndarray] = None) -> "DiversifyClassD":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        x = np.asarray(id_train_windows, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"expected (N, C, T) windows, got shape {x.shape}")
        n, in_channels, _T = x.shape

        # pseudo-class labels: use provided y, else a single trivial class
        if y is not None:
            y_arr = np.asarray(y).astype(np.int64)
            self.n_classes = max(2, int(len(np.unique(y_arr))))
            # remap labels to a contiguous [0, n_classes) range
            uniq = {v: i for i, v in enumerate(sorted(np.unique(y_arr)))}
            y_arr = np.array([uniq[v] for v in y_arr], dtype=np.int64)
        else:
            y_arr = np.zeros(n, dtype=np.int64)
            self.n_classes = 2

        # ---- clamp latent-domain count to the data (the crash guard) ----
        self.eff_domain_num = self._clamp_domain_num(n)
        use_adv = self.eff_domain_num >= 2

        self.extractor = _Extractor1D(in_channels, self.feat_dim).to(self.device)
        self.class_head = _Head(self.feat_dim, self.n_classes).to(self.device)
        self.domain_head = _Head(self.feat_dim, self.eff_domain_num).to(self.device)

        params = list(self.extractor.parameters()) + list(self.class_head.parameters())
        if use_adv:
            params += list(self.domain_head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)

        xt = torch.from_numpy(x).float().to(self.device)
        yt = torch.from_numpy(y_arr).long().to(self.device)
        dom_labels = np.zeros(n, dtype=np.int64)

        for ep in range(self.epochs):
            # (a) latent-domain characterization: re-cluster on current features
            if use_adv:
                self.extractor.eval()
                with torch.no_grad():
                    feats_all = self.extractor(xt).cpu().numpy()
                dom_labels = self._assign_domains(feats_all, self.eff_domain_num, rng)
            dt = torch.from_numpy(dom_labels).long().to(self.device)

            # (b) adversarial update: class CE + domain CE through the GRL
            self.extractor.train(); self.class_head.train()
            if use_adv:
                self.domain_head.train()
            for bi in self._batches(n, rng):
                bt = torch.from_numpy(bi).long().to(self.device)
                feats = self.extractor(xt[bt])
                logits_c = self.class_head(feats)
                loss = F.cross_entropy(logits_c, yt[bt])
                if use_adv:
                    rev = grad_reverse(feats, self.alpha)
                    logits_d = self.domain_head(rev)
                    loss = loss + F.cross_entropy(logits_d, dt[bt])
                opt.zero_grad()
                loss.backward()
                opt.step()

        # ---- store L2-normalised class centroids for the cosine-centroid score ----
        self.extractor.eval(); self.class_head.eval()
        with torch.no_grad():
            feats_all = self.extractor(xt).cpu().numpy()
        fn = feats_all / (np.linalg.norm(feats_all, axis=1, keepdims=True) + 1e-8)
        cents = []
        for c in range(self.n_classes):
            m = y_arr == c
            if m.any():
                v = fn[m].mean(axis=0)
                v = v / (np.linalg.norm(v) + 1e-8)
                cents.append(v)
        if not cents:  # degenerate: use the global centroid
            v = fn.mean(axis=0); v = v / (np.linalg.norm(v) + 1e-8)
            cents = [v]
        self.class_centroids = np.stack(cents).astype(np.float32)
        return self

    # -- feature extraction (batched, no grad) --
    def _features(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        out = []
        self.extractor.eval()
        with torch.no_grad():
            for i in range(0, len(arr), 256):
                xb = torch.from_numpy(arr[i:i + 256]).float().to(self.device)
                out.append(self.extractor(xb).cpu().numpy())
        if not out:
            return np.empty((0, self.feat_dim), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    def _energy(self, feats: np.ndarray) -> np.ndarray:
        """Primary invented score: energy = -logsumexp(logits/T). Higher = OOD."""
        with torch.no_grad():
            ft = torch.from_numpy(feats).float().to(self.device)
            logits = self.class_head(ft) / self.temperature
            lse = torch.logsumexp(logits, dim=1).cpu().numpy()
        return (-lse).astype(np.float64)

    def _cosine_centroid(self, feats: np.ndarray) -> np.ndarray:
        """Secondary invented score: cosine distance to nearest class centroid on
        L2-normalised features. Higher = OOD."""
        fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sims = fn @ self.class_centroids.T           # (N, K) cosine similarity
        return (1.0 - np.max(sims, axis=1)).astype(np.float64)

    def score(self, x: np.ndarray, variant: str = "energy") -> np.ndarray:
        if self.extractor is None or self.class_head is None:
            raise RuntimeError("DiversifyClassD must be fit before scoring.")
        if variant not in self.SCORE_VARIANTS:
            raise ValueError(f"variant must be one of {self.SCORE_VARIANTS}, got {variant!r}")
        feats = self._features(x)
        if len(feats) == 0:
            return np.empty((0,), dtype=np.float64)
        if variant == "energy":
            return self._energy(feats)
        return self._cosine_centroid(feats)
