"""DiverseMix (Class-D appendix build) — energy head + score-adaptive mixup, both arms.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Faithful re-build of DiverseMix ("Out-Of-Distribution Detection with Diversification
(Provably)", Yao et al., NeurIPS 2024), per methods/_validation/CLASS_D_DECISIONS.md
and BUILD_PLAN_CLASS_D.md §3.

The production benchmark1/models/ood_methods/diversemix.py fabricated its auxiliary
outliers by cross-class ID feature mixing (lands 100% inside the ID distribution, so
it sat at chance ~0.52 — see methods/diversemix/VERIFICATION.md) and scored with the
WRONG orientation (+logsumexp at :306). This Class-D build makes two faithful fixes:

  1. It trains the energy head on the REAL held-out auxiliary-outlier corpus
     (benchmark1/datasets/aux_outliers.py), with score-adaptive mixup
        lambda ~ Beta(s_hat_i * alpha, s_hat_j * alpha)
     pairing REAL aux outliers with ID samples (NOT ID-ID), where s_hat are the
     temperature-softmaxed energies (the paper's "score-adaptive diversification").
  2. The OOD score is -logsumexp of the energy head (the official
     eval_ood_detection.py orientation; higher = more OOD).

Training objective:
    L = CE(head(f_id), y_id) + omega * mean( relu( logsumexp(head(mixed_out)) + 1 ) )
pushing the mixed real-aux outliers' energy down while classifying ID.

Both fine-tune arms are run and the pair reported:
  * head_only : backbone frozen, only the energy head trains on frozen features
                (least-unfair vs the frozen-backbone 17).
  * full_net  : backbone + energy head train jointly (paper-faithful).

The backbone handed in is a DEEP COPY made by the runner, so the shared frozen
backbone the 17 production methods use is NEVER mutated. Appendix-only,
fair-comparison-BREAKING (trains an extra head on an external corpus) — never in the 17.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta


class EnergyHead(nn.Module):
    """Lightweight energy-based classifier head (feat_dim -> hidden -> n_classes)."""

    def __init__(self, feat_dim: int, n_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """logsumexp of the logits (the free energy, up to sign)."""
        return torch.logsumexp(self.forward(x), dim=1)


class DiverseMixClassD:
    """DiverseMix energy-head detector, both arms, real aux corpus, -logsumexp score.

    Interface (matches the appendix runner's per-sample fine-tune path):
      __init__(backbone, config)  -- backbone is a deep copy; config carries the
                                     shared head (for feature dim / class count is
                                     inferred from y), real aux windows, arm, hp.
      fit(x_id, y_id)             -- trains the energy head (and, for full_net, the
                                     backbone) with score-adaptive real-aux mixup.
      score(x) -> np.ndarray      -- -logsumexp of the energy head (higher = OOD).
    """

    EVAL_MODE = "per_sample_finetune"
    SCORE_TYPE = "energy"
    ARMS = ("head_only", "full_net")

    def __init__(self, backbone: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.bb = backbone
        self.config = config or {}
        self.arm = self.config.get("arm", "head_only")
        if self.arm not in ("head_only", "full_net"):
            raise ValueError(f"arm must be head_only/full_net, got {self.arm!r}")
        self.device = self.config.get("device", "cpu")
        self.seed = int(self.config.get("seed", 42))

        self.aux_x = self.config.get("aux_x")
        if self.aux_x is None:
            raise ValueError("DiverseMixClassD needs real aux windows in config['aux_x'].")
        self.aux_x = np.asarray(self.aux_x, dtype=np.float32)

        self.n_epochs = int(self.config.get("n_epochs", 10))
        self.alpha = float(self.config.get("alpha", 2.0))
        self.temperature_mix = float(self.config.get("temperature", 0.5))
        self.omega = float(self.config.get("omega", 0.5))
        self.hidden_dim = int(self.config.get("hidden_dim", 128))
        self.lr = float(self.config.get("lr", 1e-3))
        self.batch_size = int(self.config.get("batch_size", 64))

        self.energy_head: Optional[EnergyHead] = None
        self.n_classes: Optional[int] = None

    # -- feature extraction (backbone in eval for head_only, train for full_net) --
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.bb.model(x)

    def fit(self, x_id: np.ndarray, y_id: Optional[np.ndarray] = None) -> "DiverseMixClassD":
        if y_id is None:
            raise ValueError("DiverseMixClassD requires ID labels for training.")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        model = self.bb.model
        model.to(self.device)
        x_id = np.asarray(x_id, dtype=np.float32)
        y_id = np.asarray(y_id).astype(np.int64)
        self.n_classes = max(2, int(len(np.unique(y_id))))

        train_backbone = self.arm == "full_net"
        if train_backbone:
            for p in model.parameters():
                p.requires_grad_(True)
            model.train()
        else:
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()

        # infer feature dim
        with torch.no_grad():
            probe = model(torch.from_numpy(x_id[:1]).float().to(self.device))
        feat_dim = int(probe.shape[1])

        self.energy_head = EnergyHead(feat_dim, self.n_classes, self.hidden_dim).to(self.device)
        params = list(self.energy_head.parameters())
        if train_backbone:
            params = params + list(model.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)

        id_t = torch.from_numpy(x_id).float()
        y_t = torch.from_numpy(y_id).long()
        aux_t = torch.from_numpy(self.aux_x).float()
        n_id = len(x_id)
        n_aux = len(self.aux_x)

        for _ in range(self.n_epochs):
            self.energy_head.train()
            if train_backbone:
                model.train()
            perm_id = rng.permutation(n_id)
            perm_aux = rng.permutation(n_aux)
            for i in range(0, n_id, self.batch_size):
                bi = perm_id[i:i + self.batch_size]
                xb = id_t[bi].to(self.device)
                yb = y_t[bi].to(self.device)

                ai = (i // self.batch_size) * self.batch_size % max(n_aux, 1)
                aidx = perm_aux[ai:ai + len(bi)]
                if len(aidx) < len(bi):
                    aidx = np.concatenate([aidx, perm_aux[:len(bi) - len(aidx)]])
                xa = aux_t[aidx].to(self.device)

                # features
                if train_backbone:
                    feats_id = self._features(xb)
                    feats_aux = self._features(xa)
                else:
                    with torch.no_grad():
                        feats_id = self._features(xb)
                        feats_aux = self._features(xa)

                # score-adaptive mixup between REAL aux outliers and ID samples
                mixed_out = self._score_adaptive_mixup(feats_aux, feats_id)

                logits_id = self.energy_head(feats_id)
                energy_out = self.energy_head.energy(mixed_out)

                loss_ce = F.cross_entropy(logits_id, yb)
                loss_aux = torch.mean(F.relu(energy_out + 1.0))  # push aux energy < -1
                loss = loss_ce + self.omega * loss_aux

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.energy_head.eval()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)  # leave the deep copy in a clean state
        return self

    def _score_adaptive_mixup(self, feats_aux: torch.Tensor, feats_id: torch.Tensor) -> torch.Tensor:
        """Mix each REAL aux outlier with a random ID sample using score-adaptive Beta.

        lambda ~ Beta(s_hat_aux * alpha, s_hat_id * alpha), s_hat = softmax(energy / T).
        Produces boundary outliers biased toward the more-uncertain (higher-energy)
        endpoint (the paper's diversification), NOT ID-ID mixing.
        """
        b = feats_aux.size(0)
        if b < 1 or feats_id.size(0) < 1:
            return feats_aux
        # pair each aux with a random ID sample
        idx = torch.randint(0, feats_id.size(0), (b,), device=feats_id.device)
        paired_id = feats_id[idx]

        with torch.no_grad():
            e_aux = self.energy_head.energy(feats_aux)
            e_id = self.energy_head.energy(paired_id)
            s_aux = F.softmax(e_aux / self.temperature_mix, dim=0)
            s_id = F.softmax(e_id / self.temperature_mix, dim=0)

        a = np.clip(s_aux.detach().cpu().numpy() * self.alpha, 0.1, None)
        c = np.clip(s_id.detach().cpu().numpy() * self.alpha, 0.1, None)
        lam = np.empty(b, dtype=np.float32)
        for k in range(b):
            lam[k] = Beta(torch.tensor(float(a[k])), torch.tensor(float(c[k]))).sample().item()
        lam_t = torch.from_numpy(lam).to(feats_aux.device).view(-1, 1)
        return lam_t * feats_aux + (1.0 - lam_t) * paired_id

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.energy_head is None:
            raise RuntimeError("DiverseMixClassD must be fit before scoring.")
        self.energy_head.eval()
        model = self.bb.model
        model.eval()
        arr = np.asarray(x, dtype=np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), 256):
                xb = torch.from_numpy(arr[i:i + 256]).float().to(self.device)
                feats = model(xb)
                e = self.energy_head.energy(feats)          # logsumexp
                out.append((-e).detach().cpu().numpy())     # -logsumexp: higher = OOD
        if not out:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(out).astype(np.float64)
