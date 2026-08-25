"""DivOE (Class-D appendix build) — OE + diversified-outlier SYNTHESIS, both arms.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Faithful re-build of DivOE ("Diversified Outlier Exposure", Zhu et al., NeurIPS 2023;
repo ZFancy/DivOE, src/train_DivOE.py), per methods/_validation/CLASS_D_DECISIONS.md
and BUILD_PLAN_CLASS_D.md §2.

DivOE = OE PLUS synthesising DIVERSE informative outliers by multi-step input-space
projected-gradient ascent on the OE objective ("informative extrapolation",
train_DivOE.py:177-197): start from an auxiliary outlier, take `num_steps`
sign-gradient steps that INCREASE the OE uncertainty loss, projecting back into the
epsilon-ball; replace a fraction (`extrapolation_ratio` = 0.5) of the aux batch with
these synthesised outliers, then apply the OE fine-tune loss
`CE(id) + lambda * CE_to_uniform(synthesised ∪ aux)`. Scoring is the ENERGY score
-logsumexp(z/T) on the FINE-TUNED net (higher = more OOD) — identical to OE.

Per CLASS_D_DECISIONS the PGD runs in RAW INPUT SPACE (paper-faithful) on the
fine-tuned copy, with epsilon in NORMALISED-WINDOW units (num_steps=5, ratio=0.5).
No [0, 1] clamp is applied (that is an image-domain artefact; the windows here are
per-series/global normalised, not pixels) — the projection keeps x_adv within the
epsilon-ball around the original aux window only.

`extrapolate_pgd()` (the synthesis) lives here — the method's distinctive part; the
appendix runner's `finetune_divoe()` (parallel to `finetune()`) drives it on a deep
copy of the shared backbone across BOTH arms (head_only / full_net). `DivOEClassD`
is the same thin energy scorer as OE over the resulting fine-tuned net.
Appendix-only, fair-comparison-BREAKING (training + synthesis) — never in the 17.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def energy_scores(
    model: nn.Module,
    head: nn.Module,
    x: np.ndarray,
    temperature: float = 1.0,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Per-window energy score  -logsumexp(head(model(x)) / T)  (higher = more OOD).

    Same definition as outlier_exposure_classd.energy_scores; duplicated here so
    the module is self-contained under the runner's dynamic file loading (no package
    context, hence no cross-module relative import).
    """
    model.eval()
    head.eval()
    model.to(device)
    head.to(device)
    arr = np.asarray(x, dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(arr), batch_size):
            xb = torch.from_numpy(arr[i:i + batch_size]).float().to(device)
            logits = head(model(xb))
            e = -torch.logsumexp(logits / temperature, dim=-1)
            out.append(e.detach().cpu().numpy())
    if not out:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(out).astype(np.float64)


def extrapolate_pgd(
    model: nn.Module,
    head: nn.Module,
    data: torch.Tensor,
    epsilon: float,
    rel_step_size: float = 0.25,
    num_steps: int = 5,
    extrapolation_score: str = "MSP",
    m_out: float = -7.0,
    device: str = "cpu",
    rand_init: bool = True,
) -> torch.Tensor:
    """Input-space PGD synthesis of diversified outliers (train_DivOE.py:177-200).

    Starting from `data` (a batch of aux outlier windows), take `num_steps`
    sign-gradient ASCENT steps that increase the outlier-uncertainty loss, keeping
    x_adv within the epsilon-ball around `data`. Returns a detached tensor of the
    same shape. Only the INPUT carries gradient here — model/head parameters are
    left untouched (no grad accumulated into them), and their train/eval modes are
    saved and restored so a full_net fine-tune's BatchNorm statistics are unharmed.
    """
    was_model_training = model.training
    was_head_training = head.training
    model.eval()
    head.eval()
    data = data.to(device).detach()
    if rand_init:
        noise = torch.empty_like(data).uniform_(-epsilon, epsilon)
        x_adv = (data + noise).detach()
    else:
        x_adv = data.clone().detach()

    step_size = float(epsilon) * float(rel_step_size)
    for _ in range(int(num_steps)):
        x_adv.requires_grad_(True)
        logits = head(model(x_adv))
        if extrapolation_score == "energy":
            ec_out = -torch.logsumexp(logits, dim=1)
            loss_adv = torch.pow(F.relu(m_out - ec_out), 2).mean()
        else:  # 'MSP' — increase deviation from a uniform posterior
            loss_adv = -(logits.mean(dim=1) - torch.logsumexp(logits, dim=1)).mean()
        grad = torch.autograd.grad(loss_adv, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)

    if was_model_training:
        model.train()
    if was_head_training:
        head.train()
    return x_adv.detach()


class DivOEClassD:
    """Energy scorer over a DivOE-fine-tuned (backbone, head).

    Identical scoring interface to OutlierExposureClassD — the difference is in the
    fine-tuning (runner's `finetune_divoe`, which injects PGD-synthesised outliers).

      __init__(backbone, config)  -- backbone is the fine-tuned deep copy;
                                     config["classifier"] is the fine-tuned head.
      fit(x_id=None, y_id=None)   -- no-op (fine-tuning already done by the runner).
      score(x) -> np.ndarray      -- per-window energy, higher = more OOD.
    """

    EVAL_MODE = "per_sample_finetune"
    SCORE_TYPE = "energy"
    ARMS = ("head_only", "full_net")

    def __init__(self, backbone: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.bb = backbone
        self.config = config or {}
        self.head: Optional[nn.Module] = self.config.get("classifier")
        self.temperature = float(self.config.get("temperature", 1.0))
        self.device = self.config.get("device", "cpu")
        if self.head is None:
            raise ValueError(
                "DivOEClassD needs the fine-tuned head in config['classifier']."
            )

    def fit(self, x_id: Any = None, y_id: Any = None) -> "DivOEClassD":
        return self

    def score(self, x: Any) -> np.ndarray:
        return energy_scores(
            self.bb.model, self.head, x,
            temperature=self.temperature, device=self.device,
        )
