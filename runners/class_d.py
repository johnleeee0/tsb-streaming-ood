"""Class-D appendix sweep — orchestration for the 7 excluded detectors.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Ported verbatim (behaviour-preserving) from experiments/run_class_d.py and rewired to
the THESIS_FINAL package scheme:

  * runners.pipeline   -> set_seed, train_backbone (was run_experiments as RE)
  * data.tsb_loader    -> load_tsb (with ordered_eval / dataset["stream"])
  * data.aux_outliers  -> get_eval_files / get_aux_windows / build_manifest
  * core.metrics       -> compute_auroc / compute_aupr / compute_fpr95
  * models.detectors.class_d.<name> -> the 7 faithful Class-D builds (in-package)

This is a SEPARATE runner for the Class-D appendix study (the 7 detectors that cannot
join the fair-comparison 17-method leaderboard). It never touches the production
harness (runners/run.py --group main / runners/pipeline.py) or its resumable results,
and writes only to results/class_d_*.csv.

Shared plumbing provided here (NOT the detectors themselves):

  * clone_backbone(bb)            -> deep copy of the shared trained backbone, so the
                                     shared frozen backbone the 17 use is NEVER mutated.
  * finetune(...) / finetune_divoe(...) -> a NEW (bb, head) trained with the OE
                                     objective; arm selects head-only / full-net.
  * make_monitoring_batches(...)  -> batches of B consecutive windows + batch labels.
  * batch_level_auroc(...)        -> AUROC at batch granularity (driftlens).
  * per_sample_auroc(...)         -> the existing per-sample metric, reused.
  * CLASS_D_REGISTRY              -> the 7-entry detector registry (eval_mode metadata).
  * run_group1 / run_group2 / run_group3 -> the group sweeps.

The public entrypoint is `run_class_d(splits, n_per_cell, group=...)`, dispatched from
runners/run.py when --group class_d is given.
"""

from __future__ import annotations

import copy
import csv
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --- repo root on sys.path so core/, data/, models/ import ---------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.metrics import compute_aupr, compute_auroc, compute_fpr95  # noqa: E402
from data import aux_outliers as AUX  # noqa: E402
from data.tsb_loader import load_tsb  # noqa: E402
from runners import pipeline  # noqa: E402

# The 7 faithful Class-D detector builds (in-package; no dynamic file loading).
from models.detectors.class_d import ae_adwin_lstm as _ae_adwin_lstm  # noqa: E402
from models.detectors.class_d import diversemix as _diversemix  # noqa: E402
from models.detectors.class_d import diversify as _diversify  # noqa: E402
from models.detectors.class_d import divoe as _divoe  # noqa: E402
from models.detectors.class_d import driftlens as _driftlens  # noqa: E402
from models.detectors.class_d import outlier_exposure as _outlier_exposure  # noqa: E402
from models.detectors.class_d import tdivdm as _tdivdm  # noqa: E402

SEED = 42
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")


# ===========================================================================
# Backbone cloning — never mutate the shared trained backbone
# ===========================================================================

def clone_backbone(bb: Any) -> Any:
    """Deep-copy the shared trained backbone (and its inner nn.Module)."""
    return copy.deepcopy(bb)


def _clone_head(head: nn.Module) -> nn.Module:
    return copy.deepcopy(head)


# ===========================================================================
# Data loaders
# ===========================================================================

def make_loader(x: np.ndarray, y: Optional[np.ndarray] = None,
                batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """Wrap arrays in a DataLoader. If y is None, the loader yields (x,) tuples."""
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).float()
    if y is None:
        ds = TensorDataset(xt)
    else:
        yt = torch.from_numpy(np.asarray(y)).long()
        ds = TensorDataset(xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _oe_uniform_loss(logits: torch.Tensor) -> torch.Tensor:
    """OE cross-entropy-to-uniform term: -(mean_k z_k - logsumexp_k z_k).

    Matches reference/CIFAR/oe_tune.py — pushes aux-outlier logits toward a
    uniform posterior. Averaged over the aux batch.
    """
    return -(logits.mean(dim=1) - torch.logsumexp(logits, dim=1)).mean()


# ===========================================================================
# Generic fine-tuning helper (group-I methods build on this)
# ===========================================================================

def finetune(
    backbone: Any,
    head: nn.Module,
    id_loader: DataLoader,
    aux_loader: Optional[DataLoader],
    arm: str = "head_only",
    epochs: int = 10,
    lr: float = 1e-3,
    oe_weight: float = 0.5,
    seed: int = SEED,
    device: str = "cpu",
) -> Tuple[Any, nn.Module]:
    """Fine-tune a DEEP COPY of (backbone, head) with the OE objective.

    L = CE(f(x_id), y_id) + oe_weight * CE_to_uniform(f(x_aux))

    arm:
      'head_only' — freeze the ResNet, train only the linear head (least-unfair
                    vs the frozen-backbone 17).
      'full_net'  — train backbone + head (paper-faithful).

    Returns a NEW (backbone, head). The inputs are never mutated: both are
    deep-copied before any parameter update.
    """
    if arm not in ("head_only", "full_net"):
        raise ValueError(f"arm must be 'head_only' or 'full_net', got {arm!r}")

    pipeline.set_seed(seed)
    bb2 = clone_backbone(backbone)
    head2 = _clone_head(head)
    model = bb2.model
    model.to(device)
    head2.to(device)

    if arm == "head_only":
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()                       # keep BN stats frozen
        params = list(head2.parameters())
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        params = list(model.parameters()) + list(head2.parameters())
    head2.train()

    opt = torch.optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss()

    aux_iter_src = list(aux_loader) if aux_loader is not None else []

    for _ in range(epochs):
        ai = 0
        for id_batch in id_loader:
            x_id, y_id = id_batch
            x_id = x_id.to(device)
            y_id = y_id.to(device)
            opt.zero_grad()

            feat_id = model(x_id)
            logits_id = head2(feat_id)
            loss = ce(logits_id, y_id)

            if aux_iter_src:
                x_aux = aux_iter_src[ai % len(aux_iter_src)][0].to(device)
                ai += 1
                feat_aux = model(x_aux)
                logits_aux = head2(feat_aux)
                loss = loss + oe_weight * _oe_uniform_loss(logits_aux)

            loss.backward()
            opt.step()

    model.eval()
    head2.eval()
    for p in model.parameters():
        p.requires_grad_(True)             # leave the copy in a clean state
    return bb2, head2


def finetune_divoe(
    backbone: Any,
    head: nn.Module,
    id_loader: DataLoader,
    aux_loader: Optional[DataLoader],
    arm: str = "head_only",
    pgd_fn: Optional[Callable] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    oe_weight: float = 0.5,
    epsilon: float = 0.1,
    num_steps: int = 5,
    rel_step_size: float = 0.25,
    extrapolation_ratio: float = 0.5,
    extrapolation_score: str = "MSP",
    seed: int = SEED,
    device: str = "cpu",
) -> Tuple[Any, nn.Module]:
    """DivOE fine-tune: OE objective with PGD-SYNTHESISED diversified outliers.

    Parallel to finetune(): deep-copies (backbone, head) then, per aux batch,
    replaces an `extrapolation_ratio` fraction with input-space PGD-synthesised
    outliers (`pgd_fn`, from models/detectors/class_d/divoe.py::extrapolate_pgd)
    before applying L = CE(id) + oe_weight * CE_to_uniform(synthesised ∪ aux).

    Never mutates the inputs (both are deep-copied). `arm` selects head_only /
    full_net exactly as finetune().
    """
    if arm not in ("head_only", "full_net"):
        raise ValueError(f"arm must be 'head_only' or 'full_net', got {arm!r}")
    if pgd_fn is None:
        raise ValueError("finetune_divoe requires pgd_fn (extrapolate_pgd).")

    pipeline.set_seed(seed)
    bb2 = clone_backbone(backbone)
    head2 = _clone_head(head)
    model = bb2.model
    model.to(device)
    head2.to(device)

    if arm == "head_only":
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        params = list(head2.parameters())
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        params = list(model.parameters()) + list(head2.parameters())
    head2.train()

    opt = torch.optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss()
    aux_iter_src = list(aux_loader) if aux_loader is not None else []

    for _ in range(epochs):
        ai = 0
        for x_id, y_id in id_loader:
            x_id = x_id.to(device)
            y_id = y_id.to(device)
            opt.zero_grad()

            feat_id = model(x_id)
            logits_id = head2(feat_id)
            loss = ce(logits_id, y_id)

            if aux_iter_src:
                x_aux = aux_iter_src[ai % len(aux_iter_src)][0].to(device)
                ai += 1
                k = int(len(x_aux) * extrapolation_ratio)
                if k > 0:
                    adv = pgd_fn(
                        model, head2, x_aux[:k], epsilon,
                        rel_step_size=rel_step_size, num_steps=num_steps,
                        extrapolation_score=extrapolation_score, device=device,
                    )
                    x_comb = torch.cat([adv.to(device), x_aux[k:]], dim=0)
                else:
                    x_comb = x_aux
                logits_aux = head2(model(x_comb))
                loss = loss + oe_weight * _oe_uniform_loss(logits_aux)

            loss.backward()
            opt.step()

    model.eval()
    head2.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return bb2, head2


# ===========================================================================
# Ordered / window-level evaluation path
# ===========================================================================

def make_monitoring_batches(
    stream_x: np.ndarray,
    stream_y: np.ndarray,
    batch_size: int,
    batch_stride: Optional[int] = None,
    tau: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Group a temporally-ordered window stream into monitoring batches.

    Returns:
      batches       : (M, B, C, T) — M batches of B consecutive windows
      batch_labels  : (M,)         — 1 iff frac(OOD windows) >= tau, else 0

    Non-overlapping by default (batch_stride == batch_size). A trailing partial
    batch (< batch_size) is dropped so every batch has exactly B windows.
    """
    x = np.asarray(stream_x)
    y = np.asarray(stream_y)
    B = int(batch_size)
    stride = int(batch_stride) if batch_stride else B
    if B <= 0 or len(x) < B:
        return (np.empty((0, B) + x.shape[1:], dtype=x.dtype),
                np.empty((0,), dtype=np.int64))

    starts = list(range(0, len(x) - B + 1, stride))
    batches = np.stack([x[s:s + B] for s in starts], axis=0)
    labels = np.array(
        [1 if (y[s:s + B].mean() >= tau) else 0 for s in starts], dtype=np.int64
    )
    return batches, labels


def batch_level_auroc(
    stream_x: np.ndarray,
    stream_y: np.ndarray,
    batch_score_fn: Callable[[np.ndarray], float],
    batch_size: int,
    batch_stride: Optional[int] = None,
    tau: float = 0.5,
) -> float:
    """AUROC at batch granularity for window-level detectors (driftlens etc.).

    batch_score_fn maps a batch (B, C, T) -> scalar (higher = more OOD).
    Returns np.nan if fewer than two batch classes are present.
    """
    batches, labels = make_monitoring_batches(
        stream_x, stream_y, batch_size, batch_stride, tau
    )
    if len(batches) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    scores = np.asarray([float(batch_score_fn(b)) for b in batches], dtype=np.float64)
    if not np.isfinite(scores).all():
        return float("nan")
    return compute_auroc(labels, scores)


def per_sample_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Per-sample AUROC (the existing metric), for ordered per-window methods."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64).ravel()
    if len(np.unique(y)) < 2 or not np.isfinite(s).all():
        return float("nan")
    return compute_auroc(y, s)


# ===========================================================================
# Class-D detector registry — the 7 excluded detectors
# ===========================================================================
#
# Each entry carries enough metadata for the sweep to know HOW to evaluate it:
#   module / class      : the in-package faithful Class-D build (models/detectors/
#                         class_d/<id>.py — NOT the production models/detectors/ set).
#   eval_mode           : "batch_level"          -> batch_level_auroc (driftlens)
#                         "ordered_per_window"    -> per_sample_auroc on the stream
#                         "per_sample_finetune"   -> OE-family fine-tune + energy
#                         "per_sample_selftrain"  -> DIVERSIFY from-scratch GRL
#   params              : detector constructor params.
#   batch_size / tau    : monitoring-batch settings for batch_level detectors
#                         (B=32 for U, 16 for M; tau=0.5 — per CLASS_D_DECISIONS).
#   caption             : the honest label carried into the appendix / VERIFICATION.

_CLASS_D_MODULES = {
    "driftlens": _driftlens,
    "ae_adwin_lstm": _ae_adwin_lstm,
    "tdivdm": _tdivdm,
    "outlier_exposure": _outlier_exposure,
    "divoe": _divoe,
    "diversemix": _diversemix,
    "diversify": _diversify,
}

CLASS_D_REGISTRY: Dict[str, Dict[str, Any]] = {
    "driftlens": {
        "module": _driftlens,
        "class": "DriftLensClassD",
        "eval_mode": "batch_level",
        "params": {"n_pc": 20},
        "batch_size": {"U": 32, "M": 16},
        "tau": 0.5,
        "caption": "Batch-level Fréchet (Wasserstein-2) drift vs an ID baseline "
                   "(official DriftLens granularity; per-sample proxy omitted — "
                   "it duplicates mahalanobis).",
    },
    "ae_adwin_lstm": {
        "module": _ae_adwin_lstm,
        "class": "AEADWINLSTMClassD",
        "eval_mode": "ordered_per_window",
        "params": {"n_epochs_ae": 20, "n_epochs_lstm": 20, "hidden_dim": 64,
                   "lstm_layers": 1, "seq_len": 10, "adwin_delta": 0.002,
                   "batch_size": 64, "lr": 1e-3, "incremental_update": True},
        "caption": "AE reconstruction + real ADWIN (exp-histogram, all cut points, "
                   "Hoeffding delta/n) + LSTM next-step with drift-triggered "
                   "incremental update; per-window AUROC on the ordered stream, "
                   "corrected orientation; drift-delay reported as secondary.",
    },
    "tdivdm": {
        "module": _tdivdm,
        "class": "TDIVDMClassD",
        "eval_mode": "ordered_per_window",
        "params": {"scales": [5, 10, 20], "bandwidth": "scott"},
        "caption": "TD-IVDM-inspired (unverifiable — paper paywalled, no code): "
                   "multi-scale Gaussian KDE over PCA subspaces of frozen "
                   "backbone embeddings; per-window neg-log-density aggregated "
                   "across scales.",
    },
    # ----------------------------------------------------------------------
    # GROUP I — auxiliary-outlier corpus + fine-tuning (OE family).
    # eval_mode "per_sample_finetune": per (method, arm) the runner deep-copies +
    # fine-tunes the shared backbone on a channel-matched hold-out aux corpus, then
    # scores val/test with ENERGY -> per_sample_auroc. BOTH arms (head_only, full_net)
    # are run and reported. The shared frozen backbone is NEVER mutated.
    # ----------------------------------------------------------------------
    "outlier_exposure": {
        "module": _outlier_exposure,
        "class": "OutlierExposureClassD",
        "eval_mode": "per_sample_finetune",
        "train_kind": "oe",
        "score_type": "energy",
        "arms": ["head_only", "full_net"],
        "params": {"epochs": 10, "lr": 1e-3, "oe_weight": 0.5,
                   "batch_size": 64, "oe_batch_size": 64, "temperature": 1.0},
        "caption": "Faithful Outlier Exposure (Hendrycks 2019): fine-tune "
                   "(backbone+head copy) with CE(id) + 0.5*CE-to-uniform(aux over a "
                   "hold-out TSB aux corpus); score = energy -logsumexp on the "
                   "fine-tuned net; both arms (head_only / full_net). "
                   "Appendix only — BREAKS the frozen-backbone fair comparison.",
    },
    "divoe": {
        "module": _divoe,
        "class": "DivOEClassD",
        "eval_mode": "per_sample_finetune",
        "train_kind": "divoe",
        "score_type": "energy",
        "arms": ["head_only", "full_net"],
        "params": {"epochs": 10, "lr": 1e-3, "oe_weight": 0.5,
                   "batch_size": 64, "oe_batch_size": 64, "temperature": 1.0,
                   "epsilon": 0.1, "num_steps": 5, "rel_step_size": 0.25,
                   "extrapolation_ratio": 0.5, "extrapolation_score": "MSP"},
        "caption": "Faithful DivOE (Zhu 2023): OE + input-space PGD synthesis of "
                   "diversified outliers (num_steps=5, ratio=0.5, eps in normalised "
                   "units) folded into the OE fine-tune; score = energy on the "
                   "fine-tuned net; both arms. Appendix only — BREAKS fair comparison.",
    },
    "diversemix": {
        "module": _diversemix,
        "class": "DiverseMixClassD",
        "eval_mode": "per_sample_finetune",
        "train_kind": "self",
        "score_type": "energy",
        "arms": ["head_only", "full_net"],
        "params": {"n_epochs": 10, "alpha": 2.0, "temperature": 0.5, "omega": 0.5,
                   "hidden_dim": 128, "lr": 1e-3, "batch_size": 64},
        "caption": "Faithful DiverseMix (Yao 2024): energy head trained on a REAL "
                   "hold-out aux corpus with score-adaptive mixup between ID and aux "
                   "outliers; score = -logsumexp (official orientation); both arms. "
                   "Appendix only — BREAKS fair comparison.",
    },
    # ----------------------------------------------------------------------
    # GROUP III — DIVERSIFY: from-scratch adversarial (GRL) representation
    # learning + an INVENTED OOD score (the paper defines none).
    # eval_mode "per_sample_selftrain": the detector trains its OWN 1-D extractor
    # from scratch on the ID train windows (NO shared-backbone fine-tune, NO aux
    # corpus). Per file the runner scores val/test under BOTH invented score
    # variants (energy primary, cosine-centroid secondary) -> per_sample_auroc.
    # The score-variant is carried in the "arm" column (energy / cosine_centroid).
    # ----------------------------------------------------------------------
    "diversify": {
        "module": _diversify,
        "class": "DiversifyClassD",
        "eval_mode": "per_sample_selftrain",
        "score_variants": ["energy", "cosine"],
        "params": {"latent_domain_num": 5, "epochs": 30, "alpha": 1.0,
                   "lr": 1e-3, "batch_size": 64, "feat_dim": 64,
                   "temperature": 1.0, "min_per_domain": 2},
        "caption": "From-scratch DIVERSIFY (Lu 2023, arXiv:2209.07027): minimal "
                   "DANN-style adversarial extractor for 1-D windows — a 1-D CNN "
                   "featurizer + pseudo-class head + latent-domain characterizer "
                   "(cosine k-means on L2-normalised features) + domain-adversarial "
                   "branch through a Gradient Reversal Layer that maximises latent- "
                   "domain diversity while enforcing class-invariance. Trains its OWN "
                   "extractor from scratch (NOT the shared frozen backbone). "
                   "N < latent_domain_num crash guarded (domain count clamped to the "
                   "data; adversarial branch skipped when < 2 domains). INVENTED score "
                   "(paper defines NONE): energy -logsumexp primary, cosine-centroid "
                   "secondary; higher = OOD. Appendix only — BREAKS fair comparison.",
    },
}

# Which registry entries belong to each group's sweep (by eval_mode).
GROUP2_MODES = ("batch_level", "ordered_per_window")
GROUP1_MODES = ("per_sample_finetune",)
GROUP3_MODES = ("per_sample_selftrain",)


def _load_detector(entry: Dict[str, Any]):
    """Return (class, module) for a Class-D detector from its in-package module."""
    module = entry["module"]
    return getattr(module, entry["class"]), module


# ===========================================================================
# Results output (separate appendix CSVs — never benchmark.csv)
# ===========================================================================

CSV_COLS = [
    "method", "dataset", "split", "category", "arm", "eval_mode",
    "metric", "auroc", "aupr", "fpr95", "n_eval", "drift_delay", "seed", "timestamp",
]


def write_results(rows: List[Dict[str, Any]], suffix: str = "results") -> str:
    """Append rows to results/class_d_<suffix>.csv (creating it if absent)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"class_d_{suffix}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLS})
    return path


# ===========================================================================
# Group-II sweep helpers
# ===========================================================================

CATEGORIES = ("DRIFT", "OOD", "STABLE")


def _split_window_stride(split: str) -> Tuple[int, int]:
    return (128, 64) if split.upper() == "M" else (64, 32)


def _choose_batch_size(stream_y: np.ndarray, base_B: int, tau: float) -> int:
    """Pick a monitoring batch size giving >=2 non-overlapping batches of BOTH
    classes. Prefers the configured B (32/16); shrinks only when the ordered
    stream is too short for it to yield a well-defined batch-level AUROC.
    """
    y = np.asarray(stream_y).astype(int)
    n = len(y)
    candidates = [int(base_B)] + [max(4, n // k) for k in (4, 6, 8, 10, 12)]
    seen = set()
    for B in candidates:
        if B < 2 or B > n or B in seen:
            continue
        seen.add(B)
        starts = range(0, n - B + 1, B)
        labels = [1 if y[s:s + B].mean() >= tau else 0 for s in starts]
        if len(labels) >= 2 and len(set(labels)) >= 2:
            return B
    return int(base_B)


def _n_batches(n: int, B: int) -> int:
    return (n - B) // B + 1 if (B >= 1 and n >= B) else 0


def _base_row(name: str, dataset_id: str, split: str, category: str,
              eval_mode: str) -> Dict[str, Any]:
    return {
        "method": name, "dataset": dataset_id, "split": split,
        "category": category, "arm": "none", "eval_mode": eval_mode,
        "metric": None, "auroc": None, "aupr": None, "fpr95": None,
        "n_eval": None, "drift_delay": None, "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _score_one_detector(name, cls, module, entry, split, category, dataset_id,
                        bb, id_windows, stream) -> Dict[str, Any]:
    """Fit + score a single Group-II detector on one file's ordered stream."""
    row = _base_row(name, dataset_id, split, category, entry["eval_mode"])
    pipeline.set_seed(SEED)
    det_bb = clone_backbone(bb)                    # never mutate the shared backbone
    stream_x, stream_y = stream["x"], stream["y"]

    if entry["eval_mode"] == "batch_level":
        base_B = entry.get("batch_size", {}).get(split.upper(), 32)
        B = _choose_batch_size(stream_y, base_B, entry.get("tau", 0.5))
        params = dict(entry["params"]); params["batch_size"] = B
        det = cls(det_bb, params)
        det.fit(id_windows)
        auroc = batch_level_auroc(stream_x, stream_y, det.score_batch,
                                  B, tau=entry.get("tau", 0.5))
        row.update(metric="batch_auroc", auroc=_round(auroc),
                   n_eval=_n_batches(len(stream_x), B))
        row["_batch_size"] = B
    else:  # ordered_per_window
        params = dict(entry["params"]); params["device"] = "cpu"; params["seed"] = SEED
        det = cls(det_bb, params)
        det.fit(id_windows)
        out = det.score_stream(stream_x)
        alarms = None
        if isinstance(out, tuple):
            scores, alarms = out
        else:
            scores = out
        scores = np.asarray(scores, dtype=np.float64).ravel()
        auroc = per_sample_auroc(stream_y, scores)
        aupr = fpr95 = None
        if len(np.unique(np.asarray(stream_y).astype(int))) >= 2 and np.isfinite(scores).all():
            aupr = _round(compute_aupr(np.asarray(stream_y).astype(int), scores))
            fpr95 = _round(compute_fpr95(np.asarray(stream_y).astype(int), scores))
        dd = None
        if alarms is not None and hasattr(module, "drift_delay"):
            dd = module.drift_delay(alarms, stream_y)
            dd = None if (dd is None or not np.isfinite(dd)) else float(dd)
        row.update(metric="per_sample_auroc", auroc=_round(auroc),
                   aupr=aupr, fpr95=fpr95, n_eval=int(len(scores)), drift_delay=dd)
    return row


def _round(v):
    return None if (v is None or not np.isfinite(v)) else round(float(v), 4)


def _process_file(path, split, category, dets, rows) -> None:
    """Load one eval file (ordered), train the shared backbone, score all Group-II
    detectors. Per-detector errors are isolated so one failure does not abort."""
    window, stride = _split_window_stride(split)
    dataset_id = f"TSB-{split.upper()}-" + os.path.splitext(os.path.basename(path))[0][:48]
    dataset = load_tsb(
        data_path=path, window_size=window, stride=stride, n_pseudo_classes=4,
        train_frac=0.70, seed=SEED, normalize="per_series",
        boundary_split=True, ordered_eval=True, dataset_name=dataset_id,
    )
    if "stream" not in dataset:
        raise RuntimeError("ordered stream missing from load_tsb output")
    stream = dataset["stream"]
    in_channels = int(dataset["train"]["x"].shape[1])
    bb, _head = pipeline.train_backbone(dataset, in_channels)
    id_windows = dataset["train"]["x"]

    for name, (cls, module, entry) in dets.items():
        try:
            row = _score_one_detector(name, cls, module, entry, split, category,
                                      dataset_id, bb, id_windows, stream)
            bs = row.pop("_batch_size", None)
            tag = row.get("auroc")
            extra = f" B={bs}" if bs is not None else ""
            dd = row.get("drift_delay")
            extra += f" delay={dd}" if dd is not None else ""
            print(f"  {name:16s} {entry['eval_mode']:18s} auroc={tag}{extra}", flush=True)
            rows.append(row)
            write_results([row], suffix="group2")
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  {name:16s} ERROR {exc.__class__.__name__}: {exc}", flush=True)
            print(traceback.format_exc()[-800:], flush=True)


# ===========================================================================
# Group-I sweep helpers (per_sample_finetune — OE family)
# ===========================================================================

def _draw_aux(split: str, in_channels: int, n_id: int, exclude_file: str) -> np.ndarray:
    """Draw a channel-matched hold-out aux corpus for one eval file.

    Size scales with the (tiny) ID set; AUX.get_aux_windows guarantees the exact
    channel count and falls back to synthetic outliers if no channel-matched
    hold-out file exists (per CLASS_D_DECISIONS §2).
    """
    n_aux = int(min(512, max(64, 4 * max(n_id, 1))))
    return AUX.get_aux_windows(
        split, n_aux, in_channels, seed=SEED, normalize="per_series",
        exclude_file=exclude_file,
    )


def _finetune_and_score(name, cls, module, entry, split, category, dataset_id,
                        bb, head, dataset, arm, aux_x) -> Dict[str, Any]:
    """Fine-tune ONE Group-I detector for ONE arm, then score val/test with energy.

    The shared (bb, head) are never mutated: OE/DivOE fine-tune a deep copy via the
    runner helpers; DiverseMix trains on a freshly cloned backbone. Returns one row.
    """
    row = _base_row(name, dataset_id, split, category, entry["eval_mode"])
    row["arm"] = arm
    hp = dict(entry.get("params", {}))
    train_kind = entry.get("train_kind", "self")

    id_x = dataset["train"]["x"]
    id_y = dataset["train"]["y"]
    val_x = dataset["val"]["x"]
    test_x = dataset["test"]["x"]
    test_y = np.asarray(dataset["test"]["y"]).astype(int)

    pipeline.set_seed(SEED)
    if train_kind == "oe":
        id_loader = make_loader(id_x, id_y, batch_size=hp.get("batch_size", 64), shuffle=True)
        aux_loader = make_loader(aux_x, None, batch_size=hp.get("oe_batch_size", 64), shuffle=True)
        bb2, head2 = finetune(
            bb, head, id_loader, aux_loader, arm,
            epochs=hp.get("epochs", 10), lr=hp.get("lr", 1e-3),
            oe_weight=hp.get("oe_weight", 0.5), seed=SEED,
        )
        det = cls(bb2, {"classifier": head2, "device": "cpu",
                        "temperature": hp.get("temperature", 1.0)})
        det.fit()
    elif train_kind == "divoe":
        id_loader = make_loader(id_x, id_y, batch_size=hp.get("batch_size", 64), shuffle=True)
        aux_loader = make_loader(aux_x, None, batch_size=hp.get("oe_batch_size", 64), shuffle=True)
        bb2, head2 = finetune_divoe(
            bb, head, id_loader, aux_loader, arm, pgd_fn=module.extrapolate_pgd,
            epochs=hp.get("epochs", 10), lr=hp.get("lr", 1e-3),
            oe_weight=hp.get("oe_weight", 0.5), epsilon=hp.get("epsilon", 0.1),
            num_steps=hp.get("num_steps", 5), rel_step_size=hp.get("rel_step_size", 0.25),
            extrapolation_ratio=hp.get("extrapolation_ratio", 0.5),
            extrapolation_score=hp.get("extrapolation_score", "MSP"), seed=SEED,
        )
        det = cls(bb2, {"classifier": head2, "device": "cpu",
                        "temperature": hp.get("temperature", 1.0)})
        det.fit()
    else:  # "self" — the detector trains itself on a cloned backbone (diversemix)
        det_bb = clone_backbone(bb)
        params = dict(hp)
        params.update(classifier=_clone_head(head), arm=arm, aux_x=aux_x,
                      device="cpu", seed=SEED)
        det = cls(det_bb, params)
        det.fit(id_x, id_y)

    s_test = np.asarray(det.score(test_x), dtype=np.float64).ravel()
    _ = np.asarray(det.score(val_x), dtype=np.float64).ravel()  # val path exercised
    auroc = per_sample_auroc(test_y, s_test)
    aupr = fpr95 = None
    if len(np.unique(test_y)) >= 2 and np.isfinite(s_test).all():
        aupr = _round(compute_aupr(test_y, s_test))
        fpr95 = _round(compute_fpr95(test_y, s_test))
    row.update(metric="per_sample_auroc", auroc=_round(auroc), aupr=aupr,
               fpr95=fpr95, n_eval=int(len(s_test)))
    row["_score_mean"] = float(np.mean(s_test)) if len(s_test) else float("nan")
    return row


def _process_file_group1(path, split, category, dets1, rows) -> None:
    """Load one eval file (STANDARD shuffled — not ordered), train the shared
    backbone, draw a channel-matched aux corpus, then fine-tune+score every Group-I
    detector for BOTH arms. Per (detector, arm) errors are isolated."""
    window, stride = _split_window_stride(split)
    dataset_id = f"TSB-{split.upper()}-" + os.path.splitext(os.path.basename(path))[0][:48]
    dataset = load_tsb(
        data_path=path, window_size=window, stride=stride, n_pseudo_classes=4,
        train_frac=0.70, seed=SEED, normalize="per_series",
        boundary_split=True, ordered_eval=False, dataset_name=dataset_id,
    )
    in_channels = int(dataset["train"]["x"].shape[1])
    bb, head = pipeline.train_backbone(dataset, in_channels)
    n_id = int(len(dataset["train"]["x"]))
    aux_x = _draw_aux(split, in_channels, n_id, exclude_file=path)

    for name, (cls, module, entry) in dets1.items():
        arm_scores: Dict[str, float] = {}
        for arm in entry.get("arms", ["head_only", "full_net"]):
            try:
                row = _finetune_and_score(name, cls, module, entry, split, category,
                                          dataset_id, bb, head, dataset, arm, aux_x)
                sm = row.pop("_score_mean", None)
                if sm is not None and np.isfinite(sm):
                    arm_scores[arm] = sm
                print(f"  {name:16s} {arm:10s} auroc={row.get('auroc')} "
                      f"aupr={row.get('aupr')} score_mean={None if sm is None else round(sm, 4)}",
                      flush=True)
                rows.append(row)
                write_results([row], suffix="group1")
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"  {name:16s} {arm:10s} ERROR {exc.__class__.__name__}: {exc}", flush=True)
                print(traceback.format_exc()[-800:], flush=True)
        if len(arm_scores) == 2:
            a, b = arm_scores.get("head_only"), arm_scores.get("full_net")
            same = (a is not None and b is not None and abs(a - b) < 1e-9)
            print(f"  {name:16s} arms differ: {not same} "
                  f"(head_only mean={None if a is None else round(a, 4)}, "
                  f"full_net mean={None if b is None else round(b, 4)})", flush=True)


# ===========================================================================
# Drivers
# ===========================================================================

def run_group2(splits, n_per_cell) -> None:
    """The ordered / window-level (Group-II) sweep."""
    dets = {}
    for name, entry in CLASS_D_REGISTRY.items():
        if entry["eval_mode"] not in GROUP2_MODES:
            continue
        cls, module = _load_detector(entry)
        dets[name] = (cls, module, entry)

    print(f"Class-D GROUP II sweep: splits={splits} n_per_cell={n_per_cell} "
          f"detectors={list(dets)}", flush=True)

    rows: List[Dict[str, Any]] = []
    for split in splits:
        for cat in CATEGORIES:
            files = AUX.get_eval_files(split, cat)[:n_per_cell]
            for f in files:
                print(f"\n===== {split}/{cat}: {os.path.basename(f)[:70]} =====", flush=True)
                try:
                    _process_file(f, split, cat, dets, rows)
                except Exception as exc:  # noqa: BLE001
                    print(f"  SKIP FILE ({exc.__class__.__name__}: {exc})", flush=True)
                    continue

    finite = [r for r in rows if r.get("auroc") is not None]
    print(f"\nDONE (GROUP II): {len(finite)}/{len(rows)} detector runs produced a "
          f"finite AUROC -> results/class_d_group2.csv", flush=True)


def run_group1(splits, n_per_cell) -> None:
    """The auxiliary-outlier + fine-tuning (Group-I / OE family) sweep."""
    dets1 = {}
    for name, entry in CLASS_D_REGISTRY.items():
        if entry["eval_mode"] not in GROUP1_MODES:
            continue
        cls, module = _load_detector(entry)
        dets1[name] = (cls, module, entry)

    print(f"Class-D GROUP I sweep: splits={splits} n_per_cell={n_per_cell} "
          f"detectors={list(dets1)}", flush=True)

    rows: List[Dict[str, Any]] = []
    for split in splits:
        for cat in CATEGORIES:
            files = AUX.get_eval_files(split, cat)[:n_per_cell]
            for f in files:
                print(f"\n===== {split}/{cat}: {os.path.basename(f)[:70]} =====", flush=True)
                try:
                    _process_file_group1(f, split, cat, dets1, rows)
                except Exception as exc:  # noqa: BLE001
                    print(f"  SKIP FILE ({exc.__class__.__name__}: {exc})", flush=True)
                    continue

    finite = [r for r in rows if r.get("auroc") is not None]
    print(f"\nDONE (GROUP I): {len(finite)}/{len(rows)} (method, arm) runs produced a "
          f"finite AUROC -> results/class_d_group1.csv", flush=True)


# ===========================================================================
# Group-III sweep helpers (per_sample_selftrain — DIVERSIFY, from-scratch GRL)
# ===========================================================================

def _selftrain_and_score(name, cls, entry, split, category, dataset_id,
                         dataset) -> List[Dict[str, Any]]:
    """Train ONE Group-III detector from scratch on the ID windows, then score
    val/test under EACH invented score variant. Returns one row per variant.

    The detector trains its OWN extractor (no shared backbone, no aux corpus).
    The shared frozen backbone is not even loaded here.
    """
    hp = dict(entry.get("params", {}))
    hp["device"] = "cpu"; hp["seed"] = SEED
    variants = entry.get("score_variants", ["energy"])

    id_x = dataset["train"]["x"]
    id_y = dataset["train"]["y"]
    val_x = dataset["val"]["x"]
    test_x = dataset["test"]["x"]
    test_y = np.asarray(dataset["test"]["y"]).astype(int)

    pipeline.set_seed(SEED)
    det = cls(hp)
    try:
        det.fit(id_x, id_y)
    except TypeError:
        det.fit(id_x)

    eff_dom = getattr(det, "eff_domain_num", None)
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        row = _base_row(name, dataset_id, split, category, entry["eval_mode"])
        # the score variant is carried in the "arm" column (no CSV_COLS change)
        row["arm"] = "cosine_centroid" if variant == "cosine" else variant
        s_test = np.asarray(det.score(test_x, variant=variant), dtype=np.float64).ravel()
        _ = np.asarray(det.score(val_x, variant=variant), dtype=np.float64).ravel()
        auroc = per_sample_auroc(test_y, s_test)
        aupr = fpr95 = None
        if len(np.unique(test_y)) >= 2 and np.isfinite(s_test).all():
            aupr = _round(compute_aupr(test_y, s_test))
            fpr95 = _round(compute_fpr95(test_y, s_test))
        row.update(metric="per_sample_auroc", auroc=_round(auroc), aupr=aupr,
                   fpr95=fpr95, n_eval=int(len(s_test)))
        row["_eff_dom"] = eff_dom
        rows.append(row)
    return rows


def _process_file_group3(path, split, category, dets3, rows) -> None:
    """Load one eval file (STANDARD shuffled — not ordered), then for every
    Group-III detector self-train + score BOTH invented variants. Per-detector
    errors are isolated so one failure does not abort the file."""
    window, stride = _split_window_stride(split)
    dataset_id = f"TSB-{split.upper()}-" + os.path.splitext(os.path.basename(path))[0][:48]
    dataset = load_tsb(
        data_path=path, window_size=window, stride=stride, n_pseudo_classes=4,
        train_frac=0.70, seed=SEED, normalize="per_series",
        boundary_split=True, ordered_eval=False, dataset_name=dataset_id,
    )

    for name, (cls, module, entry) in dets3.items():
        try:
            file_rows = _selftrain_and_score(name, cls, entry, split, category,
                                             dataset_id, dataset)
            for row in file_rows:
                ed = row.pop("_eff_dom", None)
                print(f"  {name:16s} {row.get('arm'):15s} auroc={row.get('auroc')} "
                      f"aupr={row.get('aupr')} eff_domains={ed} n={row.get('n_eval')}",
                      flush=True)
                rows.append(row)
                write_results([row], suffix="group3")
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  {name:16s} ERROR {exc.__class__.__name__}: {exc}", flush=True)
            print(traceback.format_exc()[-800:], flush=True)


def run_group3(splits, n_per_cell) -> None:
    """The DIVERSIFY (from-scratch adversarial representation) Group-III sweep."""
    dets3 = {}
    for name, entry in CLASS_D_REGISTRY.items():
        if entry["eval_mode"] not in GROUP3_MODES:
            continue
        cls, module = _load_detector(entry)
        dets3[name] = (cls, module, entry)

    print(f"Class-D GROUP III sweep: splits={splits} n_per_cell={n_per_cell} "
          f"detectors={list(dets3)}", flush=True)

    rows: List[Dict[str, Any]] = []
    for split in splits:
        for cat in CATEGORIES:
            files = AUX.get_eval_files(split, cat)[:n_per_cell]
            for f in files:
                print(f"\n===== {split}/{cat}: {os.path.basename(f)[:70]} =====", flush=True)
                try:
                    _process_file_group3(f, split, cat, dets3, rows)
                except Exception as exc:  # noqa: BLE001
                    print(f"  SKIP FILE ({exc.__class__.__name__}: {exc})", flush=True)
                    continue

    finite = [r for r in rows if r.get("auroc") is not None]
    print(f"\nDONE (GROUP III): {len(finite)}/{len(rows)} (method, score-variant) runs "
          f"produced a finite AUROC -> results/class_d_group3.csv", flush=True)


# ===========================================================================
# Public entrypoint (dispatched from runners/run.py --group class_d)
# ===========================================================================

def run_class_d(splits, n_per_cell, group: str = "all") -> None:
    """Run the requested Class-D group sweep(s).

    splits     : list like ["U"] / ["U", "M"]
    n_per_cell : eval files per (split, category) cell
    group      : "all" | "1"/"i"/"group1" | "2"/"ii"/"group2" | "3"/"iii"/"group3"
    """
    # Ensure the eval/aux partition manifest exists (no-leakage guarantee).
    AUX.build_manifest()

    g = str(group).strip().lower()
    if g in ("2", "ii", "group2", "all"):
        run_group2(splits, n_per_cell)
    if g in ("1", "i", "group1", "all"):
        run_group1(splits, n_per_cell)
    if g in ("3", "iii", "group3", "all"):
        run_group3(splits, n_per_cell)


def main() -> None:
    splits = [s.strip().upper() for s in os.environ.get("TSB_SPLITS", "U").split(",") if s.strip()]
    n_per_cell = int(os.environ.get("TSB_N_PER_CELL", "3"))
    group = os.environ.get("TSB_GROUP", "all").strip().lower()
    run_class_d(splits, n_per_cell, group)


if __name__ == "__main__":
    main()
