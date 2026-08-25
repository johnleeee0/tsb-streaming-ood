"""Training + evaluation pipeline (ported from experiments/run_experiments.py).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Trains a 1-D ResNet backbone (cross-entropy on temporal pseudo-classes) on the
in-distribution windows of each TSB-StreamingAD file, then evaluates the OOD detectors
on the held-out balanced test split. All metrics are computed with core/metrics.py.

Rewired to the THESIS_FINAL package scheme (core/, data/, models/). Behaviour is
identical to the original run_experiments.py + tsb_benchmark.py: seed 42, CPU,
per-method set_seed() before fit/score (so results are independent of method order).
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime

import numpy as np
import torch
from scipy.stats import kurtosis, skew
from torch import nn

from core.metrics import compute_aupr, compute_auroc, compute_fpr95
from data.tsb_loader import load_tsb
from models.backbones.resnet import ResNetBackbone

SEED = 42
BACKBONE_EPOCHS = 40  # reduced from 100 (full-scale) for CPU tractability

# Domains present in the TSB-StreamingAD file names (drives per-series normalisation).
DOMAINS = ["Medical", "HumanActivity", "WebService", "Facility", "Synthetic", "Environment", "Sensor"]
MAX_ROWS = 150000  # cap very long series (files whose first anomaly is beyond this are skipped)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# per-file dataset prep (ported from tsb_benchmark.parse_meta + load_tsb call)
# ---------------------------------------------------------------------------

def parse_meta(path: str, split: str) -> dict:
    base = os.path.basename(path)
    category = base.split("_")[0]  # DRIFT / OOD / STABLE
    domain = next((d for d in DOMAINS if d in base), "Unknown")
    normalize = "per_series" if domain in ("Medical", "HumanActivity") else "global"
    window, stride = (128, 64) if split == "M" else (64, 32)
    return dict(category=category, domain=domain, normalize=normalize, window=window, stride=stride)


def prepare_dataset(path: str, split: str, ds_id: str, seed: int = SEED, max_rows: int = MAX_ROWS):
    """Load a TSB file into an OOD-ready dataset dict (raises on unusable files)."""
    meta = parse_meta(path, split)
    dataset = load_tsb(
        data_path=path, window_size=meta["window"], stride=meta["stride"],
        n_pseudo_classes=4, train_frac=0.70, max_rows=max_rows, seed=seed,
        normalize=meta["normalize"], boundary_split=True, dataset_name=ds_id,
    )
    return dataset, meta


# ---------------------------------------------------------------------------
# backbone training (identical to run_experiments.train_backbone)
# ---------------------------------------------------------------------------

def train_backbone(dataset, in_channels: int, epochs: int = BACKBONE_EPOCHS):
    set_seed()
    bb = ResNetBackbone(input_dim=in_channels, base_channels=64, embedding_dim=128, device="cpu")
    x = torch.from_numpy(dataset["train"]["x"]).float()
    y = torch.from_numpy(dataset["train"]["y"]).long()
    n_classes = int(len(np.unique(dataset["train"]["y"])))
    head = nn.Linear(128, max(2, n_classes))
    opt = torch.optim.Adam(list(bb.model.parameters()) + list(head.parameters()), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    bb.model.train(); head.train()
    bs = 32
    idx = np.arange(len(x))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            opt.zero_grad()
            loss = lf(head(bb.model(x[b])), y[b])
            loss.backward(); opt.step()
    bb.model.eval(); head.eval()
    return bb, head


# ---------------------------------------------------------------------------
# detection accuracy (identical to run_experiments.detection_accuracy)
# ---------------------------------------------------------------------------

def detection_accuracy(y_val, s_val, y_test, s_test) -> float:
    """Threshold chosen on val (max F1), applied to test."""
    best_thr, best_f1 = 0.0, -1.0
    cand = np.unique(s_val)
    for thr in cand:
        pred = (s_val >= thr).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    pred_t = (s_test >= best_thr).astype(int)
    return float((pred_t == y_test).mean())


# ---------------------------------------------------------------------------
# single (method, dataset) run (identical to run_experiments.run_one)
# ---------------------------------------------------------------------------

def run_one(method_spec, dataset, bb, head, out_dir, epochs: int = BACKBONE_EPOCHS):
    name, cls, params = method_spec
    config = dict(params); config["classifier"] = head; config["device"] = "cpu"
    res = {"method": name, "dataset": dataset["metadata"]["dataset_name"], "seed": SEED,
           "status": "FAILED", "timestamp": datetime.now().isoformat(timespec="seconds")}
    try:
        set_seed()
        det = cls(model=bb.model, config=config)
        try:
            det.fit(dataset["train"]["x"], dataset["train"]["y"])
        except TypeError:
            det.fit(dataset["train"]["x"])

        s_val = np.asarray(det.score(dataset["val"]["x"])).ravel()
        t0 = time.perf_counter()
        s_test = np.asarray(det.score(dataset["test"]["x"])).ravel()
        infer_ms = (time.perf_counter() - t0) * 1000.0 / max(len(s_test), 1)

        y_val = dataset["val"]["y"].astype(int)
        y_test = dataset["test"]["y"].astype(int)
        if not np.isfinite(s_test).all() or len(np.unique(y_test)) < 2:
            raise ValueError("non-finite scores or single-class test set")

        auroc = compute_auroc(y_test, s_test)
        aupr = compute_aupr(y_test, s_test)
        fpr95 = compute_fpr95(y_test, s_test)
        det_acc = detection_accuracy(y_val, s_val, y_test, s_test)

        id_s = s_test[y_test == 0]; ood_s = s_test[y_test == 1]
        res.update({
            "status": "COMPLETE", "auroc": round(auroc, 4), "aupr": round(aupr, 4),
            "fpr95": round(fpr95, 4), "det_acc": round(det_acc, 4),
            "inference_ms": round(infer_ms, 5),
            "extended_metrics": {
                "indist_score_mean": float(np.mean(id_s)), "indist_score_std": float(np.std(id_s)),
                "indist_score_skew": float(skew(id_s)) if len(id_s) > 2 else 0.0,
                "indist_score_kurt": float(kurtosis(id_s)) if len(id_s) > 2 else 0.0,
                "ood_score_mean": float(np.mean(ood_s)), "ood_score_std": float(np.std(ood_s)),
                "ood_score_skew": float(skew(ood_s)) if len(ood_s) > 2 else 0.0,
                "ood_score_kurt": float(kurtosis(ood_s)) if len(ood_s) > 2 else 0.0,
            },
            "n_test": int(len(y_test)), "backbone_epochs": epochs,
            "method_params": params,
        })
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "scores.npy"), s_test)
        np.save(os.path.join(out_dir, "labels.npy"), y_test)
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{exc.__class__.__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-1200:]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res
