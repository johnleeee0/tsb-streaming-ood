"""Smoke test: every one of the 24 detectors wires up, fits, and scores end-to-end.

Author: Stylianos Giannoulis - AUTH MSc Data and Web Science - Supervisor: John Paparrizos

Trains a tiny ResNet backbone (1 epoch) on a small SYNTHETIC dataset, then:

  * the 17 MAIN detectors are pushed through the real ``runners.pipeline.run_one``
    path (the exact production code) and must reach status COMPLETE with a finite
    AUROC; and
  * the 7 CLASS-D appendix detectors are instantiated + fit + scored through their
    native interfaces (batch_level / ordered_per_window / per_sample_finetune /
    per_sample_selftrain), and must return finite scores.

No TSB data and no GPU are required - this is the fast "does the whole codebase
still run" check used by CI and after every edit.

Run directly (exit 0 = all 24 pass)::

    C:\\THESIS\\.venv\\Scripts\\python.exe tests/test_smoke.py

or under pytest::

    C:\\THESIS\\.venv\\Scripts\\python.exe -m pytest tests/test_smoke.py -q
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile

import numpy as np
import pytest
from torch import nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models.detectors  # noqa: E402,F401  (populates OOD_REGISTRY via @register_ood)
from core.registry import OOD_REGISTRY  # noqa: E402
from models.backbones.resnet import ResNetBackbone  # noqa: E402
from runners import pipeline  # noqa: E402
from runners.run import MAIN_ORDER, METHOD_PARAMS  # noqa: E402

# Class-D in-package faithful builds.
from models.detectors.class_d import ae_adwin_lstm as _ae_adwin_lstm  # noqa: E402
from models.detectors.class_d import diversemix as _diversemix  # noqa: E402
from models.detectors.class_d import diversify as _diversify  # noqa: E402
from models.detectors.class_d import divoe as _divoe  # noqa: E402
from models.detectors.class_d import driftlens as _driftlens  # noqa: E402
from models.detectors.class_d import outlier_exposure as _outlier_exposure  # noqa: E402
from models.detectors.class_d import tdivdm as _tdivdm  # noqa: E402

C, T = 3, 32
CLASS_D_NAMES = [
    "driftlens", "ae_adwin_lstm", "tdivdm",
    "outlier_exposure", "divoe", "diversemix", "diversify",
]


# ---------------------------------------------------------------------------
# Shared synthetic fixtures (built once)
# ---------------------------------------------------------------------------

def _make_split(rng, n, n_pseudo=4, binary=False):
    x = rng.standard_normal((n, C, T)).astype(np.float32)
    if binary:
        y = (rng.random(n) < 0.4).astype(np.int64)
        x[y == 1] += 2.5  # shift OOD windows so AUROC is well-defined
    else:
        y = rng.integers(0, n_pseudo, size=n).astype(np.int64)
    return {"x": x, "y": y}


@pytest.fixture(scope="session")
def synthetic():
    rng = np.random.default_rng(42)
    dataset = {
        "train": _make_split(rng, 64, n_pseudo=4),
        "val":   _make_split(rng, 40, binary=True),
        "test":  _make_split(rng, 40, binary=True),
        "metadata": {"dataset_name": "SMOKE"},
    }
    aux_x = rng.standard_normal((48, C, T)).astype(np.float32) + 1.0
    return dataset, aux_x


@pytest.fixture(scope="session")
def backbone(synthetic):
    dataset, _aux = synthetic
    bb, head = pipeline.train_backbone(dataset, in_channels=C, epochs=1)
    return bb, head


# ---------------------------------------------------------------------------
# Main set (17) - through the real production run_one path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", MAIN_ORDER)
def test_main_detector_runs(name, synthetic, backbone):
    dataset, _aux = synthetic
    bb, head = backbone
    cls = OOD_REGISTRY._items[name]
    params = METHOD_PARAMS.get(name, {})
    scratch = tempfile.mkdtemp(prefix=f"smoke_{name}_")
    r = pipeline.run_one((name, cls, params), dataset, bb, head,
                         os.path.join(scratch, name), epochs=1)
    assert r.get("status") == "COMPLETE", f"{name}: {r.get('error')}"
    assert np.isfinite(r.get("auroc")), f"{name}: non-finite auroc"


def test_main_registry_has_17():
    assert len(MAIN_ORDER) == 17
    for name in MAIN_ORDER:
        assert name in OOD_REGISTRY._items, f"{name} not registered"


# ---------------------------------------------------------------------------
# Class-D set (7) - through their native interfaces
# ---------------------------------------------------------------------------

def _fresh_bb_head(backbone):
    bb, head = backbone
    return copy.deepcopy(bb), copy.deepcopy(head)


def test_class_d_driftlens(synthetic, backbone):
    dataset, _aux = synthetic
    bb, _head = _fresh_bb_head(backbone)
    det = _driftlens.DriftLensClassD(bb, {"n_pc": 8, "batch_size": 16})
    det.fit(dataset["train"]["x"])
    s = det.score_batch(dataset["test"]["x"][:16])
    assert np.isfinite(s)


def test_class_d_ae_adwin_lstm(synthetic, backbone):
    dataset, _aux = synthetic
    bb, _head = _fresh_bb_head(backbone)
    det = _ae_adwin_lstm.AEADWINLSTMClassD(bb, {
        "n_epochs_ae": 1, "n_epochs_lstm": 1, "hidden_dim": 16, "lstm_layers": 1,
        "seq_len": 5, "adwin_delta": 0.002, "batch_size": 16, "lr": 1e-3,
        "incremental_update": True, "device": "cpu", "seed": 42,
    })
    det.fit(dataset["train"]["x"])
    out = det.score_stream(dataset["test"]["x"])
    scores = out[0] if isinstance(out, tuple) else out
    scores = np.asarray(scores, dtype=np.float64).ravel()
    assert len(scores) == len(dataset["test"]["x"])
    assert np.isfinite(scores).all()


def test_class_d_tdivdm(synthetic, backbone):
    dataset, _aux = synthetic
    bb, _head = _fresh_bb_head(backbone)
    det = _tdivdm.TDIVDMClassD(bb, {"scales": [5, 10], "bandwidth": "scott"})
    det.fit(dataset["train"]["x"])
    scores = np.asarray(det.score_stream(dataset["test"]["x"]), dtype=np.float64).ravel()
    assert len(scores) == len(dataset["test"]["x"])
    assert np.isfinite(scores).all()


def test_class_d_outlier_exposure(synthetic, backbone):
    dataset, _aux = synthetic
    bb, head = _fresh_bb_head(backbone)
    det = _outlier_exposure.OutlierExposureClassD(
        bb, {"classifier": head, "device": "cpu", "temperature": 1.0})
    det.fit()
    scores = np.asarray(det.score(dataset["test"]["x"]), dtype=np.float64).ravel()
    assert len(scores) == len(dataset["test"]["x"])
    assert np.isfinite(scores).all()


def test_class_d_divoe(synthetic, backbone):
    dataset, _aux = synthetic
    bb, head = _fresh_bb_head(backbone)
    det = _divoe.DivOEClassD(
        bb, {"classifier": head, "device": "cpu", "temperature": 1.0})
    det.fit()
    scores = np.asarray(det.score(dataset["test"]["x"]), dtype=np.float64).ravel()
    assert len(scores) == len(dataset["test"]["x"])
    assert np.isfinite(scores).all()
    # extrapolate_pgd (the DivOE synthesis primitive) must be importable/callable.
    assert callable(getattr(_divoe, "extrapolate_pgd"))


def test_class_d_diversemix(synthetic, backbone):
    dataset, aux_x = synthetic
    bb, head = _fresh_bb_head(backbone)
    det = _diversemix.DiverseMixClassD(bb, {
        "classifier": head, "arm": "head_only", "aux_x": aux_x,
        "device": "cpu", "seed": 42, "n_epochs": 1, "hidden_dim": 32,
        "batch_size": 16,
    })
    det.fit(dataset["train"]["x"], dataset["train"]["y"])
    scores = np.asarray(det.score(dataset["test"]["x"]), dtype=np.float64).ravel()
    assert len(scores) == len(dataset["test"]["x"])
    assert np.isfinite(scores).all()


def test_class_d_diversify(synthetic):
    dataset, _aux = synthetic
    det = _diversify.DiversifyClassD({
        "latent_domain_num": 3, "epochs": 2, "alpha": 1.0, "lr": 1e-3,
        "batch_size": 32, "feat_dim": 32, "temperature": 1.0,
        "min_per_domain": 2, "device": "cpu", "seed": 42,
    })
    det.fit(dataset["train"]["x"], dataset["train"]["y"])
    for variant in ("energy", "cosine"):
        scores = np.asarray(det.score(dataset["test"]["x"], variant=variant),
                            dtype=np.float64).ravel()
        assert len(scores) == len(dataset["test"]["x"])
        assert np.isfinite(scores).all()


def test_class_d_has_7():
    assert len(CLASS_D_NAMES) == 7


# ---------------------------------------------------------------------------
# Standalone runner (exit 0 = all 24 pass) - mirrors the CI invocation
# ---------------------------------------------------------------------------

def main() -> int:
    rc = pytest.main([os.path.abspath(__file__), "-q"])
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
