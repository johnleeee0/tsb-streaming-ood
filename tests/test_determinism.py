"""Determinism test: same seed -> bit-identical scores.

Author: Stylianos Giannoulis - AUTH MSc Data and Web Science - Supervisor: John Paparrizos

The production ``runners.pipeline.run_one`` calls ``set_seed(42)`` before each
detector's fit/score, so a detector run twice on the same backbone and data must
produce identical score arrays (this is what makes results independent of method
order and reproducible). We verify this through the real run_one path - for a
purely deterministic distance detector (mahalanobis, dfm_pca) and for a detector
that trains a small network (m2n2), so both code paths are covered.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models.detectors  # noqa: E402,F401
from core.registry import OOD_REGISTRY  # noqa: E402
from runners import pipeline  # noqa: E402
from runners.run import METHOD_PARAMS  # noqa: E402

C, T = 3, 32


def _make_split(rng, n, n_pseudo=4, binary=False):
    x = rng.standard_normal((n, C, T)).astype(np.float32)
    if binary:
        y = (rng.random(n) < 0.4).astype(np.int64)
        x[y == 1] += 2.5
    else:
        y = rng.integers(0, n_pseudo, size=n).astype(np.int64)
    return {"x": x, "y": y}


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(42)
    dataset = {
        "train": _make_split(rng, 64, n_pseudo=4),
        "val":   _make_split(rng, 40, binary=True),
        "test":  _make_split(rng, 40, binary=True),
        "metadata": {"dataset_name": "DET"},
    }
    bb, head = pipeline.train_backbone(dataset, in_channels=C, epochs=1)
    return dataset, bb, head


def _run_scores(name, dataset, bb, head):
    cls = OOD_REGISTRY._items[name]
    params = METHOD_PARAMS.get(name, {})
    out_dir = tempfile.mkdtemp(prefix=f"det_{name}_")
    r = pipeline.run_one((name, cls, params), dataset, bb, head, out_dir, epochs=1)
    assert r.get("status") == "COMPLETE", f"{name}: {r.get('error')}"
    return np.load(os.path.join(out_dir, "scores.npy"))


@pytest.mark.parametrize("name", ["mahalanobis", "dfm_pca", "m2n2"])
def test_same_seed_identical_scores(name, fixture):
    dataset, bb, head = fixture
    s1 = _run_scores(name, dataset, bb, head)
    s2 = _run_scores(name, dataset, bb, head)
    assert s1.shape == s2.shape
    assert np.array_equal(s1, s2), f"{name}: scores differ across identical runs"
