"""Unit tests for data.tsb_loader - the ordered_eval flag contract.

Author: Stylianos Giannoulis - AUTH MSc Data and Web Science - Supervisor: John Paparrizos

Confirms:
  * the DEFAULT behaviour (ordered_eval=False) is unchanged - no ``stream`` key,
    metadata records ordered_eval=False, and balanced binary val/test are produced;
  * ordered_eval=True adds a temporally-ORDERED ``stream`` whose window start-times
    are monotonically non-decreasing (order preserved, no shuffle/balancing).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from data.tsb_loader import load_tsb


@pytest.fixture(scope="module")
def tsb_csv():
    """Write a small synthetic univariate TSB-style CSV (Data, Label)."""
    rng = np.random.default_rng(0)
    T = 6000
    t = np.arange(T)
    data = np.sin(t / 25.0) + 0.05 * rng.standard_normal(T)
    label = np.zeros(T, dtype=int)
    # a contiguous late anomaly block (keeps a large clean Source-1 prefix)
    data[3800:4200] += 4.0
    label[3800:4200] = 1
    path = os.path.join(tempfile.mkdtemp(prefix="tsb_csv_"), "OOD_Synthetic_0001.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("Data,Label\n")
        for d, l in zip(data, label):
            fh.write(f"{d:.6f},{l}\n")
    return path


def test_default_has_no_stream(tsb_csv):
    ds = load_tsb(tsb_csv, window_size=64, stride=32, seed=42,
                  normalize="global", dataset_name="T-DEFAULT")
    assert "stream" not in ds
    assert ds["metadata"]["ordered_eval"] is False
    # shapes: (N, C, T) windows, matching labels
    for split in ("train", "val", "test"):
        x, y = ds[split]["x"], ds[split]["y"]
        assert x.ndim == 3 and x.shape[2] == 64
        assert len(x) == len(y) and len(x) > 0
    # val/test are binary 0/1
    assert set(np.unique(ds["val"]["y"])).issubset({0, 1})
    assert set(np.unique(ds["test"]["y"])).issubset({0, 1})


def test_ordered_adds_stream_in_temporal_order(tsb_csv):
    ds = load_tsb(tsb_csv, window_size=64, stride=32, seed=42,
                  normalize="global", ordered_eval=True, dataset_name="T-ORDERED")
    assert "stream" in ds
    assert ds["metadata"]["ordered_eval"] is True
    stream = ds["stream"]
    for k in ("x", "y", "t"):
        assert k in stream
    t = np.asarray(stream["t"])
    # window start-times must be non-decreasing == original temporal order preserved
    assert np.all(np.diff(t) >= 0), "ordered stream is not in temporal order"
    assert len(stream["x"]) == len(stream["y"]) == len(t)
    # ordered stream must carry BOTH classes (it is the full held-out eval pool)
    assert set(np.unique(stream["y"])) == {0, 1}


def test_default_and_ordered_share_the_same_train(tsb_csv):
    """The ordered flag only changes the eval construction, not training data."""
    a = load_tsb(tsb_csv, window_size=64, stride=32, seed=42, normalize="global",
                 dataset_name="A")
    b = load_tsb(tsb_csv, window_size=64, stride=32, seed=42, normalize="global",
                 ordered_eval=True, dataset_name="B")
    assert a["train"]["x"].shape == b["train"]["x"].shape
    assert np.allclose(a["train"]["x"], b["train"]["x"])
    assert np.array_equal(a["train"]["y"], b["train"]["y"])
