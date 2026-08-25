"""Unit tests for core.metrics on known inputs.

Author: Stylianos Giannoulis - AUTH MSc Data and Web Science - Supervisor: John Paparrizos
"""
from __future__ import annotations

import numpy as np
import pytest

from core.metrics import (
    compute_aupr,
    compute_auroc,
    compute_fpr95,
    compute_id_accuracy,
    split_scores,
)


def test_auroc_perfect_separation():
    # OOD (label 1) strictly above ID (label 0) -> AUROC 1.0
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_auroc(y, s) == pytest.approx(1.0)


def test_auroc_inverted_is_zero():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert compute_auroc(y, s) == pytest.approx(0.0)


def test_auroc_chance_is_half():
    # Perfectly interleaved / tied ranks -> 0.5
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_auroc(y, s) == pytest.approx(0.5)


def test_auroc_known_value():
    # 2 ID, 2 OOD; one OOD ranks below one ID -> 3/4 concordant pairs.
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.4, 0.9])
    assert compute_auroc(y, s) == pytest.approx(0.75)


def test_aupr_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_aupr(y, s) == pytest.approx(1.0)


def test_fpr95_perfect_separation_is_zero():
    # Clean separation: at 95% TPR the FPR is 0.
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    assert compute_fpr95(y, s) == pytest.approx(0.0, abs=1e-9)


def test_fpr95_all_overlap_is_one():
    # ID and OOD identical -> to reach 95% TPR you flag ~everything -> FPR ~ 1.
    y = np.array([0] * 20 + [1] * 20)
    s = np.concatenate([np.ones(20) * 0.5, np.ones(20) * 0.5])
    assert compute_fpr95(y, s) == pytest.approx(1.0)


def test_id_accuracy():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])  # 2/3 ID correct
    assert compute_id_accuracy(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_split_scores():
    y = np.array([0, 1, 0, 1])
    s = np.array([1.0, 2.0, 3.0, 4.0])
    id_s, ood_s = split_scores(y, s)
    assert np.array_equal(id_s, np.array([1.0, 3.0]))
    assert np.array_equal(ood_s, np.array([2.0, 4.0]))
