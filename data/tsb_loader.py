"""
TSB-StreamingAD Dataset Loader
==============================
Loads univariate (TSB-StreamingAD-U) and multivariate (TSB-StreamingAD-M)
streaming time-series datasets for OOD detection benchmarking.

Source: TSB-StreamingAD benchmark (Boniol et al., 2023 / 2024)
Format: CSV files with feature columns + binary 'Label' column (0=normal, 1=anomaly).

OOD Setup
---------
We define OOD at the *window* level:
  - ID  (label=0): windows where ALL timesteps are normal.
  - OOD (label=1): windows containing at least one anomalous timestep.

A sliding window with configurable size and stride extracts fixed-length
subsequences from the long streaming series.  Normal windows form the ID
training set; the evaluation sets contain balanced ID and OOD windows.

Source-Boundary Split (boundary_split=True, default)
-----------------------------------------------------
Each TSB file concatenates TWO source recordings.  The boundary between them
is the first timestep labelled anomalous (i.e., the start of Source 2).
Training uses ONLY Source 1 windows — those that end strictly before this
boundary.  The evaluation pool is then:
  - ID  windows : held-out Source 1 normal windows (after train_frac)
  - OOD windows : all windows that contain at least one anomalous timestep

This prevents the backbone from seeing Source 2 signal during training, which
would otherwise contaminate the representation and collapse AUROC toward 0.5.

Normalisation
-------------
  'per_series' : per-window z-score (removes level/trend signals — default)
  'global'     : fit mean/std on training windows (channel-wise, all timesteps),
                 apply to val/test.  Preserves inter-source level differences.
  'none'       : no normalisation.

Backbone Training — Pseudo-class Protocol
------------------------------------------
Normal training windows are divided into n_pseudo_classes equal temporal
segments, each assigned a unique integer label.  This gives the backbone a
discrimination objective without requiring real class annotations.

Parameters (all configurable via YAML)
---------------------------------------
  data_path        : path to the .csv file (relative to working directory)
  window_size      : timesteps per window (default 64)
  stride           : sliding-window stride in timesteps (default 32)
  n_pseudo_classes : temporal bins for backbone training labels (default 4)
  train_frac       : fraction of the SOURCE-1 pool used for training (default 0.70)
  val_frac         : not used directly — eval pool split 50/50 val/test
  max_rows         : cap on rows read (default None)
  seed             : random seed (default 42)
  normalize        : 'per_series' | 'global' | 'none' (default 'per_series')
  boundary_split   : if True (default), use only Source 1 for training
  dataset_name     : override display/results-dir name
  ordered_eval     : if True, keep val/test windows in original TEMPORAL order
                     (no balancing/shuffle) and additionally return a "stream"
                     entry — the full held-out eval window sequence in temporal
                     order with each window's start row. Default False reproduces
                     the current balanced+shuffled behaviour byte-for-byte.

Ordered evaluation (ordered_eval=True)
--------------------------------------
Temporal methods (drift detectors, LSTM/ADWIN, batch-level Fréchet) need the
eval windows in the order they occur in the file, not the shuffled balanced
pool. With ordered_eval=True the eval pool (held-out Source-1 normals + all
anomalous windows) is sorted by window index; val = first half, test = second
half, both in temporal order and NOT class-balanced. The full ordered pool is
also returned under dataset["stream"] = {"x", "y", "t"} where "t" is each
window's start row in the raw series.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .registry import register


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sliding windows from a 1-D or 2-D time series.

    Args:
        data  : (T, C) float array — C channels, T timesteps.
        labels: (T,) int array   — per-timestep binary labels.
        window_size: window length in timesteps.
        stride: step between consecutive windows.

    Returns:
        x : (N, C, window_size) float32 — N windows, each (C, window_size).
        y : (N,) int32          — 0 if all timesteps normal, 1 if any anomalous.
    """
    T, C = data.shape
    n_windows = max(0, (T - window_size) // stride + 1)
    x = np.empty((n_windows, C, window_size), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.int32)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        x[i] = data[start:end].T              # (C, window_size)
        y[i] = int(labels[start:end].max())   # 1 if any anomaly inside

    return x, y


def _normalize_per_series(x: np.ndarray) -> np.ndarray:
    """Per-window z-score normalisation: mean and std along the time axis."""
    mean = x.mean(axis=2, keepdims=True)
    std  = x.std(axis=2,  keepdims=True) + 1e-6
    return (x - mean) / std


def _normalize_global(
    x_train: np.ndarray,
    *others: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """
    Channel-wise global normalisation.  Fit mean/std on x_train (all windows
    and all time steps within them), then apply the same statistics to every
    array in *others.

    Shape: (N, C, T) — statistics are computed per channel.
    """
    # mean/std over (N, T) for each channel c -> shape (1, C, 1)
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std  = x_train.std(axis=(0, 2),  keepdims=True) + 1e-6
    result = tuple((arr - mean) / std for arr in (x_train, *others))
    return result


def _balance_binary(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Down-sample the majority class so that ID and OOD counts are equal.
    Returns (x_balanced, y_balanced) in a shuffled order.
    """
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return x, y
    n = min(len(idx0), len(idx1))
    sel0 = rng.choice(idx0, size=n, replace=False)
    sel1 = rng.choice(idx1, size=n, replace=False)
    idx = np.concatenate([sel0, sel1])
    idx = rng.permutation(idx)
    return x[idx], y[idx]


# ---------------------------------------------------------------------------
# Public loader — registered as 'tsb'
# ---------------------------------------------------------------------------

@register("tsb")
def load_tsb(
    data_path: str,
    window_size: int = 64,
    stride: int = 32,
    n_pseudo_classes: int = 4,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    max_rows: Optional[int] = None,
    seed: int = 42,
    normalize: str = "per_series",
    boundary_split: bool = True,
    dataset_name: Optional[str] = None,
    ordered_eval: bool = False,
) -> Dict[str, Any]:
    """
    Load a TSB-StreamingAD CSV file and build an OOD-ready dataset.

    The file may be either univariate (Data, Label) or multivariate
    (feat0, feat1, ..., Label).  The last column is always the label.

    Returns a dict compatible with the existing benchmark pipeline:
      {
        "train": {"x": (N_tr, C, T), "y": (N_tr,)},   # pseudo-class labels
        "val"  : {"x": (N_v,  C, T), "y": (N_v,)},    # binary 0/1
        "test" : {"x": (N_te, C, T), "y": (N_te,)},   # binary 0/1
        "metadata": {...}
      }
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"TSB dataset file not found: {data_path}")

    df = pd.read_csv(data_path, nrows=max_rows)

    label_col = df.columns[-1]
    feat_cols  = [c for c in df.columns if c != label_col]

    data_np   = df[feat_cols].to_numpy(dtype=np.float32)  # (T, C)
    labels_np = df[label_col].to_numpy(dtype=np.int32)    # (T,)

    T = len(data_np)
    C = len(feat_cols)
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    # ------------------------------------------------------------------
    # 2. Sliding windows
    # ------------------------------------------------------------------
    x_all, y_all = _sliding_windows(data_np, labels_np, window_size, stride)
    N_total = len(x_all)

    if N_total == 0:
        raise ValueError(
            f"No windows extracted from {data_path}. "
            f"File has {T} rows but window_size={window_size}."
        )

    idx_anomaly = np.where(y_all == 1)[0]

    if len(idx_anomaly) == 0:
        raise ValueError(
            f"No anomalous windows found in {data_path}. "
            "Cannot construct an OOD evaluation set."
        )

    # ------------------------------------------------------------------
    # 3. Determine training pool
    # ------------------------------------------------------------------
    # Find the first anomalous TIMESTEP — this is the Source 1 / Source 2
    # boundary in TSB files (all Source-1 timesteps are normal by construction).
    first_anom_row = int(np.argmax(labels_np > 0))
    has_anomaly = bool(labels_np[first_anom_row] > 0)

    use_boundary = boundary_split and has_anomaly

    if use_boundary:
        # Window i covers rows [i*stride, i*stride + window_size).
        # A window is entirely within Source 1 iff its last row < first_anom_row.
        src1_mask = np.array(
            [i * stride + window_size <= first_anom_row for i in range(N_total)]
        )
        src1_normal_idx = np.where(src1_mask & (y_all == 0))[0]

        if len(src1_normal_idx) < n_pseudo_classes:
            raise ValueError(
                f"Only {len(src1_normal_idx)} Source-1 normal windows found "
                f"(need >= {n_pseudo_classes}).  "
                "Try a smaller window_size/stride or set boundary_split: false."
            )

        n_train = max(n_pseudo_classes, int(len(src1_normal_idx) * train_frac))
        n_train = min(n_train, len(src1_normal_idx))

        train_idx        = src1_normal_idx[:n_train]
        eval_normal_idx  = src1_normal_idx[n_train:]   # held-out Source-1 normals
        eval_anomaly_idx = idx_anomaly                  # all anomalous windows

        split_mode = "boundary (Source-1 only)"
        src1_windows = len(src1_normal_idx)

    else:
        # Fallback: original temporal-fraction split on ALL normal windows
        idx_normal  = np.where(y_all == 0)[0]
        n_train     = max(n_pseudo_classes, int(len(idx_normal) * train_frac))
        n_train     = min(n_train, len(idx_normal))

        train_idx        = idx_normal[:n_train]
        eval_normal_idx  = idx_normal[n_train:]
        eval_anomaly_idx = idx_anomaly

        split_mode   = "temporal fraction (no boundary)"
        src1_windows = None

    if len(eval_anomaly_idx) < 4:
        raise ValueError(
            f"Too few anomalous windows ({len(eval_anomaly_idx)}) for evaluation. "
            "Try a smaller stride or pick a file with a higher anomaly rate."
        )

    # ------------------------------------------------------------------
    # 4. Val / test split of the eval pool
    # ------------------------------------------------------------------
    half_anom = len(eval_anomaly_idx) // 2
    half_norm = max(1, len(eval_normal_idx) // 2)

    val_anom_idx  = eval_anomaly_idx[:half_anom]
    test_anom_idx = eval_anomaly_idx[half_anom:]
    val_norm_idx  = eval_normal_idx[:half_norm]
    test_norm_idx = eval_normal_idx[half_norm:]

    # ------------------------------------------------------------------
    # 5. Construct raw splits
    # ------------------------------------------------------------------
    x_train = x_all[train_idx]
    actual_classes = min(n_pseudo_classes, len(train_idx))
    bin_edges = np.linspace(0, len(train_idx), actual_classes + 1, dtype=int)
    y_train   = np.zeros(len(train_idx), dtype=np.int32)
    for cls_idx in range(actual_classes):
        s, e = bin_edges[cls_idx], bin_edges[cls_idx + 1]
        y_train[s:e] = cls_idx

    stream = None
    if ordered_eval:
        # ------------------------------------------------------------------
        # Ordered evaluation: keep temporal order, no balancing, no shuffle.
        # val = first half of the ordered eval pool, test = second half.
        # ------------------------------------------------------------------
        val_idx  = np.sort(np.concatenate([val_norm_idx,  val_anom_idx]))
        test_idx = np.sort(np.concatenate([test_norm_idx, test_anom_idx]))

        x_val = x_all[val_idx]
        y_val = y_all[val_idx].astype(np.int32)
        x_test = x_all[test_idx]
        y_test = y_all[test_idx].astype(np.int32)

        # Full held-out eval window sequence in original temporal order.
        stream_idx = np.sort(np.concatenate([eval_normal_idx, eval_anomaly_idx]))
        stream_x = x_all[stream_idx]
        stream_y = y_all[stream_idx].astype(np.int32)
        stream_t = (stream_idx * stride).astype(np.int64)  # start row of each window
        stream = {"x": stream_x, "y": stream_y, "t": stream_t}

        if len(x_val) == 0 or len(x_test) == 0:
            raise ValueError(
                "Val or test split is empty in ordered_eval mode. "
                "Try a file with more anomalous/normal windows."
            )
    else:
        x_val = np.concatenate([x_all[val_norm_idx], x_all[val_anom_idx]], axis=0)
        y_val = np.concatenate([
            np.zeros(len(val_norm_idx), dtype=np.int32),
            np.ones( len(val_anom_idx), dtype=np.int32),
        ])
        x_val, y_val = _balance_binary(x_val, y_val, rng)

        x_test = np.concatenate([x_all[test_norm_idx], x_all[test_anom_idx]], axis=0)
        y_test = np.concatenate([
            np.zeros(len(test_norm_idx), dtype=np.int32),
            np.ones( len(test_anom_idx), dtype=np.int32),
        ])
        x_test, y_test = _balance_binary(x_test, y_test, rng)

        if len(x_val) == 0 or len(x_test) == 0:
            raise ValueError(
                "Val or test split is empty after balancing. "
                "Try a file with more anomalous windows."
            )

    # ------------------------------------------------------------------
    # 6. Normalise
    # ------------------------------------------------------------------
    if normalize == "per_series":
        x_train = _normalize_per_series(x_train)
        x_val   = _normalize_per_series(x_val)
        x_test  = _normalize_per_series(x_test)
        if stream is not None:
            stream["x"] = _normalize_per_series(stream["x"])
    elif normalize == "global":
        if stream is not None:
            x_train, x_val, x_test, stream["x"] = _normalize_global(
                x_train, x_val, x_test, stream["x"]
            )
        else:
            x_train, x_val, x_test = _normalize_global(x_train, x_val, x_test)
    # else: 'none' — leave raw

    # ------------------------------------------------------------------
    # 7. Summary printout
    # ------------------------------------------------------------------
    anom_rate = labels_np.mean() * 100
    src1_str  = (
        f"  Source-1   : {src1_windows} normal windows (boundary row {first_anom_row})\n"
        if use_boundary else ""
    )
    print(
        f"[TSB] {dataset_name[:60]}\n"
        f"  Raw series : {T} timesteps, {C} channel(s), "
        f"{anom_rate:.1f}% anomalous points\n"
        f"  Windows    : size={window_size}, stride={stride}, "
        f"{N_total} total ({(y_all==0).sum()} normal, {len(idx_anomaly)} anomalous)\n"
        f"{src1_str}"
        f"  Split      : {split_mode}, normalize={normalize}\n"
        f"  Train      : {len(x_train)} windows ({actual_classes} pseudo-classes)\n"
        f"  Val        : {len(x_val)} windows "
        f"({y_val.sum()} OOD / {(y_val==0).sum()} ID)\n"
        f"  Test       : {len(x_test)} windows "
        f"({y_test.sum()} OOD / {(y_test==0).sum()} ID)"
    )

    result = {
        "train": {"x": x_train, "y": y_train},
        "val":   {"x": x_val,   "y": y_val},
        "test":  {"x": x_test,  "y": y_test},
        "metadata": {
            "dataset_name":    dataset_name,
            "source_file":     data_path,
            "n_channels":      C,
            "window_size":     window_size,
            "stride":          stride,
            "n_pseudo_classes": actual_classes,
            "anomaly_rate_pct": float(anom_rate),
            "total_timesteps":  int(T),
            "first_anom_row":   int(first_anom_row) if has_anomaly else None,
            "boundary_split":   use_boundary,
            "normalize":        normalize,
            "ordered_eval":     bool(ordered_eval),
        },
    }
    if stream is not None:
        result["stream"] = stream
    return result
