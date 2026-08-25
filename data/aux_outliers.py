"""Auxiliary-outlier corpus for the Class-D appendix study.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

This module builds an auxiliary-outlier corpus for the training/fine-tuning based
Class-D detectors (outlier_exposure / divoe / diversemix) from HOLD-OUT TSB files,
per the binding decisions in methods/_validation/CLASS_D_DECISIONS.md:

  * The 600-file TSB corpus (300 U + 300 M) is partitioned ONCE, stratified by
    category (DRIFT / OOD / STABLE), seed 42, into EVAL vs AUX file lists. The
    partition is persisted to a JSON manifest (aux_manifest.json) so that no aux
    file is ever used as an evaluation file — a hard no-leakage guarantee for the
    Class-D study.
  * Multivariate matching: aux windows must match the eval file's channel count
    exactly; if no aux file with that channel count exists, fall back to synthetic
    noise (get_aux_windows) or synthetic outliers from ID windows (the ablation
    generator).
  * Windows match the eval windowing (same window/stride and the same
    tsb_loader._sliding_windows extraction, same normalisation rule).

Public API
----------
  get_eval_files(split, category=None) -> list[str]
  get_aux_windows(split, n, in_channels, seed=42, normalize="per_series",
                  category=None, exclude_file=None) -> np.ndarray  # (n, C, T)
  synthetic_outliers(id_windows, n, seed=42, mode="mixed") -> np.ndarray  # (n, C, T)
  build_manifest(force=False) -> dict          # (re)build & persist the partition
  load_manifest() -> dict                      # load (building it if absent)

The synthetic generator is the ablation arm (noise / FFT phase-shuffle /
cross-window mixup). It is also the fallback when no channel-matched aux file
exists for a given eval file.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .tsb_loader import _normalize_per_series, _sliding_windows

# ---------------------------------------------------------------------------
# Locations & fixed conventions (kept in sync with experiments/tsb_benchmark.py)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))          # THESIS_FINAL/data
# The raw TSB corpus is NOT vendored into THESIS_FINAL (gitignored); it lives in
# the read-only backup repo. Resolve the data root from TSB_DATA_ROOT (the same
# env var runners/run.py honours for --data-root), falling back to that backup.
# The eval/aux partition is stratified by category with seed 42 over the SORTED
# file list, so pointing at the same corpus yields a partition identical to the
# original experiments/run_class_d.py run (a hard parity requirement).
DATA_ROOT = os.environ.get("TSB_DATA_ROOT", r"C:\THESIS\benchmark1\datasets")
U_DIR = os.path.join(DATA_ROOT, "TSB-StreamingAD-U")
M_DIR = os.path.join(DATA_ROOT, "TSB-StreamingAD-M")
MANIFEST_PATH = os.path.join(_HERE, "aux_manifest.json")

CATEGORIES = ("DRIFT", "OOD", "STABLE")
SEED = 42
# Fraction of each (split, category) cell reserved for the AUX corpus. The
# remainder are EVAL files. 40% aux leaves the majority for evaluation while
# giving enough channel-diverse files for exact-C matching on the M split.
AUX_FRAC = 0.40
MAX_ROWS = 150000            # cap very long series when extracting aux windows
MAX_WINDOWS_PER_FILE = 400   # cap windows drawn from any single aux file


def _split_dir(split: str) -> str:
    s = split.upper()
    if s == "U":
        return U_DIR
    if s == "M":
        return M_DIR
    raise ValueError(f"unknown split {split!r} (expected 'U' or 'M')")


def _split_window_stride(split: str):
    """Default window/stride per split, matching tsb_benchmark.parse_meta."""
    return (128, 64) if split.upper() == "M" else (64, 32)


def _category_of(path: str) -> str:
    return os.path.basename(path).split("_")[0]


def _list_split_files(split: str) -> List[str]:
    import glob
    return sorted(glob.glob(os.path.join(_split_dir(split), "*.csv")))


# ---------------------------------------------------------------------------
# Manifest: the persisted EVAL / AUX partition
# ---------------------------------------------------------------------------

def build_manifest(force: bool = False) -> Dict[str, Any]:
    """Partition every (split, category) cell into eval/aux, seed 42, and persist.

    Idempotent: if the manifest already exists and force is False it is returned
    unchanged (so the partition is stable across runs — the whole point of the
    no-leakage guarantee).
    """
    if os.path.exists(MANIFEST_PATH) and not force:
        return load_manifest()

    rng = np.random.default_rng(SEED)
    manifest: Dict[str, Any] = {
        "seed": SEED,
        "aux_frac": AUX_FRAC,
        "splits": {},
    }
    for split in ("U", "M"):
        files = _list_split_files(split)
        by_cat: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
        for f in files:
            c = _category_of(f)
            if c in by_cat:
                by_cat[c].append(f)
        eval_files: List[str] = []
        aux_files: List[str] = []
        for c in CATEGORIES:
            lst = sorted(by_cat[c])                      # deterministic base order
            perm = rng.permutation(len(lst))            # seed-42 stratified shuffle
            shuffled = [lst[i] for i in perm]
            n_aux = int(round(len(shuffled) * AUX_FRAC))
            aux_c = shuffled[:n_aux]
            eval_c = shuffled[n_aux:]
            aux_files.extend(aux_c)
            eval_files.extend(eval_c)
        # store basenames (portable); resolve to full paths on read
        manifest["splits"][split] = {
            "eval": sorted(os.path.basename(f) for f in eval_files),
            "aux": sorted(os.path.basename(f) for f in aux_files),
        }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_manifest() -> Dict[str, Any]:
    """Load the persisted manifest, building it on first use."""
    if not os.path.exists(MANIFEST_PATH):
        return build_manifest()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve(split: str, basenames: List[str]) -> List[str]:
    d = _split_dir(split)
    return [os.path.join(d, b) for b in basenames]


def get_eval_files(split: str, category: Optional[str] = None) -> List[str]:
    """Full paths of EVAL files for a split (optionally one category)."""
    man = load_manifest()
    names = man["splits"][split.upper()]["eval"]
    paths = _resolve(split, names)
    if category is not None:
        paths = [p for p in paths if _category_of(p) == category.upper()]
    return paths


def get_aux_files(split: str, category: Optional[str] = None) -> List[str]:
    """Full paths of AUX files for a split (optionally one category)."""
    man = load_manifest()
    names = man["splits"][split.upper()]["aux"]
    paths = _resolve(split, names)
    if category is not None:
        paths = [p for p in paths if _category_of(p) == category.upper()]
    return paths


# ---------------------------------------------------------------------------
# Channel-count detection & window extraction
# ---------------------------------------------------------------------------

def _file_channels(path: str) -> int:
    """Number of feature channels in a TSB file (= n_columns - 1 label col)."""
    header = pd.read_csv(path, nrows=0)
    return max(0, len(header.columns) - 1)


def _windows_from_file(path: str, window: int, stride: int, normalize: str) -> Optional[np.ndarray]:
    """Extract all windows from a file using the eval windowing, then normalise.

    Returns (M, C, window) float32 or None if the file yields no windows.
    Uses the SAME sliding-window extraction as load_tsb so aux windows match the
    eval windowing exactly. Labels are irrelevant for aux, so a zero label vector
    is passed to _sliding_windows.
    """
    df = pd.read_csv(path, nrows=MAX_ROWS)
    label_col = df.columns[-1]
    feat_cols = [c for c in df.columns if c != label_col]
    data = df[feat_cols].to_numpy(dtype=np.float32)                 # (T, C)
    if len(data) < window:
        return None
    labels = np.zeros(len(data), dtype=np.int32)
    x, _ = _sliding_windows(data, labels, window, stride)
    if len(x) == 0:
        return None
    if len(x) > MAX_WINDOWS_PER_FILE:
        # deterministic even subsample to bound cost
        sel = np.linspace(0, len(x) - 1, MAX_WINDOWS_PER_FILE, dtype=int)
        x = x[sel]
    if normalize == "per_series":
        x = _normalize_per_series(x)
    elif normalize == "global":
        mean = x.mean(axis=(0, 2), keepdims=True)
        std = x.std(axis=(0, 2), keepdims=True) + 1e-6
        x = (x - mean) / std
    # 'none' -> leave raw
    return x.astype(np.float32)


def get_aux_windows(
    split: str,
    n: int,
    in_channels: int,
    seed: int = 42,
    normalize: str = "per_series",
    category: Optional[str] = None,
    exclude_file: Optional[str] = None,
) -> np.ndarray:
    """Draw n auxiliary-outlier windows of exactly `in_channels` channels.

    Windows are drawn from HOLD-OUT aux files (never eval files) whose channel
    count == in_channels. If no such file exists, a synthetic-noise corpus of the
    right shape is returned instead (fallback). Returns (n, in_channels, T).
    """
    window, stride = _split_window_stride(split)
    rng = np.random.default_rng(seed)

    aux_files = get_aux_files(split, category)
    if exclude_file is not None:
        ex = os.path.basename(exclude_file)
        aux_files = [f for f in aux_files if os.path.basename(f) != ex]

    # exact channel-count match
    matched = [f for f in aux_files if _file_channels(f) == in_channels]
    rng.shuffle(matched)

    collected: List[np.ndarray] = []
    total = 0
    for f in matched:
        try:
            x = _windows_from_file(f, window, stride, normalize)
        except Exception:
            x = None
        if x is None or len(x) == 0:
            continue
        collected.append(x)
        total += len(x)
        if total >= max(n, 1) * 3:          # enough to sample without replacement
            break

    if total == 0:
        # synthetic-noise fallback (no channel-matched aux file available)
        return _synthetic_noise(n, in_channels, window, seed)

    pool = np.concatenate(collected, axis=0)
    if len(pool) >= n:
        idx = rng.choice(len(pool), size=n, replace=False)
    else:
        idx = rng.choice(len(pool), size=n, replace=True)
    return pool[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic outlier generator (ablation arm + fallback)
# ---------------------------------------------------------------------------

def _synthetic_noise(n: int, in_channels: int, window: int, seed: int = 42) -> np.ndarray:
    """Pure Gaussian-noise windows — the last-resort fallback."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, in_channels, window)).astype(np.float32)


def synthetic_outliers(
    id_windows: np.ndarray,
    n: int,
    seed: int = 42,
    mode: str = "mixed",
) -> np.ndarray:
    """Generate synthetic outliers from ID windows (ablation arm).

    Modes (all produce (n, C, T)):
      'noise'  : additive Gaussian noise injection
      'fft'    : FFT phase-shuffle (destroys temporal structure, keeps spectrum)
      'mixup'  : cross-window convex mixup of two ID windows
      'mixed'  : an equal blend of the three (default)

    id_windows: (M, C, T) array of in-distribution windows. If empty, falls back
    to pure Gaussian noise of shape (n, 1, T?) — callers should pass real windows.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(id_windows, dtype=np.float32)
    if x.ndim != 3 or len(x) == 0:
        raise ValueError("id_windows must be a non-empty (M, C, T) array")
    M, C, T = x.shape

    def _noise(k: int) -> np.ndarray:
        base = x[rng.integers(0, M, size=k)]
        sigma = base.std(axis=2, keepdims=True) + 1e-6
        return base + rng.standard_normal((k, C, T)).astype(np.float32) * sigma

    def _fft(k: int) -> np.ndarray:
        base = x[rng.integers(0, M, size=k)]
        spec = np.fft.rfft(base, axis=2)
        mag = np.abs(spec)
        rand_phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=spec.shape))
        shuffled = mag * rand_phase
        out = np.fft.irfft(shuffled, n=T, axis=2)
        return out.astype(np.float32)

    def _mixup(k: int) -> np.ndarray:
        a = x[rng.integers(0, M, size=k)]
        b = x[rng.integers(0, M, size=k)]
        lam = rng.uniform(0.2, 0.8, size=(k, 1, 1)).astype(np.float32)
        return lam * a + (1.0 - lam) * b

    if mode == "noise":
        out = _noise(n)
    elif mode == "fft":
        out = _fft(n)
    elif mode == "mixup":
        out = _mixup(n)
    elif mode == "mixed":
        k = n // 3
        parts = [_noise(k), _fft(k), _mixup(n - 2 * k)]
        out = np.concatenate(parts, axis=0)
        out = out[rng.permutation(len(out))]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience wrapper matching the BUILD_PLAN signature
# ---------------------------------------------------------------------------

def load_aux_outliers(
    split: str,
    in_channels: int,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    exclude_file: Optional[str] = None,
    n: int = 256,
    seed: int = 42,
    source: str = "tsb_holdout",
    normalize: str = "per_series",
    id_windows: Optional[np.ndarray] = None,
) -> np.ndarray:
    """BUILD_PLAN-compatible entry point. source in {'tsb_holdout','synthetic'}."""
    if source == "synthetic":
        if id_windows is not None and len(np.asarray(id_windows)) > 0:
            return synthetic_outliers(id_windows, n, seed=seed)
        w = window_size or _split_window_stride(split)[0]
        return _synthetic_noise(n, in_channels, w, seed)
    return get_aux_windows(
        split, n, in_channels, seed=seed, normalize=normalize, exclude_file=exclude_file
    )


if __name__ == "__main__":  # pragma: no cover - manual manifest build
    m = build_manifest()
    for sp in ("U", "M"):
        e = len(m["splits"][sp]["eval"])
        a = len(m["splits"][sp]["aux"])
        print(f"{sp}: {e} eval + {a} aux = {e + a} files")
