"""Rebuild results/benchmark.csv authoritatively from on-disk results.json files.

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

The per-dataset CSV append (runners/run.py) can lag behind disk if a run dies mid-dataset.
This rebuilder treats the per-(dataset, method) results.json files under results/tsb/ as the
source of truth, re-derives split/category/domain/normalize from the dataset id by matching
back to the actual TSB file, and writes a clean results/benchmark.csv.

Ported from experiments/aggregate_tsb.py and rewired to the THESIS_FINAL package scheme:
  * experiments/run_experiments + tsb_benchmark  -> runners/pipeline (parse_meta)
  * experiments/tsb/                              -> results/tsb/
  * results/tsb_benchmark.csv                     -> results/benchmark.csv
  * TB.U_DIR / TB.M_DIR                           -> data_root/TSB-StreamingAD-{U,M}

Usage
-----
    python runners/aggregate.py                       # default data root + results dir
    TSB_DATA_ROOT=... python runners/aggregate.py     # override the raw TSB corpus
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

# --- repo root on sys.path so core/, data/, models/, runners/ import ----------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from runners import pipeline  # noqa: E402  (parse_meta — same as run.py)

# Default TSB data root (read-only backup repo). Override with --data-root or the
# TSB_DATA_ROOT env var. Mirrors runners/run.py.
DEFAULT_DATA_ROOT = os.environ.get("TSB_DATA_ROOT", r"C:\THESIS\benchmark1\datasets")

OUT_ROOT = os.path.join(_REPO_ROOT, "results", "tsb")
CSV = os.path.join(_REPO_ROOT, "results", "benchmark.csv")


def build_meta_map(data_root: str):
    """Map dataset-id -> {split, category, domain, normalize} for every TSB file.

    The dataset id is built exactly as runners/run.py does:
    ``TSB-{split}-{basename[:40]}``.
    """
    m = {}
    for split in ("U", "M"):
        directory = os.path.join(data_root, f"TSB-StreamingAD-{split}")
        if not os.path.isdir(directory):
            continue
        for path in glob.glob(os.path.join(directory, "*.csv")):
            ds_id = f"TSB-{split}-{os.path.basename(path)[:40]}"
            meta = pipeline.parse_meta(path, split)
            m[ds_id] = {"split": split, "category": meta["category"],
                        "domain": meta["domain"], "normalize": meta["normalize"]}
    return m


def main():
    ap = argparse.ArgumentParser(description="Rebuild results/benchmark.csv from results/tsb/.")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT, help="per-run results tree (results/tsb)")
    ap.add_argument("--csv", default=CSV, help="combined CSV to (re)write")
    args = ap.parse_args()

    meta = build_meta_map(args.data_root)
    cols = ["method", "dataset", "split", "category", "domain", "normalize", "status",
            "auroc", "aupr", "fpr95", "det_acc", "inference_ms", "n_test", "seed", "timestamp"]
    rows = []
    for rj in glob.glob(os.path.join(args.out_root, "*", "*", "results.json")):
        try:
            with open(rj, encoding="utf-8") as fh:
                r = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        ds_id = os.path.basename(os.path.dirname(os.path.dirname(rj)))
        md = meta.get(ds_id, {"split": "?", "category": "?", "domain": "?", "normalize": "?"})
        rows.append({**md, "method": r.get("method"), "dataset": ds_id, "status": r.get("status"),
                     "auroc": r.get("auroc"), "aupr": r.get("aupr"), "fpr95": r.get("fpr95"),
                     "det_acc": r.get("det_acc"), "inference_ms": r.get("inference_ms"),
                     "n_test": r.get("n_test"), "seed": r.get("seed"), "timestamp": r.get("timestamp")})
    rows.sort(key=lambda x: (x["split"], x["category"], x["dataset"], x["method"]))
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    ds = {r["dataset"] for r in rows}
    print(f"rebuilt {args.csv}: {len(rows)} rows, {len(ds)} datasets, "
          f"{sum(1 for r in rows if r['status']=='COMPLETE')} COMPLETE")


if __name__ == "__main__":
    main()
