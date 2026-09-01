"""BOUNDED feasibility probe: run ONLY the SRS detector on TSB-StreamingAD-M files.

This is a throwaway probe (NOT part of the main sweep). SRS is excluded on TSB-M in
run.py/method_set because it is univariate-seasonal (STL + conditional VAE) and was
feared to hang on long multivariate series. This script tests that empirically under a
HARD per-dataset timeout so a hang is killed and reported, never left unbounded.

Modes
-----
  worker : python srs_m_probe.py --file <path>       -> runs 1 dataset, prints one JSON line
  driver : python srs_m_probe.py --driver --n <k>    -> runs the k smallest M files, each in a
                                                        subprocess with --timeout seconds; writes
                                                        results/srs_m_probe.csv

Nothing here touches results/benchmark.csv, the paper, or method_set.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DATA_ROOT = os.environ.get("TSB_DATA_ROOT", r"C:\THESIS\benchmark1\datasets")
M_DIR = os.path.join(DATA_ROOT, "TSB-StreamingAD-M")
SRS_PARAMS = {"period": None, "latent_dim": 32, "n_epochs": 10}
SPLIT = "M"


def _smallest_m_files(n: int):
    files = glob.glob(os.path.join(M_DIR, "*.csv"))
    files.sort(key=lambda f: os.path.getsize(f))
    return files[:n]


def _spread_m_files(n: int):
    """Pick n files evenly spaced across the size distribution (smallest..largest),
    so the probe tests the hang hypothesis on medium/long series too, not just tiny ones."""
    files = glob.glob(os.path.join(M_DIR, "*.csv"))
    files.sort(key=lambda f: os.path.getsize(f))
    if n >= len(files):
        return files
    import numpy as np
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(files[i])
    return out


# ---------------------------------------------------------------------------
# worker: run one dataset (SRS only), print a single JSON result line
# ---------------------------------------------------------------------------

def run_worker(path: str, epochs: int) -> dict:
    import numpy as np
    import models.detectors  # noqa: F401  populate registry
    from core.registry import OOD_REGISTRY
    from runners import pipeline

    out = {"file": os.path.basename(path)}
    t0 = time.perf_counter()
    dataset, meta = pipeline.prepare_dataset(path, SPLIT, f"TSB-M-{os.path.basename(path)[:40]}")
    out["n_channels"] = int(dataset["train"]["x"].shape[1])
    out["window"] = int(dataset["train"]["x"].shape[2])
    out["n_train"] = int(dataset["train"]["x"].shape[0])
    out["n_test"] = int(dataset["test"]["x"].shape[0])
    out["category"] = meta["category"]

    in_ch = int(dataset["train"]["x"].shape[1])
    bb, head = pipeline.train_backbone(dataset, in_ch, epochs=epochs)

    cls = OOD_REGISTRY._items["srs"]
    tfit = time.perf_counter()
    r = pipeline.run_one(("srs", cls, SRS_PARAMS), dataset, bb, head,
                         os.path.join(_REPO_ROOT, "results", "_srs_m_probe_tmp",
                                      os.path.basename(path)[:40]),
                         epochs=epochs)
    out["srs_seconds"] = round(time.perf_counter() - tfit, 2)
    out["total_seconds"] = round(time.perf_counter() - t0, 2)
    out["status"] = r.get("status")
    out["auroc"] = r.get("auroc")
    out["aupr"] = r.get("aupr")
    out["error"] = r.get("error")
    return out


# ---------------------------------------------------------------------------
# driver: spawn one subprocess per dataset with a hard timeout
# ---------------------------------------------------------------------------

def run_driver(n: int, timeout: int, epochs: int, csv_path: str, strategy: str = "smallest"):
    files = _spread_m_files(n) if strategy == "spread" else _smallest_m_files(n)
    print(f"[driver] {len(files)} M files ({strategy}), per-dataset timeout={timeout}s, epochs={epochs}")
    rows = []
    for i, f in enumerate(files, 1):
        size_mb = os.path.getsize(f) / 1e6
        print(f"\n[{i}/{len(files)}] {os.path.basename(f)[:70]}  ({size_mb:.1f} MB)", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--file", f, "--epochs", str(epochs)]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            elapsed = round(time.perf_counter() - t0, 1)
            line = None
            for ln in proc.stdout.splitlines():
                if ln.startswith("__SRS_RESULT__"):
                    line = ln[len("__SRS_RESULT__"):]
            if line is not None:
                rec = json.loads(line)
                rec["wall_seconds"] = elapsed
                rec["outcome"] = "COMPLETED"
            else:
                rec = {"file": os.path.basename(f), "outcome": "NO_RESULT",
                       "wall_seconds": elapsed,
                       "stderr_tail": proc.stderr[-400:]}
            print(f"    -> outcome={rec['outcome']} status={rec.get('status')} "
                  f"auroc={rec.get('auroc')} srs_s={rec.get('srs_seconds')} wall={elapsed}s", flush=True)
        except subprocess.TimeoutExpired:
            elapsed = round(time.perf_counter() - t0, 1)
            rec = {"file": os.path.basename(f), "outcome": "TIMED_OUT",
                   "wall_seconds": elapsed}
            print(f"    -> TIMED_OUT after {elapsed}s (killed)", flush=True)
        rows.append(rec)

    cols = ["file", "outcome", "status", "category", "n_channels", "window",
            "n_train", "n_test", "auroc", "aupr", "srs_seconds", "total_seconds",
            "wall_seconds", "error", "stderr_tail"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"\n[driver] wrote {csv_path}")

    completed = [r for r in rows if r["outcome"] == "COMPLETED" and r.get("status") == "COMPLETE"]
    aurocs = [r["auroc"] for r in completed if isinstance(r.get("auroc"), (int, float))]
    print(f"[driver] completed-with-finite-auroc: {len(aurocs)}/{len(rows)}")
    if aurocs:
        print(f"[driver] SRS-M mean AUROC = {sum(aurocs)/len(aurocs):.4f} over {len(aurocs)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--driver", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--csv", default=os.path.join(_REPO_ROOT, "results", "srs_m_probe.csv"))
    ap.add_argument("--strategy", choices=["smallest", "spread"], default="smallest")
    args = ap.parse_args()

    if args.file:
        res = run_worker(args.file, args.epochs)
        print("__SRS_RESULT__" + json.dumps(res), flush=True)
    elif args.driver:
        run_driver(args.n, args.timeout, args.epochs, args.csv, args.strategy)
    else:
        ap.error("pass --file <path> (worker) or --driver (driver)")


if __name__ == "__main__":
    main()
