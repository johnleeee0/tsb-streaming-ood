"""Single CLI entrypoint for the TSB-StreamingAD benchmark sweep (main detector set).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Builds the detector set from the OOD registry (all @register_ood main detectors in
models/detectors/), runs the stratified sweep over TSB-StreamingAD files, and writes
results/benchmark.csv. Resumable: a (dataset, method) pair whose results.json already
exists is skipped.

Examples
--------
    python runners/run.py --scale full                 # 100 files / (split, category), U
    python runners/run.py --scale few --n-per-cell 6   # small stratified sample
    python runners/run.py --scale one --dataset OOD_009 # a single matching file
    python runners/run.py --scale full --split both     # U and M
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

# --- repo root on sys.path so core/, data/, models/ import ---------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models.detectors  # noqa: E402,F401  (populates OOD_REGISTRY via @register_ood)
from core.registry import OOD_REGISTRY  # noqa: E402
from runners import pipeline  # noqa: E402

SEED = 42

# Default TSB data root (read-only backup repo). Override with --data-root or the
# TSB_DATA_ROOT env var. THESIS_FINAL gitignores the raw corpus.
DEFAULT_DATA_ROOT = os.environ.get(
    "TSB_DATA_ROOT", r"C:\THESIS\benchmark1\datasets"
)

# ---------------------------------------------------------------------------
# Main detector set — canonical order + EXACT params (from tsb_benchmark.method_set).
# Classes are pulled from the registry so there is no hardcoded class split.
# ---------------------------------------------------------------------------
MAIN_ORDER = [
    "msp", "odin", "energy", "mahalanobis", "dfm_pca", "srs", "react", "dice",
    "scale", "gradnorm", "dimmad", "catsight", "codit", "invad", "m2n2",
    "deedee", "diffad",
]

METHOD_PARAMS = {
    "msp": {"temperature": 1.0},
    "odin": {"temperature": 1000.0, "epsilon": 0.001},
    "energy": {"score_type": "energy"},
    "mahalanobis": {},
    "dfm_pca": {"n_components": 32},
    "srs": {"period": None, "latent_dim": 32, "n_epochs": 10},
    "react": {"percentile": 90},
    "dice": {"top_k": 20},
    "scale": {"percentile": 85},
    "gradnorm": {"temperature": 1.0},
    "dimmad": {"class_agg": "min", "metric_agg": "median"},
    "catsight": {"n_components": 6, "reg": 1e-4},
    "codit": {"n_epochs": 10, "eval_n": 5},
    "invad": {"n_epochs": 10, "hidden_dim": 128, "n_layers": 2},
    "m2n2": {"n_epochs": 10, "hidden_dim": 32, "gamma": 0.999},
    "deedee": {},
    "diffad": {"n_steps": 20, "n_epochs": 8, "hidden_dim": 64, "recon_samples": 2},
}


def build_specs(split: str):
    """Return [(name, cls, params)] for the main set, honouring the split rules.

    SRS (univariate-seasonal; hangs on long multivariate M series) is excluded on
    TSB-M, matching tsb_benchmark.method_set.
    """
    specs = []
    for name in MAIN_ORDER:
        if split == "M" and name == "srs":
            continue
        cls = OOD_REGISTRY._items[name]
        specs.append((name, cls, METHOD_PARAMS.get(name, {})))
    return specs


def stratified_files(directory: str, rng: np.random.Generator):
    """Full shuffled candidate list per DRIFT/OOD/STABLE cell (take first n that load)."""
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    by_cat = {"DRIFT": [], "OOD": [], "STABLE": []}
    for f in files:
        c = os.path.basename(f).split("_")[0]
        if c in by_cat:
            by_cat[c].append(f)
    chosen = []
    for c, lst in by_cat.items():
        rng.shuffle(lst)
        chosen.append((c, lst))
    return chosen


def append_rows(csv_path: str, rows):
    new = not os.path.exists(csv_path)
    cols = ["method", "dataset", "split", "category", "domain", "normalize", "status",
            "auroc", "aupr", "fpr95", "det_acc", "inference_ms", "n_test", "seed", "timestamp"]
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _already_done(out_root: str, ds_id: str, method: str) -> bool:
    return os.path.exists(os.path.join(out_root, ds_id, method, "results.json"))


def _run_file(path, split, specs, out_root, csv_path, epochs, seed, cat_label=""):
    """Load one file, train the backbone once, run all (pending) methods. Returns
    True if the file loaded, False if it was unusable (skip and try next)."""
    ds_id = f"TSB-{split}-{os.path.basename(path)[:40]}"
    spec_names = [s[0] for s in specs]
    if all(_already_done(out_root, ds_id, n) for n in spec_names):
        print(f"  [{split}/{cat_label}] {ds_id[:34]:34s} all {len(spec_names)} methods done -> skip", flush=True)
        return True
    try:
        dataset, meta = pipeline.prepare_dataset(path, split, ds_id, seed=seed)
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {ds_id}: {str(exc)[:80]}", flush=True)
        return False
    try:
        in_ch = int(dataset["train"]["x"].shape[1])
        bb, head = pipeline.train_backbone(dataset, in_ch, epochs=epochs)
        for name, cls, params in specs:
            if _already_done(out_root, ds_id, name):
                continue
            out_dir = os.path.join(out_root, ds_id, name)
            r = pipeline.run_one((name, cls, params), dataset, bb, head, out_dir, epochs=epochs)
            row = {**r, "split": split, "category": meta["category"],
                   "domain": meta["domain"], "normalize": meta["normalize"], "dataset": ds_id}
            append_rows(csv_path, [row])
            print(f"  [{split}/{cat_label}] {ds_id[:34]:34s} {name:16s} {r['status']:8s} "
                  f"auroc={r.get('auroc', r.get('error', '?'))}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  dataset-level error on {ds_id}: {str(exc)[:100]}", flush=True)
    return True


def _run_class_d_group(args):
    """Dispatch the Class-D appendix sweep (--group class_d).

    Honours --scale / --n-per-cell / --split exactly like the main sweep, and
    --cd-group to pick group 1/2/3 (default: all). Writes results/class_d_group{1,2,3}.csv
    via runners/class_d.py. The main --group behaviour is untouched.
    """
    # aux_outliers resolves the raw TSB corpus from TSB_DATA_ROOT; forward --data-root
    # (set BEFORE importing runners.class_d so aux_outliers reads it at import time).
    os.environ["TSB_DATA_ROOT"] = args.data_root

    from runners import class_d as CD  # noqa: E402  (lazy: after env is set)

    if args.scale == "full":
        n_per_cell = 100
    elif args.scale == "few":
        n_per_cell = args.n_per_cell
    else:  # one
        n_per_cell = 1

    splits = ["U", "M"] if args.split == "both" else [args.split]

    print(f"Run: group=class_d cd_group={args.cd_group} scale={args.scale} "
          f"n_per_cell={n_per_cell} splits={splits} seed={args.seed}", flush=True)
    print(f"     data_root={args.data_root}", flush=True)
    print(f"     results -> {os.path.join(_REPO_ROOT, 'results')}\\class_d_*.csv", flush=True)

    CD.run_class_d(splits, n_per_cell, group=args.cd_group)
    print("CLASS-D SWEEP COMPLETE -> results/class_d_group{1,2,3}.csv", flush=True)


def main():
    ap = argparse.ArgumentParser(description="TSB-StreamingAD benchmark (main detector set).")
    ap.add_argument("--scale", choices=["full", "few", "one"], default="few")
    ap.add_argument("--dataset", default=None, help="basename substring (required for --scale one)")
    ap.add_argument("--n-per-cell", type=int, default=6, dest="n_per_cell")
    ap.add_argument("--split", choices=["U", "M", "both"], default="U")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--group", choices=["main", "class_d"], default="main")
    ap.add_argument("--cd-group", dest="cd_group", default="all",
                    choices=["all", "1", "2", "3", "group1", "group2", "group3"],
                    help="(--group class_d only) which Class-D group sweep(s) to run")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", default=os.path.join(_REPO_ROOT, "results", "tsb"))
    ap.add_argument("--csv", default=os.path.join(_REPO_ROOT, "results", "benchmark.csv"))
    args = ap.parse_args()

    # -- Class-D appendix sweep: a SEPARATE path (never touches --group main) -----
    if args.group == "class_d":
        _run_class_d_group(args)
        return

    if args.scale == "full":
        n_per_cell = 100
    elif args.scale == "few":
        n_per_cell = args.n_per_cell
    else:  # one
        n_per_cell = 1

    splits = ["U", "M"] if args.split == "both" else [args.split]
    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Run: scale={args.scale} n_per_cell={n_per_cell} splits={splits} "
          f"epochs={args.epochs} seed={args.seed} group={args.group}", flush=True)
    print(f"     data_root={args.data_root}", flush=True)
    print(f"     csv={args.csv}", flush=True)

    for split in splits:
        directory = os.path.join(args.data_root, f"TSB-StreamingAD-{split}")
        if not os.path.isdir(directory):
            print(f"  [WARN] missing data dir: {directory}", flush=True)
            continue
        specs = build_specs(split)

        if args.scale == "one":
            if not args.dataset:
                raise SystemExit("--scale one requires --dataset <basename substring>")
            matches = sorted(
                f for f in glob.glob(os.path.join(directory, "*.csv"))
                if args.dataset in os.path.basename(f)
            )
            if not matches:
                print(f"  [WARN] no file in {split} matching '{args.dataset}'", flush=True)
                continue
            _run_file(matches[0], split, specs, args.out_root, args.csv,
                      args.epochs, args.seed, cat_label="one")
            continue

        for cat, candidates in stratified_files(directory, rng):
            loaded = 0
            for path in candidates:
                if loaded >= n_per_cell:
                    break
                ok = _run_file(path, split, specs, args.out_root, args.csv,
                               args.epochs, args.seed, cat_label=cat)
                if ok:
                    loaded += 1
            print(f"== {split}/{cat}: {loaded} datasets done ==", flush=True)

    print("BENCHMARK SWEEP COMPLETE -> results/benchmark.csv", flush=True)


if __name__ == "__main__":
    main()
