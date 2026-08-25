"""Reproduce the full study end-to-end: main sweep + class-D appendix -> aggregate -> evaluate.

Seed 42, CPU, deterministic. Every table/figure in the paper is regenerated from results/.

Usage:
    .venv/Scripts/python runners/reproduce_all.py                 # full (~183 U datasets)
    .venv/Scripts/python runners/reproduce_all.py --scale few --n-per-cell 6   # quick
    .venv/Scripts/python runners/reproduce_all.py --split both    # U and M
    .venv/Scripts/python runners/reproduce_all.py --skip-class-d  # main only

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=["full", "few", "one"])
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--split", default="U", choices=["U", "M", "both"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-class-d", action="store_true")
    a = ap.parse_args()

    common = ["--scale", a.scale, "--split", a.split,
              "--n-per-cell", str(a.n_per_cell), "--epochs", str(a.epochs),
              "--seed", str(a.seed)]
    if a.dataset:
        common += ["--dataset", a.dataset]

    # 1. Main 17-detector sweep
    run([PY, "runners/run.py", "--group", "main", *common])
    # 2. Class-D appendix (separate protocols)
    if not a.skip_class_d:
        run([PY, "runners/run.py", "--group", "class_d", *common])
    # 3. Aggregate raw results -> results/benchmark.csv
    run([PY, "runners/aggregate.py"])
    # 4. Evaluate -> rankings, Friedman, tables, figures, findings.md
    run([PY, "runners/evaluate.py"])

    print("\nREPRODUCE COMPLETE -> results/ (findings.md, tables/, figures/)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
