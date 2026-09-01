"""Online / incremental-covariance Mahalanobis under true streaming.

Author: Stylianos Giannoulis - AUTH MSc Data and Web Science - Supervisor: John Paparrizos

Tests the thesis's headline recommendation ("deploy Mahalanobis") under a GENUINE
streaming protocol. For a stratified set of TSB-StreamingAD-U files (categories
DRIFT / OOD / STABLE), we train the shared 1-D ResNet backbone exactly as
runners/pipeline.py does, then evaluate BOTH Mahalanobis variants on the SAME
temporally-ordered evaluation stream (data.tsb_loader.load_tsb(ordered_eval=True)):

  (a) batch  : models/detectors/mahalanobis.py - mean + tied covariance fit ONCE
               on the ID (Source-1) training windows, then frozen while scoring
               the whole ordered stream.
  (b) online : models/detectors/online_mahalanobis.py - the same warm-start, then
               the running mean + covariance are updated INCREMENTALLY after each
               streamed window (score-then-update, unsupervised, with slow
               exponential forgetting so the estimate can track gradual drift).

Per-window AUROC / AUPR / FPR95 are computed on the ordered stream for each variant
(core/metrics.py). Rows are written to results/online_incremental.csv and a summary
to results/online_findings.md, plus an optional comparison figure and LaTeX table.

Nothing here touches paper/, benchmark.csv, or the production run harness.

Config via environment:
  TSB_DATA_ROOT  : raw TSB corpus root (default C:\\THESIS\\benchmark1\\datasets)
  TSB_N_PER_CELL : eval files per (category) cell (default 20)
  TSB_SPLITS     : split(s) to run (default "U")
  ONLINE_DECAY   : forgetting factor for the online variant (default 0.999)
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# --- repo root on sys.path so core/, data/, models/ import ---------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.metrics import compute_aupr, compute_auroc, compute_fpr95  # noqa: E402
from data import aux_outliers as AUX  # noqa: E402
from data.tsb_loader import load_tsb  # noqa: E402
from models.detectors.mahalanobis import MahalanobisDetector  # noqa: E402
from models.detectors.online_mahalanobis import OnlineMahalanobisDetector  # noqa: E402
from runners import pipeline  # noqa: E402

SEED = 42
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "online_incremental.csv")
FINDINGS_PATH = os.path.join(RESULTS_DIR, "online_findings.md")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")

CATEGORIES = ("DRIFT", "OOD", "STABLE")
CSV_COLS = ["dataset", "category", "variant", "auroc", "aupr", "fpr95", "n", "seed", "timestamp"]


def _split_window_stride(split: str):
    return (128, 64) if split.upper() == "M" else (64, 32)


def _round(v) -> Optional[float]:
    return None if (v is None or not np.isfinite(v)) else round(float(v), 4)


def _stream_metrics(y: np.ndarray, s: np.ndarray) -> Dict[str, Optional[float]]:
    """Per-window AUROC / AUPR / FPR95 on the ordered stream (NaN-guarded)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64).ravel()
    if len(np.unique(y)) < 2 or not np.isfinite(s).all():
        return {"auroc": None, "aupr": None, "fpr95": None}
    return {
        "auroc": _round(compute_auroc(y, s)),
        "aupr": _round(compute_aupr(y, s)),
        "fpr95": _round(compute_fpr95(y, s)),
    }


def _write_rows(rows: List[Dict[str, Any]]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLS})


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def _process_file(path: str, split: str, category: str, decay: float,
                  rows: List[Dict[str, Any]]) -> None:
    window, stride = _split_window_stride(split)
    dataset_id = f"TSB-{split.upper()}-" + os.path.splitext(os.path.basename(path))[0][:48]
    dataset = load_tsb(
        data_path=path, window_size=window, stride=stride, n_pseudo_classes=4,
        train_frac=0.70, seed=SEED, normalize="per_series",
        boundary_split=True, ordered_eval=True, dataset_name=dataset_id,
    )
    if "stream" not in dataset:
        raise RuntimeError("ordered stream missing from load_tsb output")
    stream = dataset["stream"]
    stream_x, stream_y = stream["x"], np.asarray(stream["y"]).astype(int)
    if len(np.unique(stream_y)) < 2:
        raise RuntimeError("ordered stream has a single class (no AUROC defined)")

    in_channels = int(dataset["train"]["x"].shape[1])
    bb, head = pipeline.train_backbone(dataset, in_channels)
    train_x, train_y = dataset["train"]["x"], dataset["train"]["y"]

    ts = datetime.now().isoformat(timespec="seconds")
    n_stream = int(len(stream_y))

    # (a) BATCH Mahalanobis: fit once on ID training windows, freeze, score stream.
    pipeline.set_seed(SEED)
    batch_det = MahalanobisDetector(
        model=bb.model, config={"classifier": head, "device": "cpu"}
    )
    batch_det.fit(train_x, train_y)
    t0 = time.perf_counter()
    s_batch = np.asarray(batch_det.score(stream_x), dtype=np.float64).ravel()
    batch_secs = time.perf_counter() - t0
    m_batch = _stream_metrics(stream_y, s_batch)

    # (b) ONLINE incremental Mahalanobis: warm-start, then update per window.
    pipeline.set_seed(SEED)
    online_det = OnlineMahalanobisDetector(
        model=bb.model,
        config={"device": "cpu", "decay": decay, "mode": "decay",
                "ridge": 1e-6, "shrinkage": 0.02, "mini_batch": 1},
    )
    online_det.fit(train_x, train_y)
    t0 = time.perf_counter()
    s_online = np.asarray(online_det.score_stream(stream_x), dtype=np.float64).ravel()
    online_secs = time.perf_counter() - t0
    m_online = _stream_metrics(stream_y, s_online)

    for variant, m, secs in (("batch", m_batch, batch_secs),
                             ("online", m_online, online_secs)):
        row = {
            "dataset": dataset_id, "category": category, "variant": variant,
            "auroc": m["auroc"], "aupr": m["aupr"], "fpr95": m["fpr95"],
            "n": n_stream, "seed": SEED, "timestamp": ts,
        }
        rows.append(row)
        _write_rows([row])
    print(f"  batch  auroc={m_batch['auroc']} aupr={m_batch['aupr']} "
          f"fpr95={m_batch['fpr95']} ({batch_secs:.2f}s) | "
          f"online auroc={m_online['auroc']} aupr={m_online['aupr']} "
          f"fpr95={m_online['fpr95']} ({online_secs:.2f}s) | n={n_stream}", flush=True)


# ---------------------------------------------------------------------------
# Summary + optional figure/table
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _summarise(rows: List[Dict[str, Any]], runtime_s: float, decay: float,
               splits: List[str], n_per_cell: int) -> None:
    # Pair batch/online per dataset.
    by_ds: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["variant"]] = r
    paired = {ds: v for ds, v in by_ds.items() if "batch" in v and "online" in v}

    cats = sorted({r["category"] for r in rows})
    lines: List[str] = []
    lines.append("# Online / incremental-covariance Mahalanobis under true streaming")
    lines.append("")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')} - "
                 f"seed {SEED}, splits {','.join(splits)}, {n_per_cell} files/category, "
                 f"online decay={decay}._")
    lines.append("")
    lines.append("Per-window AUROC/AUPR/FPR95 on the temporally-ordered evaluation "
                 "stream (`load_tsb(ordered_eval=True)`). **batch** = tied covariance "
                 "fit once on the ID training windows then frozen; **online** = same "
                 "warm-start, then mean+covariance updated incrementally after each "
                 "streamed window (score-then-update, unsupervised, exponential "
                 "forgetting). Every number below is traceable to "
                 "`results/online_incremental.csv`.")
    lines.append("")

    def _cat_means(metric: str, cat: Optional[str] = None):
        b = [p["batch"].get(metric) for ds, p in paired.items()
             if cat is None or p["batch"]["category"] == cat]
        o = [p["online"].get(metric) for ds, p in paired.items()
             if cat is None or p["online"]["category"] == cat]
        return _mean(b), _mean(o)

    # Overall + per-category AUROC table.
    lines.append("## Mean AUROC: batch vs online")
    lines.append("")
    lines.append("| Scope | n datasets | batch AUROC | online AUROC | delta (online-batch) |")
    lines.append("|---|---|---|---|---|")
    n_all = len(paired)
    b_all, o_all = _cat_means("auroc")
    delta_all = None if (b_all is None or o_all is None) else o_all - b_all
    lines.append(f"| **Overall** | {n_all} | {_fmt(b_all)} | {_fmt(o_all)} | {_fmt(delta_all)} |")
    for cat in cats:
        n_c = sum(1 for p in paired.values() if p["batch"]["category"] == cat)
        b_c, o_c = _cat_means("auroc", cat)
        d_c = None if (b_c is None or o_c is None) else o_c - b_c
        lines.append(f"| {cat} | {n_c} | {_fmt(b_c)} | {_fmt(o_c)} | {_fmt(d_c)} |")
    lines.append("")

    # AUPR / FPR95 overall.
    b_aupr, o_aupr = _cat_means("aupr")
    b_fpr, o_fpr = _cat_means("fpr95")
    lines.append("## Mean AUPR / FPR95 (overall)")
    lines.append("")
    lines.append("| Metric | batch | online |")
    lines.append("|---|---|---|")
    lines.append(f"| AUPR (higher better) | {_fmt(b_aupr)} | {_fmt(o_aupr)} |")
    lines.append(f"| FPR@95 (lower better) | {_fmt(b_fpr)} | {_fmt(o_fpr)} |")
    lines.append("")

    # Win / tie / loss by AUROC (online vs batch), tolerance 0.005.
    tol = 0.005
    wins = ties = losses = 0
    for p in paired.values():
        ba, oa = p["batch"].get("auroc"), p["online"].get("auroc")
        if ba is None or oa is None:
            continue
        if oa - ba > tol:
            wins += 1
        elif ba - oa > tol:
            losses += 1
        else:
            ties += 1
    lines.append("## Online vs batch, per-dataset AUROC (tolerance +/-0.005)")
    lines.append("")
    lines.append(f"- online **wins**: {wins}")
    lines.append(f"- **ties**: {ties}")
    lines.append(f"- online **loses**: {losses}")
    lines.append("")

    verdict = "matches"
    if delta_all is not None:
        if delta_all > tol:
            verdict = "beats"
        elif delta_all < -tol:
            verdict = "trails"
    lines.append(f"**Verdict (streaming-deployment claim):** online incremental "
                 f"Mahalanobis **{verdict}** the fit-once batch detector on mean AUROC "
                 f"(delta {_fmt(delta_all)}).")
    lines.append("")
    lines.append(f"_Total wall-clock runtime: {runtime_s/60.0:.1f} min "
                 f"({len(paired)} datasets x 2 variants)._")
    lines.append("")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(FINDINGS_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {FINDINGS_PATH}", flush=True)

    _make_figure(paired, cats)
    _make_table(paired, cats, b_all, o_all, delta_all)


def _make_figure(paired, cats) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (figure skipped: {exc})", flush=True)
        return
    b = [p["batch"].get("auroc") for p in paired.values()
         if p["batch"].get("auroc") is not None and p["online"].get("auroc") is not None]
    o = [p["online"].get("auroc") for p in paired.values()
         if p["batch"].get("auroc") is not None and p["online"].get("auroc") is not None]
    c = [p["batch"]["category"] for p in paired.values()
         if p["batch"].get("auroc") is not None and p["online"].get("auroc") is not None]
    if not b:
        print("  (figure skipped: no paired finite AUROCs)", flush=True)
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    colours = {"DRIFT": "#d1495b", "OOD": "#00798c", "STABLE": "#edae49"}
    for cat in cats:
        xs = [bb for bb, cc in zip(b, c) if cc == cat]
        ys = [oo for oo, cc in zip(o, c) if cc == cat]
        ax.scatter(xs, ys, s=28, alpha=0.75, label=cat,
                   color=colours.get(cat, "#666666"), edgecolors="white", linewidths=0.5)
    lim = [min(min(b), min(o)) - 0.02, 1.005]
    ax.plot(lim, lim, ls="--", c="#888888", lw=1.0, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("batch Mahalanobis AUROC (fit-once)")
    ax.set_ylabel("online incremental Mahalanobis AUROC")
    ax.set_title("Per-window AUROC on the ordered stream\n(points above diagonal: online wins)")
    ax.legend(title="category", frameon=False, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"online_vs_batch.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(FIG_DIR, 'online_vs_batch.{png,pdf}')}", flush=True)


def _make_table(paired, cats, b_all, o_all, delta_all) -> None:
    os.makedirs(TAB_DIR, exist_ok=True)
    path = os.path.join(TAB_DIR, "online_incremental.tex")

    def _cat_mean(metric, variant, cat=None):
        vals = [p[variant].get(metric) for p in paired.values()
                if (cat is None or p[variant]["category"] == cat)
                and p[variant].get(metric) is not None]
        return float(np.mean(vals)) if vals else None

    def _c(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Batch (fit-once) vs.\ online incremental-covariance Mahalanobis "
        r"under a true streaming protocol: mean per-window AUROC on the "
        r"temporally-ordered TSB-StreamingAD-U evaluation stream. "
        r"$\Delta = \text{online} - \text{batch}$.}",
        r"\label{tab:online_incremental}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Category & $n$ & Batch AUROC & Online AUROC & $\Delta$ \\",
        r"\midrule",
    ]
    for cat in cats:
        n_c = sum(1 for p in paired.values() if p["batch"]["category"] == cat)
        b_c = _cat_mean("auroc", "batch", cat)
        o_c = _cat_mean("auroc", "online", cat)
        d_c = None if (b_c is None or o_c is None) else o_c - b_c
        lines.append(f"{cat} & {n_c} & {_c(b_c)} & {_c(o_c)} & {_c(d_c)} \\\\")
    lines.append(r"\midrule")
    n_all = len(paired)
    lines.append(f"\\textbf{{Overall}} & {n_all} & {_c(b_all)} & {_c(o_all)} & {_c(delta_all)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(splits: List[str], n_per_cell: int, decay: float) -> None:
    AUX.build_manifest()  # ensure the eval/aux partition exists (no-leakage guarantee)
    print(f"Online-Mahalanobis streaming experiment: splits={splits} "
          f"n_per_cell={n_per_cell} decay={decay}", flush=True)

    # Fresh CSV each run (idempotent, single authoritative artefact).
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for split in splits:
        for cat in CATEGORIES:
            files = AUX.get_eval_files(split, cat)[:n_per_cell]
            for f in files:
                print(f"\n===== {split}/{cat}: {os.path.basename(f)[:70]} =====", flush=True)
                try:
                    _process_file(f, split, cat, decay, rows)
                except Exception as exc:  # noqa: BLE001
                    print(f"  SKIP FILE ({exc.__class__.__name__}: {exc})", flush=True)
                    print(traceback.format_exc()[-600:], flush=True)
                    continue
    runtime_s = time.perf_counter() - t_start

    finite = [r for r in rows if r.get("auroc") is not None]
    print(f"\nDONE: {len(finite)}/{len(rows)} variant runs produced a finite AUROC "
          f"-> {CSV_PATH}", flush=True)
    if rows:
        _summarise(rows, runtime_s, decay, splits, n_per_cell)


def main() -> None:
    splits = [s.strip().upper() for s in os.environ.get("TSB_SPLITS", "U").split(",") if s.strip()]
    n_per_cell = int(os.environ.get("TSB_N_PER_CELL", "20"))
    decay = float(os.environ.get("ONLINE_DECAY", "0.999"))
    run(splits, n_per_cell, decay)


if __name__ == "__main__":
    main()
