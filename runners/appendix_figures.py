"""Appendix figures from saved per-run scores (real arrays, no synthesis).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Ported from results/appendix_figures.py and rewired to the THESIS_FINAL layout:
  * experiments/<ds>/<method>/{scores,labels}.npy  -> results/tsb/<ds>/<method>/*.npy
  * figures land under results/figures/

Reads the per-run scores.npy / labels.npy that runners/pipeline.run_one saves under
results/tsb/, so it only draws panels for (dataset, method) pairs actually present on
disk (missing pairs are silently skipped). The DATASETS / HILITE lists below are the
paper's chosen case studies; adjust them to match the ids in your results/tsb/ tree.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402,F401

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TSB = os.path.join(_REPO_ROOT, "results", "tsb")
FIG = os.path.join(_REPO_ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

DATASETS = ["TSB-M-DRIFT003", "TSB-U-OOD009", "TSB-U-DRIFT024", "TSB-U-STABLE001"]
HILITE = ["mahalanobis", "driftlens", "dfm_pca", "msp", "gradnorm", "odin"]


def load(ds, m):
    d = os.path.join(TSB, ds, m)
    sp, lp = os.path.join(d, "scores.npy"), os.path.join(d, "labels.npy")
    if os.path.exists(sp) and os.path.exists(lp):
        return np.load(sp), np.load(lp)
    return None, None


def roc_panel():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, ds in zip(axes.ravel(), DATASETS):
        for m in HILITE:
            s, y = load(ds, m)
            if s is None or len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, label=m, lw=1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(ds); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("ROC curves: feature/drift detectors (solid leaders) vs post-hoc softmax")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"appendix_roc.{ext}"), dpi=150)
    plt.close(fig)


def violin_case_study():
    ds = "TSB-U-OOD009"
    pair = ["mahalanobis", "msp"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, pair):
        s, y = load(ds, m)
        if s is None:
            continue
        parts = [s[y == 0], s[y == 1]]
        ax.violinplot(parts, showmeans=True)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["ID", "OOD"])
        ax.set_title(f"{m} on {ds}"); ax.set_ylabel("OOD score")
    fig.suptitle("Score separation under global normalisation: "
                 "feature-space (left) vs softmax (right, inverted)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"appendix_violin_case.{ext}"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    roc_panel()
    violin_case_study()
    print("wrote appendix_roc and appendix_violin_case figures")
