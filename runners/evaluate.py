"""Evaluator for the TSB-StreamingAD-M/-U benchmark (main + Class-D appendix).

Author: Stylianos Giannoulis — AUTH MSc Data and Web Science — Supervisor: John Paparrizos

Consumes results/benchmark.csv (main 17-detector set) and, if present,
results/class_d_group{1,2,3}.csv (the 7-detector appendix study), and answers the
Phase 3 research questions:
  * Which methods generalize best (overall + per split U/M)?
  * Which fail under streaming conditions (mean AUROC <= 0.5 / inverted)?
  * How does performance break down by category (DRIFT/OOD/STABLE), domain, and
    normalization (global vs per_series) — i.e. does the overconfidence dichotomy
    hold at scale?
  * Accuracy/efficiency tradeoffs (AUROC vs inference time).
  * Statistical significance (Friedman).

Ported from results/evaluate_tsb.py and rewired to the THESIS_FINAL layout:
  * results/tsb_benchmark.csv          -> results/benchmark.csv
  * results/class_d_group{1,2,3}.csv   -> Class-D appendix summary (new, optional)
  * outputs land under results/tables/ and results/figures/ (unchanged names)

Emits: results/findings.md, results/tables/tsb_main.tex, results/tables/tsb_by_split.tex,
       results/tables/class_d_appendix.tex (if Class-D CSVs exist),
       results/figures/tsb_heatmap_category.{png,pdf}, tsb_split_bar.*, tsb_norm_dichotomy.*,
       tsb_efficiency.*
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import friedmanchisquare  # noqa: E402

# results/ dir = this file's ../results
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(_REPO_ROOT, "results")
T = os.path.join(R, "tables"); F = os.path.join(R, "figures")
os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)

BENCHMARK_CSV = os.path.join(R, "benchmark.csv")


def load():
    df = pd.read_csv(BENCHMARK_CSV)
    df = df[df["status"] == "COMPLETE"].copy()
    for c in ["auroc", "aupr", "fpr95", "det_acc", "inference_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["auroc"])


def ranking(df):
    g = df.groupby("method")
    out = pd.DataFrame({
        "mean_auroc": g["auroc"].mean(), "std_auroc": g["auroc"].std(),
        "mean_aupr": g["aupr"].mean(), "mean_fpr95": g["fpr95"].mean(),
        "mean_inference_ms": g["inference_ms"].mean(), "n": g["auroc"].count(),
    }).sort_values("mean_auroc", ascending=False)
    return out


def friedman(df):
    piv = df.pivot_table(index="method", columns="dataset", values="auroc", aggfunc="mean").dropna(axis=1)
    piv = piv.dropna()
    if piv.shape[0] < 3 or piv.shape[1] < 2:
        return "insufficient complete cells"
    # (a) Complete-case over ALL methods. If a method is unrun on a split (e.g. SRS on
    #     TSB-M), complete-case pruning drops those columns -> N is that split's size.
    s, p = friedmanchisquare(*[piv.loc[m].values for m in piv.index])
    out = f"chi2={s:.2f}, p={p:.3g} (k={piv.shape[0]} methods, N={piv.shape[1]} datasets, complete-case)"
    # (b) Full-coverage: drop methods missing on >10% of datasets (e.g. SRS), so all
    #     splits are retained. Report only if it yields more datasets than (a).
    full = df.pivot_table(index="method", columns="dataset", values="auroc", aggfunc="mean")
    keep = full.index[full.notna().mean(axis=1) >= 0.90]
    piv2 = full.loc[keep].dropna(axis=1)
    if piv2.shape[0] >= 3 and piv2.shape[1] > piv.shape[1]:
        s2, p2 = friedmanchisquare(*[piv2.loc[m].values for m in piv2.index])
        dropped = sorted(set(full.index) - set(keep))
        out += (f"; full-coverage chi2={s2:.2f}, p={p2:.3g} "
                f"(k={piv2.shape[0]}, N={piv2.shape[1]}, excl. {','.join(dropped)})")
    return out


def tex_table(rank, path, caption, label):
    lines = [f"% auto-generated from benchmark.csv",
             "\\begin{table}[t]\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\small\\begin{tabular}{lrrrr}\\toprule",
             "Method & Mean AUROC & Std & Mean FPR95 & Infer (ms) \\\\\\midrule"]
    for m, r in rank.iterrows():
        lines.append(f"{m.replace('_','-')} & {r['mean_auroc']:.3f} & {r['std_auroc']:.3f} "
                     f"& {r['mean_fpr95']:.3f} & {r['mean_inference_ms']:.3f} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Class-D appendix (the 7 excluded detectors) — read results/class_d_*.csv
# ---------------------------------------------------------------------------

def load_class_d():
    """Concatenate any results/class_d_group{1,2,3}.csv. Returns None if none exist."""
    paths = sorted(glob.glob(os.path.join(R, "class_d_group*.csv")))
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        d["source"] = os.path.basename(p)
        frames.append(d)
    if not frames:
        return None
    cd = pd.concat(frames, ignore_index=True)
    cd["auroc"] = pd.to_numeric(cd["auroc"], errors="coerce")
    return cd.dropna(subset=["auroc"])


def class_d_summary(cd):
    """Mean AUROC per (method, arm) across the Class-D appendix sweep."""
    g = cd.groupby(["method", "arm"])
    out = pd.DataFrame({
        "mean_auroc": g["auroc"].mean(), "std_auroc": g["auroc"].std(),
        "n": g["auroc"].count(),
    }).sort_values("mean_auroc", ascending=False)
    return out


def class_d_tex(summary, path):
    lines = ["% auto-generated from class_d_group*.csv",
             "\\begin{table}[t]\\centering",
             "\\caption{Class-D appendix detectors: mean AUROC by evaluation arm "
             "(these detectors break the frozen-backbone fair comparison and are "
             "reported separately).}",
             "\\label{tab:class_d_appendix}",
             "\\small\\begin{tabular}{llrrr}\\toprule",
             "Method & Arm & Mean AUROC & Std & n \\\\\\midrule"]
    for (m, arm), r in summary.iterrows():
        std = "" if pd.isna(r["std_auroc"]) else f"{r['std_auroc']:.3f}"
        lines.append(f"{m.replace('_','-')} & {str(arm).replace('_','-')} & "
                     f"{r['mean_auroc']:.3f} & {std} & {int(r['n'])} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    df = load()
    n_ds = df["dataset"].nunique()
    lines = ["# TSB-StreamingAD Benchmark — Findings", "",
             f"Datasets evaluated: {n_ds} (U: {df[df.split=='U'].dataset.nunique()}, "
             f"M: {df[df.split=='M'].dataset.nunique()}). "
             f"Detectors: {df['method'].nunique()}. Completed runs: {len(df)}.", ""]

    overall = ranking(df)
    tex_table(overall, os.path.join(T, "tsb_main.tex"),
              "Overall mean AUROC across all TSB-StreamingAD datasets (higher is better).", "tab:tsb_main")
    lines += ["## Overall ranking (mean AUROC across all datasets)", ""]
    lines += [overall.round(3).to_string()]

    # per split
    for sp in ["U", "M"]:
        sub = df[df.split == sp]
        if len(sub):
            lines += ["", f"## TSB-StreamingAD-{sp} ranking", "", ranking(sub).round(3)["mean_auroc"].to_string()]

    # per category
    lines += ["", "## Mean AUROC by category (DRIFT / OOD / STABLE)", ""]
    piv_cat = df.pivot_table(index="method", columns="category", values="auroc", aggfunc="mean")
    lines += [piv_cat.round(3).to_string()]

    # per normalization (dichotomy)
    lines += ["", "## Mean AUROC by normalization (global vs per_series) — dichotomy test", ""]
    piv_norm = df.pivot_table(index="method", columns="normalize", values="auroc", aggfunc="mean")
    lines += [piv_norm.round(3).to_string()]

    # which fail under streaming
    fail = overall[overall["mean_auroc"] <= 0.5]
    lines += ["", "## Methods failing under streaming (mean AUROC <= 0.5, i.e. at/below chance or inverted)", ""]
    lines += [", ".join(f"{m} ({v:.3f})" for m, v in fail["mean_auroc"].items()) or "none"]

    # statistics
    lines += ["", "## Statistical test", "", f"- Friedman: {friedman(df)}"]

    # ---- Class-D appendix (optional) ----
    cd = load_class_d()
    if cd is not None and len(cd):
        cd_sum = class_d_summary(cd)
        class_d_tex(cd_sum, os.path.join(T, "class_d_appendix.tex"))
        lines += ["", "## Class-D appendix detectors (mean AUROC by arm)",
                  "", f"Rows: {len(cd)} across {cd['method'].nunique()} detectors "
                  f"({', '.join(sorted(cd['method'].unique()))}).", "",
                  cd_sum.round(3).to_string()]

    # ---- figures ----
    # heatmap method x category
    order = overall.index.tolist()
    plt.figure(figsize=(7, max(5, 0.32 * len(order) + 1)))
    sns.heatmap(piv_cat.reindex(order), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                cbar_kws={"label": "AUROC"})
    plt.title("Mean AUROC by method and category"); plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(os.path.join(F, f"tsb_heatmap_category.{e}"), dpi=150)
    plt.close()

    # split bar
    if df["split"].nunique() >= 1:
        piv_sp = df.pivot_table(index="method", columns="split", values="auroc", aggfunc="mean").reindex(order)
        piv_sp.plot(kind="barh", figsize=(8, max(5, 0.32 * len(order))))
        plt.axvline(0.5, color="k", ls="--", lw=1); plt.gca().invert_yaxis()
        plt.xlabel("Mean AUROC"); plt.title("Mean AUROC by method and split (U vs M)")
        plt.tight_layout()
        for e in ("png", "pdf"):
            plt.savefig(os.path.join(F, f"tsb_split_bar.{e}"), dpi=150)
        plt.close()

    # normalization dichotomy scatter
    if {"global", "per_series"}.issubset(set(piv_norm.columns)):
        plt.figure(figsize=(6, 6))
        plt.scatter(piv_norm["per_series"], piv_norm["global"], s=30)
        for m in piv_norm.index:
            plt.annotate(m, (piv_norm.loc[m, "per_series"], piv_norm.loc[m, "global"]),
                         fontsize=6, xytext=(2, 2), textcoords="offset points")
        plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlim(0, 1); plt.ylim(0, 1)
        plt.xlabel("Mean AUROC (per_series norm)"); plt.ylabel("Mean AUROC (global norm)")
        plt.title("Per-series vs global normalization per method")
        plt.tight_layout()
        for e in ("png", "pdf"):
            plt.savefig(os.path.join(F, f"tsb_norm_dichotomy.{e}"), dpi=150)
        plt.close()

    # efficiency
    plt.figure(figsize=(7, 5))
    eff = overall.dropna(subset=["mean_inference_ms"])
    plt.scatter(eff["mean_inference_ms"], eff["mean_auroc"], s=35)
    for m, r in eff.iterrows():
        plt.annotate(m, (r["mean_inference_ms"], r["mean_auroc"]), fontsize=6,
                     xytext=(3, 3), textcoords="offset points")
    plt.xscale("log"); plt.xlabel("Mean inference (ms/sample, log)"); plt.ylabel("Mean AUROC")
    plt.title("Accuracy vs inference cost"); plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(os.path.join(F, f"tsb_efficiency.{e}"), dpi=150)
    plt.close()

    tex_table(ranking(df[df.split == "U"]), os.path.join(T, "tsb_by_split.tex"),
              "Mean AUROC on TSB-StreamingAD-U (univariate streaming).", "tab:tsb_u")

    with open(os.path.join(R, "findings.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(str(x) for x in lines) + "\n")
    print("\n".join(str(x) for x in lines[:60]))
    print("\nWrote findings.md + tables + figures")


if __name__ == "__main__":
    main()
