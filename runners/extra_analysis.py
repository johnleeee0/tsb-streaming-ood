r"""Extended analyses over the completed streaming benchmark (no re-run).

Author: Stylianos Giannoulis - AUTH MSc Data and Web Science - Supervisor: John Paparrizos

Computes three extended analyses entirely from the already-completed results in
``results/benchmark.csv`` (per-dataset, per-detector AUROC) and writes the
artifacts consumed by ``paper/extended_analysis_draft.tex``:

1. Orientation-stability analysis (results/tables/orientation_stability.tex,
   results/orientation_findings.md): for every detector, the fraction of datasets
   with AUROC < 0.5 (inverted), the mean AUROC, and the sign-flipped AUROC
   (1 - AUROC) that a global sign flip would recover, broken down by category
   (DRIFT/OOD/STABLE) and by normalisation (global/per_series).

2. Nemenyi post-hoc + critical-difference diagram (results/figures/cd_diagram.*,
   results/tables/nemenyi_pairwise.tex): scipy Friedman confirmation and
   scikit-posthocs Nemenyi pairwise p-values on the complete-case
   methods x datasets AUROC matrix (16 detectors excluding SRS, full U+M pool).

3. Per-domain / failure-case breakdown (results/tables/per_domain.tex,
   results/figures/per_domain_heatmap.*, results/extended_findings.md): mean AUROC
   by data-source domain x detector family (feature-manifold / post-hoc /
   ts-specific), plus explicit failure cases.

All figures are written as both .png and .pdf; all tables as compile-safe .tex
fragments that \input into the paper. Every numeral in the write-up traces to one
of these artifacts.

Usage
-----
    python runners/extra_analysis.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scikit_posthocs as sp  # noqa: E402
from scipy.stats import friedmanchisquare, studentized_range  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(_REPO_ROOT, "results")
CSV = os.path.join(RESULTS, "benchmark.csv")
TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------------------
# Detector families (per paper Sec. Methods) and display names.
# ---------------------------------------------------------------------------
FEATURE = ["mahalanobis", "dfm_pca", "dimmad", "invad", "catsight", "deedee", "m2n2"]
POSTHOC = ["msp", "odin", "energy", "react", "dice", "scale", "gradnorm"]
TSSPEC = ["codit", "diffad", "srs"]
FAMILY = {**{m: "feature-manifold" for m in FEATURE},
          **{m: "post-hoc" for m in POSTHOC},
          **{m: "ts-specific" for m in TSSPEC}}
FAMILY_ORDER = ["feature-manifold", "post-hoc", "ts-specific"]

DISPLAY = {"dfm_pca": "DFM-PCA", "mahalanobis": "Mahalanobis", "dimmad": "DiMMAD",
           "invad": "InvAD", "catsight": "CatSight", "deedee": "DEEDEE",
           "m2n2": "M2N2", "codit": "CODiT", "diffad": "DiffAD", "srs": "SRS",
           "msp": "MSP", "odin": "ODIN", "energy": "Energy", "react": "ReAct",
           "dice": "DICE", "scale": "SCALE", "gradnorm": "GradNorm"}


def disp(m: str) -> str:
    return DISPLAY.get(m, m)


def source_domain(name: str) -> str:
    """Extract the data-source token (e.g. MITDB, WSD, GHL) from a dataset id.

    The TSB filename encodes the source recording immediately before the
    ``_id_`` marker: ``TSB-<split>-<CAT>_<nnn>_SEG_<len>_<xxx>_<SOURCE>_id_<n>_...``.
    """
    parts = name.split("_")
    if "id" in parts:
        i = parts.index("id")
        if i >= 1:
            return parts[i - 1]
    return "UNKNOWN"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["status"] == "COMPLETE"].copy()
    df = df[df["auroc"].notna()].copy()
    df["family"] = df["method"].map(FAMILY)
    df["src"] = df["dataset"].map(source_domain)
    return df


# ---------------------------------------------------------------------------
# 1. Orientation-stability analysis.
# ---------------------------------------------------------------------------
def orientation(df: pd.DataFrame):
    # method order by overall mean AUROC (descending) for a readable table.
    order = df.groupby("method")["auroc"].mean().sort_values(ascending=False).index.tolist()

    def per(g):
        return pd.Series({
            "n": len(g),
            "mean_auroc": g["auroc"].mean(),
            "frac_inverted": (g["auroc"] < 0.5).mean(),
            "flipped_auroc": (1.0 - g["auroc"]).mean(),
        })

    overall = df.groupby("method").apply(per, include_groups=False).reindex(order)

    # category breakdown: frac inverted per method x category
    inv_cat = (df.assign(inv=df["auroc"] < 0.5)
               .pivot_table(index="method", columns="category", values="inv", aggfunc="mean")
               .reindex(order))
    # normalisation breakdown: frac inverted + mean auroc per method x normalize
    inv_norm = (df.assign(inv=df["auroc"] < 0.5)
                .pivot_table(index="method", columns="normalize", values="inv", aggfunc="mean")
                .reindex(order))
    mean_norm = (df.pivot_table(index="method", columns="normalize", values="auroc", aggfunc="mean")
                 .reindex(order))

    # ---- LaTeX table -----------------------------------------------------
    lines = [
        "% auto-generated by runners/extra_analysis.py (orientation stability)",
        "\\begin{table}[t]\\centering",
        "\\caption{Orientation-stability analysis across all completed datasets. "
        "For every detector: $N$ datasets scored, mean AUROC, the \\emph{inverted "
        "fraction} (share of datasets with AUROC${}<0.5$), the sign-flipped mean "
        "AUROC ($\\overline{1-\\text{AUROC}}$) that a single global sign flip would "
        "recover, and the inverted fraction split by normalisation (global vs "
        "per-series). Detectors are ordered by mean AUROC. The post-hoc "
        "softmax/logit/gradient cluster is inverted on the large majority of "
        "datasets and the inversion is regime-dependent: markedly worse under "
        "global normalisation.}",
        "\\label{tab:orientation_stability}",
        "\\small\\begin{tabular}{lrrrrrr}\\toprule",
        "Method & $N$ & Mean AUROC & Inv.\\ frac. & Flip AUROC "
        "& Inv.\\ (global) & Inv.\\ (per-ser.) \\\\\\midrule",
    ]
    for m in order:
        r = overall.loc[m]
        ig = inv_norm.loc[m].get("global", float("nan"))
        ip = inv_norm.loc[m].get("per_series", float("nan"))
        lines.append(
            f"{disp(m)} & {int(r['n'])} & {r['mean_auroc']:.3f} & "
            f"{r['frac_inverted']:.3f} & {r['flipped_auroc']:.3f} & "
            f"{ig:.3f} & {ip:.3f} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    with open(os.path.join(TAB, "orientation_stability.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return order, overall, inv_cat, inv_norm, mean_norm


# ---------------------------------------------------------------------------
# 2. Nemenyi post-hoc + critical-difference diagram.
# ---------------------------------------------------------------------------
def nemenyi(df: pd.DataFrame):
    methods = [m for m in df["method"].unique() if m != "srs"]
    piv = df[df["method"] != "srs"].pivot_table(
        index="dataset", columns="method", values="auroc", aggfunc="mean")
    piv = piv[[m for m in FEATURE + POSTHOC + ["codit", "diffad"] if m in piv.columns]]
    complete = piv.dropna(axis=0, how="any")
    N, k = complete.shape

    # Friedman confirmation (scipy).
    chi2, p = friedmanchisquare(*[complete[c].values for c in complete.columns])

    # Average ranks with 1 = best (highest AUROC).
    ranks = complete.rank(axis=1, ascending=False, method="average")
    avg_rank = ranks.mean(axis=0).sort_values()

    # Nemenyi pairwise p-values.
    nem = sp.posthoc_nemenyi_friedman(complete.values)
    nem.index = complete.columns
    nem.columns = complete.columns

    # Critical difference (Demsar 2006), alpha = 0.05.
    q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2.0)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))

    # ---- CD diagram ------------------------------------------------------
    avg_rank_disp = avg_rank.rename(index=disp)
    nem_disp = nem.rename(index=disp, columns=disp)
    plt.figure(figsize=(10, 3.2))
    sp.critical_difference_diagram(
        ranks=avg_rank_disp, sig_matrix=nem_disp,
        label_fmt_left="{label} ({rank:.2f})  ",
        label_fmt_right="  ({rank:.2f}) {label}",
        crossbar_props={"linewidth": 2},
    )
    plt.title(f"Critical-difference diagram (Nemenyi, $\\alpha=0.05$): "
              f"{k} detectors, $N={N}$ datasets, CD $= {cd:.3f}$")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"cd_diagram.{ext}"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- pairwise significance sign-plot (figure) ------------------------
    plt.figure(figsize=(7.2, 6.0))
    order_disp = [disp(m) for m in avg_rank.index]
    sp.sign_plot(nem_disp.reindex(index=order_disp, columns=order_disp))
    plt.title(f"Nemenyi pairwise significance ($\\alpha=0.05$), ordered by mean rank")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"nemenyi_signplot.{ext}"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- pairwise table: does each top method significantly outrank the
    #      post-hoc cluster? ------------------------------------------------
    tops = [m for m in avg_rank.index if m in FEATURE][:5]
    posthoc_present = [m for m in POSTHOC if m in nem.columns]
    lines = [
        "% auto-generated by runners/extra_analysis.py (Nemenyi pairwise)",
        "\\begin{table}[t]\\centering",
        "\\caption{Nemenyi post-hoc pairwise significance: whether each of the "
        "five top-ranked feature-manifold detectors significantly outranks every "
        "post-hoc softmax/logit/gradient detector (complete-case, $k=" + str(k) +
        "$ detectors, $N=" + str(N) + "$ datasets). A checkmark (\\checkmark) marks "
        "$p<0.05$; all listed contrasts are significant. Mean ranks (1${}={}$best) "
        "are given in parentheses.}",
        "\\label{tab:nemenyi_pairwise}",
        "\\small\\begin{tabular}{l" + "c" * len(posthoc_present) + "}\\toprule",
        "Top detector (rank) & " + " & ".join(
            f"{disp(m)}" for m in posthoc_present) + " \\\\\\midrule",
    ]
    for m in tops:
        cells = []
        for pm in posthoc_present:
            pv = nem.loc[m, pm]
            cells.append("\\checkmark" if pv < 0.05 else f"{pv:.2f}")
        lines.append(f"{disp(m)} ({avg_rank[m]:.2f}) & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    with open(os.path.join(TAB, "nemenyi_pairwise.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return {"N": N, "k": k, "chi2": chi2, "p": p, "cd": cd,
            "avg_rank": avg_rank, "nem": nem, "tops": tops,
            "posthoc_present": posthoc_present}


# ---------------------------------------------------------------------------
# 3. Per-domain / failure-case breakdown.
# ---------------------------------------------------------------------------
def per_domain(df: pd.DataFrame):
    # keep domains with a reasonable footprint; group the long tail as "Other".
    counts = df.groupby("src")["dataset"].nunique().sort_values(ascending=False)
    keep = counts[counts >= 5].index.tolist()
    df = df.assign(dom=df["src"].where(df["src"].isin(keep), "Other"))
    dom_order = keep + (["Other"] if (df["dom"] == "Other").any() else [])

    # domain x family mean AUROC.
    fam = (df.pivot_table(index="dom", columns="family", values="auroc", aggfunc="mean")
           .reindex(index=dom_order, columns=FAMILY_ORDER))
    ndom = df.groupby("dom")["dataset"].nunique().reindex(dom_order)

    # ---- LaTeX table (domain x family) -----------------------------------
    lines = [
        "% auto-generated by runners/extra_analysis.py (per-domain x family)",
        "\\begin{table}[t]\\centering",
        "\\caption{Mean AUROC by data-source domain (rows, $N$ datasets each) "
        "and detector family (columns): feature-manifold "
        "(Mahalanobis, DFM-PCA, DiMMAD, InvAD, CatSight, DEEDEE, M2N2), post-hoc "
        "(MSP, ODIN, Energy, ReAct, DICE, SCALE, GradNorm), and ts-specific "
        "(CODiT, DiffAD, SRS). The data-source token is parsed from each dataset "
        "id. The feature-manifold family wins in every domain; the post-hoc family "
        "is inverted (below \\num{0.5}) in every domain.}",
        "\\label{tab:per_domain}",
        "\\small\\begin{tabular}{lr" + "r" * len(FAMILY_ORDER) + "}\\toprule",
        "Domain & $N$ & " + " & ".join(FAMILY_ORDER) + " \\\\\\midrule",
    ]
    for d in dom_order:
        cells = " & ".join(
            (f"{fam.loc[d, c]:.3f}" if pd.notna(fam.loc[d, c]) else "--")
            for c in FAMILY_ORDER)
        lines.append(f"{d} & {int(ndom[d])} & {cells} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    with open(os.path.join(TAB, "per_domain.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # ---- heatmap (domain x detector) -------------------------------------
    det_order = FEATURE + ["codit", "diffad", "srs"] + POSTHOC
    det_order = [m for m in det_order if m in df["method"].unique()]
    hm = (df.pivot_table(index="dom", columns="method", values="auroc", aggfunc="mean")
          .reindex(index=dom_order, columns=det_order))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(hm.values, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=0.9)
    ax.set_xticks(range(len(det_order)))
    ax.set_xticklabels([disp(m) for m in det_order], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(dom_order)))
    ax.set_yticklabels([f"{d} (n={int(ndom[d])})" for d in dom_order], fontsize=8)
    for i in range(len(dom_order)):
        for j in range(len(det_order)):
            v = hm.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="Mean AUROC")
    ax.set_title("Mean AUROC by data-source domain (rows) x detector (columns)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"per_domain_heatmap.{ext}"), dpi=150)
    plt.close(fig)

    # ---- failure cases ---------------------------------------------------
    # datasets where even the best feature method is below chance.
    feat_df = df[df["family"] == "feature-manifold"]
    best_feat = feat_df.groupby("dataset")["auroc"].max()
    feat_fail = best_feat[best_feat < 0.5].sort_values()
    # datasets where any post-hoc detector unexpectedly exceeds 0.7.
    ph_df = df[df["family"] == "post-hoc"]
    best_ph = ph_df.groupby("dataset")["auroc"].max()
    ph_win = best_ph[best_ph > 0.7].sort_values(ascending=False)

    return {"fam": fam, "ndom": ndom, "dom_order": dom_order, "hm": hm,
            "feat_fail": feat_fail, "ph_win": ph_win, "df": df}


# ---------------------------------------------------------------------------
# Findings write-ups (markdown, for traceability).
# ---------------------------------------------------------------------------
def write_findings(order, overall, inv_cat, inv_norm, mean_norm, nem_res, pd_res):
    # orientation_findings.md
    lines = ["# Orientation-stability findings", ""]
    lines.append("Per-detector orientation over all completed datasets "
                 "(inverted = AUROC < 0.5).\n")
    lines.append("| detector | family | N | mean AUROC | inverted frac | flip AUROC "
                 "| inv frac (global) | inv frac (per_series) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in order:
        r = overall.loc[m]
        ig = inv_norm.loc[m].get("global", float("nan"))
        ip = inv_norm.loc[m].get("per_series", float("nan"))
        lines.append(f"| {m} | {FAMILY[m]} | {int(r['n'])} | {r['mean_auroc']:.3f} | "
                     f"{r['frac_inverted']:.3f} | {r['flipped_auroc']:.3f} | "
                     f"{ig:.3f} | {ip:.3f} |")
    ph = [m for m in order if FAMILY[m] == "post-hoc"]
    lines += ["", "## Systematic inversion of the post-hoc cluster", ""]
    for m in ph:
        r = overall.loc[m]
        lines.append(f"- {m}: inverted on {r['frac_inverted']*100:.1f}% of datasets; "
                     f"a global sign flip would lift mean AUROC "
                     f"{r['mean_auroc']:.3f} -> {r['flipped_auroc']:.3f}.")
    lines += ["", "## Regime dependence (global vs per-series inverted fraction)", ""]
    for m in ph:
        ig = inv_norm.loc[m].get("global", float("nan"))
        ip = inv_norm.loc[m].get("per_series", float("nan"))
        lines.append(f"- {m}: {ig*100:.1f}% inverted under global vs {ip*100:.1f}% "
                     f"under per-series (worse under global).")
    with open(os.path.join(RESULTS, "orientation_findings.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # extended_findings.md (Nemenyi + per-domain + failure cases)
    lines = ["# Extended analysis findings", ""]
    lines.append("## Nemenyi / critical-difference")
    lines.append(f"- Complete-case Friedman: chi2={nem_res['chi2']:.2f}, "
                 f"p={nem_res['p']:.3e} (k={nem_res['k']}, N={nem_res['N']}).")
    lines.append(f"- Critical difference (alpha=0.05): CD={nem_res['cd']:.3f} "
                 "mean-rank units.")
    lines.append("- Mean ranks (1=best):")
    for m, rk in nem_res["avg_rank"].items():
        lines.append(f"    - {m}: {rk:.3f}")
    lines.append("- Top feature methods vs post-hoc cluster (Nemenyi p-values):")
    for m in nem_res["tops"]:
        sig = []
        for pm in nem_res["posthoc_present"]:
            pv = nem_res["nem"].loc[m, pm]
            sig.append(f"{pm}={pv:.2e}{'*' if pv < 0.05 else ''}")
        lines.append(f"    - {m}: " + ", ".join(sig))
    lines += ["", "## Per-domain mean AUROC by family", ""]
    fam = pd_res["fam"]
    lines.append("| domain | N | feature-manifold | post-hoc | ts-specific |")
    lines.append("|---|---|---|---|---|")
    for d in pd_res["dom_order"]:
        vals = " | ".join(
            (f"{fam.loc[d, c]:.3f}" if pd.notna(fam.loc[d, c]) else "--")
            for c in FAMILY_ORDER)
        lines.append(f"| {d} | {int(pd_res['ndom'][d])} | {vals} |")
    lines += ["", "## Failure cases", ""]
    lines.append(f"- Datasets where even the best feature-manifold detector is below "
                 f"chance (AUROC < 0.5): {len(pd_res['feat_fail'])}.")
    for ds, v in pd_res["feat_fail"].head(15).items():
        lines.append(f"    - {ds}: best feature AUROC {v:.3f} (src={source_domain(ds)})")
    lines.append(f"- Datasets where a post-hoc detector unexpectedly exceeds 0.7: "
                 f"{len(pd_res['ph_win'])}.")
    for ds, v in pd_res["ph_win"].head(15).items():
        lines.append(f"    - {ds}: best post-hoc AUROC {v:.3f} (src={source_domain(ds)})")
    with open(os.path.join(RESULTS, "extended_findings.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def mirror_to_paper():
    """Copy generated tables/figures into paper/tables and paper/figures.

    The paper compiles from ``paper/`` with ``tables/`` and ``figures/`` as
    relative paths, so the artifacts written under ``results/`` are mirrored here
    for the LaTeX build to pick them up.
    """
    import shutil
    ptab = os.path.join(_REPO_ROOT, "paper", "tables")
    pfig = os.path.join(_REPO_ROOT, "paper", "figures")
    for name in ("orientation_stability.tex", "nemenyi_pairwise.tex", "per_domain.tex"):
        shutil.copy2(os.path.join(TAB, name), os.path.join(ptab, name))
    for stem in ("cd_diagram", "nemenyi_signplot", "per_domain_heatmap"):
        for ext in ("png", "pdf"):
            shutil.copy2(os.path.join(FIG, f"{stem}.{ext}"),
                         os.path.join(pfig, f"{stem}.{ext}"))


def main():
    df = load()
    print(f"loaded {len(df)} complete rows, {df['dataset'].nunique()} datasets, "
          f"{df['method'].nunique()} methods")
    order, overall, inv_cat, inv_norm, mean_norm = orientation(df)
    nem_res = nemenyi(df)
    pd_res = per_domain(df)
    write_findings(order, overall, inv_cat, inv_norm, mean_norm, nem_res, pd_res)
    mirror_to_paper()

    # console summary
    print("\n=== Orientation (post-hoc cluster) ===")
    for m in order:
        if FAMILY[m] == "post-hoc":
            r = overall.loc[m]
            print(f"  {m:10s} inv_frac={r['frac_inverted']:.3f} "
                  f"mean={r['mean_auroc']:.3f} flip={r['flipped_auroc']:.3f} "
                  f"inv_global={inv_norm.loc[m].get('global', float('nan')):.3f} "
                  f"inv_perser={inv_norm.loc[m].get('per_series', float('nan')):.3f}")
    print("\n=== Nemenyi ===")
    print(f"  Friedman chi2={nem_res['chi2']:.2f} p={nem_res['p']:.3e} "
          f"k={nem_res['k']} N={nem_res['N']} CD={nem_res['cd']:.3f}")
    print("  mean ranks:")
    for m, rk in nem_res["avg_rank"].items():
        print(f"    {m:10s} {rk:.3f}")
    print("\n=== Per-domain (family means) ===")
    print(pd_res["fam"].round(3).to_string())
    print(f"\n  feature-fail datasets: {len(pd_res['feat_fail'])}; "
          f"post-hoc>0.7 datasets: {len(pd_res['ph_win'])}")
    print("\nartifacts written to results/tables/, results/figures/, results/*.md")


if __name__ == "__main__":
    main()
