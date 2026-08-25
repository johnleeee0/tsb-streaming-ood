# Compilation Log

**Author:** Stylianos Giannoulis · Aristotle University of Thessaloniki, MSc Data and Web Science · Supervisor: John Paparrizos
**Date:** 2026-06-24

## Status: FINAL — main.pdf produced (27 pages) — "Benchmarking OOD Methods for Time Series"

**Update 3 (2026-06-24):** Manuscript rewritten by the writer agent into the full TSB-StreamingAD-M/-U
benchmark thesis (Introduction, Related Work, Datasets, Methods, Experimental Setup, Results,
Discussion, Conclusions + appendix). Compiles cleanly with Tectonic to **27 pages**, **0 unresolved
citations/refs**, all real numbers (Friedman χ²=61.15 p=8.5e-6; Mahalanobis 0.874; TSB-M/U), stale
figures from earlier drafts confirmed absent. Tables `\input` from `tables/tsb_main.tex`,
`tsb_by_split.tex`, `ablation_deltas.tex`; figures from `figures/tsb_*.png`. Compile:
`C:\THESIS\.tools\tectonic.exe main.tex` (the Fontconfig stderr line is a harmless warning).



**Update 2 (2026-06-24):** Added Section "Method Verification and the Limits of Implementation
Fidelity" with the original-vs-enhanced ablation table (`tables/ablation_deltas.tex`) and the
regime-dependent-orientation negative result; updated the abstract. Recompiles cleanly to 17 pages,
0 unresolved citations. Fixed a nested-`table` error (the generated ablation table file already
provides its own `table` environment, so it is `\input` directly, not wrapped).



**Update 2026-06-24:** A LaTeX engine (Tectonic 0.15.0) was installed at
`C:\THESIS\.tools\tectonic.exe`. After adding the missing `\usepackage{natbib}`, the thesis
compiles cleanly to `main.pdf` (16 pages, 582 KiB). Verification: References section present,
all citations resolved (0 unresolved `[?]` markers, Gungor2025 et al. rendered), and the measured
Friedman result (χ²=13.857, p=0.0031) appears in the text. Compile command:

```
C:\THESIS\.tools\tectonic.exe main.tex
```

The 16-page length reflects honest, concise content (no padding); it will grow as Phase 2/3 add
the original-vs-enhanced ablation and new findings. The historical note below is retained.

---

## (Historical) Status: source complete; PDF not compiled on this machine

The LaTeX source of the thesis is complete and internally consistent:

- `main.tex` — full paper (abstract, introduction, related work, problem
  formulation, methods, experimental setup, results, analysis, conclusion,
  acknowledgements).
- `appendix.tex` — fidelity audit table, statistical comparison, ROC curves,
  score-distribution case study, reproducibility.
- `references.bib` — 44 BibTeX entries.
- `tables/` — `main_results.tex`, `statistical_tests.tex` (auto-generated from
  `results/all_results.csv`).
- `figures/` — `auroc_heatmap`, `mean_auroc_bar`, `efficiency_tradeoff`,
  `appendix_roc`, `appendix_violin_case` (PNG + PDF).

Consistency checks performed (all passed):
- Every `\includegraphics` target exists in `figures/`.
- Every `\input` target exists.
- Every `\cite{...}` key is defined in `references.bib`.

## Why no PDF here

**No LaTeX engine is installed on this machine** (`pdflatex`, `xelatex`, and
`tectonic` are all absent; verified). LaTeX engines are not pip-installable, so the
PDF could not be produced in this environment. This is stated plainly rather than
reporting a page count that does not exist.

## How to compile (any machine with TeX Live / MiKTeX)

```bash
cd C:\THESIS\paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or, with the single-binary engine (no full TeX install required):

```bash
tectonic main.tex
```

## Notes for the first compile

- `tables/main_results.tex` uses `\begin{table*}`; under the single-column
  `article` class this behaves like `table`. If the 8-column results table
  overflows the text width, wrap it in `\resizebox{\textwidth}{!}{...}` or switch
  the document to `[twocolumn]`. This is the only layout item likely to need a
  one-line adjustment.
- `cleveref` must be loaded after `hyperref` (it is).
- The bibliography style is `plainnat` (natbib); switch to a venue style as needed.
