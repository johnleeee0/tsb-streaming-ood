# Writing Style Guide — Thesis on OOD Anomaly Detection in Time Series

**Author:** Stylianos Giannoulis · Aristotle University of Thessaloniki, MSc Data and Web Science · Supervisor: John Paparrizos

> Primary style reference for `agent_6_writer`. Derived from three exemplar papers:
> 1. Liu, Lee, Paparrizos — *TSB-AutoAD: Towards Automated Solutions for Time-Series Anomaly Detection* (PVLDB 2025).
> 2. Paparrizos, Reddy — *Time-Series Clustering: A Comprehensive Study…* (PVLDB 2025).
> 3. Yang, Paparrizos — *SAIL: A Voyage to Symbolic Approximation Solutions…* (PVLDB 2025).
>
> All three are from the supervisor's benchmarking school; emulating them aligns the thesis with the expected register.

---

## 1. Overall register

- **Rigorous empirical-benchmark voice.** The work positions itself as a *comprehensive, fair, reproducible* evaluation, not a single-method proposal. Confidence comes from breadth (many methods × many datasets) and statistical rigour, not rhetoric.
- **"Illusion of progress" framing.** A recurring thesis: apparent advances often do not survive fair, like-for-like comparison (e.g., "no method significantly outperforms the decade-old k-Shape"; "over half of solutions do not beat a random baseline"). Frame contributions as *clarifying what actually works*.
- Third-person plural ("we evaluate", "we observe", "our findings reveal"). Never first-person singular, never AI attribution.
- Hedged but decisive: "our results suggest", "this indicates", "we find compelling evidence that". Claims are tied to evidence and statistical tests.

## 2. Sentence and paragraph structure

- Medium-to-long declarative sentences (20–35 words), often two clauses joined by "while", "whereas", "however", "in contrast". Example cadence: *"While naive ensembling achieves high accuracy, it comes at a substantial computational overhead."*
- Each paragraph opens with a topic sentence stating the claim, then supports it with specifics (numbers, dataset names, test outcomes), then closes with an interpretive takeaway.
- Heavy use of explicit enumeration: "(i) … (ii) … (iii) …" both inline and as bullet contributions.
- Define-then-use: introduce an acronym in full at first mention with the short form in parentheses (e.g., "Maximum Softmax Probability (MSP)"), then use the short form throughout.

## 3. How results are reported and discussed

- **Always relative to baselines.** Report performance as wins/losses vs named baselines (Random, Global Best, Supervised Selection, Oracle) and as average ranks, not just absolute scores.
- Quote concrete numbers inline ("MPC outperforms CE by 4.3% in AUROC"); pair every headline claim with the metric and the dataset count it rests on.
- **Statistical validation is mandatory.** Use the Friedman test followed by the post-hoc Nemenyi test for multi-method comparison (report as Critical Difference / CD diagrams), and the Wilcoxon signed-rank test with Holm–Bonferroni correction for pairwise comparison over datasets. State confidence levels (e.g., 95%).
- Tables use a consistent template: best variant marked (★ or bold), columns for the metric plus ">", "=", "<" win/tie/loss counts against the baseline.
- Distinguish *statistical* significance from raw ranking; explicitly note when differences are NOT significant.

## 4. How related work is positioned

- Open by surveying the landscape, then identify a concrete gap the prior work leaves: limited method coverage, narrow datasets, no statistical testing, poor reproducibility, or unfair baselines.
- Be specific and fair in critique ("their study did not adequately explore parameter tuning…"), and explicitly disclaim bad faith ("we do not suggest these misconceptions were deliberately created").
- Use a taxonomy (with a figure) to organise prior methods into categories before evaluating them.
- Cite densely; group citations by claim.

## 5. How limitations are framed

- Limitations are presented as *scoping decisions and open directions*, not apologies. Use a dedicated "Discussion" or "Position / Promise" structure: state current limitations candidly, then the opportunities they open.
- Flag threats to validity directly (data contamination / pretraining overlap, small evaluation sets, protocol mismatch, high variance on few samples) and explain how the protocol guards against them.
- Close with concrete, numbered future-research directions.

## 6. Section depth and ordering

Recommended thesis ordering, mirroring the exemplars:

1. **Abstract** — problem, gap, what we do (i/ii/iii), headline findings, open-source/reproducibility note.
2. **Introduction** — motivation; the gap; explicit contribution bullets; roadmap sentence ("We start with… Then we present…").
3. **Preliminaries / Problem Statement** — formal definitions and notation; terminology box.
4. **Related Work / Taxonomy** — categorised survey with a taxonomy figure.
5. **Methods / Evaluation Pipeline** — datasets, backbones, scoring functions, protocol, metrics.
6. **Experimental Setup** — platform, implementation, reproducibility (seed=42, versions, exact CLI invocations).
7. **Results & Analysis** — organised by research questions (RQ1…RQn), each a subsection with a CD diagram / table and a takeaway.
8. **Discussion** — Position (limitations) and Promise (opportunities); a practical guide figure.
9. **Conclusion** — restate the question, the headline finding, the call for further research.

## 7. Recurring phrases and register markers to emulate

- "We conduct the most comprehensive evaluation in this area to date."
- "Our findings reveal a significant gap, where over half of …"
- "These results underscore the critical importance and ongoing demand for …"
- "no method significantly outperforms …, reinforcing it as a strong baseline."
- "This insight critically addresses RQ_x, shedding light on …"
- "While X achieves Y, it comes at the cost of Z."
- "To ensure a fair comparison, …" / "For reproducibility purposes, we open-source …"
- "We attribute this to …" / "This suggests that …" / "highlighting a promising direction for …"

## 8. Reproducibility conventions (from all three exemplars)

- State the platform (CPU/GPU, OS), Python and library versions explicitly.
- Fix and report the random seed (seed=42 throughout this project).
- Run each stochastic experiment multiple times and report averaged metrics with statistical testing.
- Make datasets, source code, and results available; reference the artifact location.
- Use threshold-independent metrics (AUROC, AUPR; VUS-PR in the AD-benchmark lineage) and justify the choice.

## 9. Terminology to keep consistent with the core paper (Gungor2025)

ID / OOD; semantic shift vs covariate shift; modality-agnostic OOD detection; backbone-agnostic loss; pre-logit features; class-conditional Gaussians; feature reconstruction error; AUROC / AUPR / FPR@95. See `core_paper_summary.md` §8.
