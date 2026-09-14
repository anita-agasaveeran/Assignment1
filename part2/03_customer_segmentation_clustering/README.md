# Credit-Card Portfolio Segmentation — a CRISP-DM study with an automated pipeline search

Unsupervised segmentation of ~9,000 active credit-card holders, run end-to-end under
CRISP-DM, where the modelling phase is a **hill-climbing search over 3.4M clustering
pipelines** rather than three algorithms tried by hand. Output is a single run record
(`artifacts/run.json`) and a dashboard rendered entirely from it.

**Dataset** — [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
(`arjunbhasin2013/ccdata`), 8,950 accounts × 17 behavioural features over six months.
Downloaded anonymously via `kagglehub`; no API token required. A copy is cached at
`data/raw/cc_general.csv` so the study re-runs offline.

**Dashboard** — https://claude.ai/code/artifact/478da2d1-3afd-41fd-a826-e7941cf7a8e2
(also `dashboard/index.html`, self-contained, open in any browser).

---

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m ccseg.cli run          # full study (~4.5 min)
.venv/bin/python dashboard/build_dashboard.py             # render the dashboard
.venv/bin/python -m pytest tests/ -q                      # 19 invariants
```

`--quick` runs a 60-evaluation smoke test in about 40 seconds. `--objective`
switches between the calibrated composite and any single validity index;
`--strategy steepest` swaps first-improvement acceptance for steepest ascent.

---

## What is actually here

### The search (Phase 4)

Combined Algorithm Selection and Hyperparameter optimisation — the CASH formulation
from Auto-WEKA (Thornton et al., 2013), carried to clustering by AutoML4Clust
(Tschechlov et al., 2021) — over a **conditional** space of 20 dimensions:
feature view, imputation, winsorising, variance-stabilising transform, scaler,
dimensionality reduction, then ten clustering algorithms across four families with
their own conditional hyperparameters.

The optimiser is an iterated local search: random restarts, single-dimension moves,
first-improvement or steepest-ascent acceptance, plateau traversal, a tabu memory
(Glover, 1986), and an adaptive neighbourhood that biases sampling toward dimensions
that have paid off. Two-level memoisation caches the objective on the canonical config
hash and the preprocessed matrix on the preprocessing sub-config, so the ~40% of moves
that change only a model dimension reuse the expensive transform chain.

**Random search over the same space and budget is the control arm.** A search reported
without one is a number, not a finding.

### Three findings that changed the design

**1. Scoring a candidate in its own space is objective-gaming.** An early run selected
`pca_n2` and reported a silhouette of **0.68** that was **0.195** when the same labels
were scored against the real features. Projecting to two components discards exactly
the variance that would have made the clusters overlap. Every candidate is now scored
in one fixed neutral space (17 features, Yeo-Johnson, standardised, no reduction) that
no candidate can influence. Across a full run, **93% of trials** would have scored higher
against themselves, by a mean of 0.22 silhouette points, and the inflation decays
monotonically with dimensionality.

**2. Unconstrained validity indices select degenerate partitions.** The objective is
maximised by one blob holding most of the book plus a few outlier pockets — silhouette
1.0, business value nil. Constraint handling is the static-penalty method
(Coello Coello, 2002): feasible solutions strictly dominate infeasible ones, with a
graded penalty inside the infeasible region so the climber can still find its way back.

**3. The objective-optimal model failed the brief — and nothing satisfies it.** The
search champion (HDBSCAN) maximised the objective and contained a segment with a
bootstrap Jaccard of 0.39, inside Hennig's "dissolved" band. CRISP-DM draws an arrow
from Evaluation back to Business Understanding for exactly this case, and the study
walks it: ten shortlisted pipelines were graded against the whole charter and **none
satisfies every criterion**, because separation and stability are in direct tension on
this data — everything clearing silhouette 0.25 abstains on ~25% of the book and leaves
a segment that dissolves; everything reproducible lands just under the bar. The study
deploys the reproducible KMeans k=3 model (per-segment Jaccard 0.94–0.98, ARI 0.955),
**records SC-1 as failed against the original bar**, and proposes an evidence-backed
threshold revision (0.237 observed ceiling) rather than presenting it as a pass.

### Evaluation (Phase 5)

Internal indices (Rousseeuw 1987; Calinski–Harabasz 1974; Davies–Bouldin 1979, chosen
on the evidence of Arbelaitz et al. 2013) are necessary and nowhere near sufficient, so:

- **Stability** — 40 resampling replicates refitting the *entire* pipeline, compared to
  the reference partition by ARI (Hubert & Arabie, 1985) and NMI, plus per-segment
  bootstrap Jaccard (Hennig, 2007) because a partition can be stable on average while
  one segment dissolves.
- **Gap statistic** (Tibshirani et al., 2001) as an independent read on *k*.
- **k-sweep** holding preprocessing fixed and clustering with KMeans (which honours every
  requested *k* — Birch does not, and an early version of this panel had seven identical
  rows labelled k=6..12), exposing the small-*k* bias Milligan & Cooper documented in 1985.
- **Transfer study** — the search runs on a subsample, which assumes the ranking
  transfers. Measured on a probe stratified across the whole objective range
  (ρ = 0.80). Measuring it on the finalists alone gives a range-restricted,
  noise-dominated number; an earlier version reported ρ = −0.52 that way.
- **Baselines and an algorithm shootout**, so "the search helped" is a comparison rather
  than an assertion. Honest result: the search beats the best hand-built pipeline by
  0.013 objective — it found a good pipeline automatically, not a better one than an
  expert would.

### Deployment (Phase 6)

The artifact is a function, not a column. It carries its own feature engineering, so a
caller passes the raw source schema and cannot reimplement the ratios slightly wrong —
training/serving skew is the most common way a segmentation quietly stops working
(Sculley et al., 2015). Transductive algorithms get an induced nearest-centroid rule
whose **fidelity to the original labels is measured and reported**.

Monitoring reads three signals, because unsupervised learning has no delayed ground
truth: per-feature PSI against training deciles, segment-mix drift, and mean distance to
the assigned centroid. It ships with a **positive control** — a deliberately shifted book
that the monitor must alarm on. That control also exposes a real limitation: PSI stays
under 0.10 on `CASH_ADVANCE` even after a 75% inflation, because the column is 52% exact
zeros and its deciles collapse onto a point mass.

---

## Results (seed 20260901)

| | |
|---|---|
| Deployed pipeline | `engineered · median · Yeo-Johnson · standard · PCA(6) → KMeans k=3` |
| Segments | High-Spend Transactor 51.5% · Cash-Advance Reliant 27.7% · Revolving Borrower 20.8% |
| Silhouette (neutral 17-D space) | 0.237 (0.307 in the model's own 6-D space) |
| Bootstrap ARI | 0.955 · per-segment Jaccard 0.94–0.98 |
| Surrogate fidelity (depth-4 tree) | 97.8% |
| Scoring latency, p95 per record | 0.007 ms at batch 1000 |
| Charter | 5/6 — SC-1 failed at 0.237 vs 0.25, revision proposed |
| Search | 706 evaluations · HC 60 s vs RS 63 s at equal budget · transfer ρ 0.80 |

## Layout

```
src/ccseg/
  config.py         Phase 1 charter, run configuration, seeds
  data.py           Phase 2 acquisition, profiling, data-quality audit, Hopkins
  prepare.py        Phase 3 fixed cleaning, feature engineering, the neutral space
  search_space.py   the conditional CASH space and its neighbourhood structure
  models.py         algorithm zoo + the inductive assigner deployment needs
  metrics.py        validity indices, calibration, the constrained objective
  autoresearch.py   hill climbing, random-search control, search analytics
  evaluate.py       stability, gap, k-sweep, baselines, transfer, charter compliance
  profiling.py      segment profiles, personas, surrogate rules, unit economics
  deploy.py         scoring artifact, latency, PSI drift monitor, model card
  research.py       31 citations mapped to the functions that implement them
  report.py         runs all six phases, emits artifacts/run.json
dashboard/          template + builder → index.html (single data source: run.json)
tests/              invariants whose violation would produce plausible-but-wrong numbers
artifacts/          run.json, scored_customers.csv, segmentation_model_1.0.0.joblib
```

## Known limitations

- Six months of behaviour with no time index. No seasonality can be modelled and there
  is no held-out period — the monitor is the mitigation, not a fix.
- Reference-space silhouette sits near 0.24. This is weak-but-real structure, which is
  what behavioural financial data usually offers; anything much higher on this dataset
  should be checked for the self-scoring inflation described above.
- Unit economics are illustrative. The file carries no revenue fields, so segment value
  is modelled from behaviour with five stated rates. The ordering is far more robust
  than the levels.
- The dataset has no demographic or protected attributes, so disparate-impact testing
  **cannot be done from this file**. Credit limit and cash-advance reliance are plausible
  income proxies; a review against the issuer's own data is a precondition of production use.
