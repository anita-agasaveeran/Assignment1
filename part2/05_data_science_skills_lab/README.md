# CRISP-DM demonstration of 46 agent skills on two Kaggle datasets

Installs [`param087/agent-ml-skills`](https://github.com/param087/agent-ml-skills) (15 skills)
and [`nimrodfisher/data-analytics-skills`](https://github.com/nimrodfisher/data-analytics-skills)
(31 skills), then exercises **every one of the 46** across the full **CRISP-DM** lifecycle on
two of Kaggle's most-used datasets.

Nothing here is illustrative. Every number in every report is computed from the raw data by a
script in `scripts/`, and `python scripts/coverage_matrix.py` fails if any skill lacks a
physical evidence artifact.

---

## The datasets

| Dataset | Rows | Why this one | Used for |
|---|---:|---|---|
| **Titanic** (`kaggle/titanic`) | 891 x 12 | Kaggle's most-entered competition; the canonical supervised-classification benchmark | Track A - the 15 ML/MLOps skills |
| **Online Retail** (`carrie1/ecommerce-data`, = UCI 352) | 541,909 x 8 | The standard public dataset for RFM, cohort and retention analytics | Track B - the 31 analytics skills |

Both tracks run all six CRISP-DM phases in parallel after Phase 2.

---

## CRISP-DM phases and where each skill lands

```
Phase 1  Business Understanding   2 skills   briefs, plan, risks   (before any data is touched)
Phase 2  Data Understanding      10 skills   profile, warehouse, reconcile
Phase 3  Data Preparation         2 skills   clean + engineer, train-only statistics
Phase 4  Modeling                17 skills   Track A: pipeline -> tuning -> nets -> RAG -> LoRA
                                             Track B: cohorts -> segments -> funnel -> TS -> RCA
Phase 5  Evaluation               9 skills   synthesise, quantify, QA, peer review
Phase 6  Deployment               6 skills   serve, document, specify, retrospect
```

Full table with evidence links: **[`outputs/reports/skill_coverage_matrix.md`](outputs/reports/skill_coverage_matrix.md)**

---

## What the analysis actually found

**Track B - Online Retail**

- **12.9% of customers produce 62.7% of revenue.** 558 "Champion" accounts (RFM + k-means,
  k=4, silhouette 0.381) generate GBP 5.6M of GBP 8.9M.
- **34% of customers never place a second order** - 1,493 people, GBP 718K of un-realised
  second-order revenue. Median gap to a second order is 50 days, so the intervention window
  is real.
- **The 70% December revenue "collapse" is not a business event.** The extract ends on
  9 December. Pro-rated, December runs **+2.2%** against November. Order volume fell 71%
  while AOV moved +1.2%, and the fall is uniform across all 38 markets - which is not how
  demand shocks behave.
- **Two revenue definitions differed by 8.58%** (GBP 836,340). Decomposed to a **zero
  residual**: cancellations (-GBP 896,812) and unidentified customers (+GBP 1,733,152).

**Track A - Titanic**

- Holdout PR-AUC **0.791** against a cross-validated **0.867**. The holdout number is the one
  reported.
- The Optuna gain over the untuned model (+0.016) is **inside fold noise** (+/-0.022), so it is
  reported as *no measurable gain*, not as an improvement.
- **SMOTE was tested and rejected** - at a 1.6:1 ratio the delta was inside noise.
- Slice metrics expose the real problem the aggregate hides: **male-passenger recall is 0.417**
  versus 0.867 for women. This blocks promotion to any automated decision.

**Three negative results that best practice would have got wrong**

Each of these is something a competent practitioner would ship on reputation alone:

| Received wisdom | What measurement showed |
|---|---|
| SMOTE helps imbalanced data | No gain at 1.6:1; inside fold noise |
| Always tune hyperparameters | +0.016 PR-AUC, smaller than fold-to-fold variance |
| Always add a cross-encoder reranker to RAG | **Reduced** retrieval MRR by 0.028 on this corpus |

---

## Bugs found in the skill packs

Sixteen of the 36 bundled skill scripts did not run. All are patched in
[`patches/`](patches/) (unified diffs, applied to the installed copies).

| Bug | Files | Effect |
|---|---|---|
| `__main__` runs a hardcoded demo instead of calling `main()` | **11** | CLI args silently ignored; the script emits **fabricated demo numbers** while appearing to work |
| Bare `%` in an `argparse` help string | 4 sites / 3 files | Hard crash on Python 3.14 (`ValueError: badly formed help string`) |
| `.dt.to_period("MS")` - `MS` is a DateOffset, not a Period frequency | 1 | `cohort_builder.py` crashes; breaks the whole cohort chain |
| `sort_values` on a column-less empty frame | 1 | `KeyError` whenever no correlation pair clears the threshold |

The first is the serious one. A tool that silently answers about the *wrong data* is worse than
no tool: `segmentation_runner.py` cheerfully reported "LTV by PLAN, N=20" for a 4,338-customer
RFM segmentation, and `catalog_extractor.py` returned a fictional `orders_fact` table for every
database it was pointed at.

Both repos ship validators, but they check documentation structure, not script execution -
which is why these survived. (The `data-analytics-skills` validator also flags 25 of its own
skills for heading-vocabulary mismatches; that one is cosmetic.)

---

## Reproducing

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/download_data.py    # fetch both datasets, verify SHA-256 (data/ is git-ignored)
./.venv/bin/python scripts/run_all.py          # all phases (~4 min; --fast skips model downloads)
./.venv/bin/python scripts/coverage_matrix.py  # verify all 46 skills have evidence
```

The skills themselves live in `~/.claude/skills/`; `installed_skills/` is a verbatim snapshot
of that install with the patches applied, so the submission is self-contained.

Seed 42 throughout; both raw datasets SHA-256 hashed and the hash logged with every run;
143 dependencies pinned exactly; git SHA recorded per run.

---

## Layout

```
05_data_science_skills_lab/
├── README.md
├── requirements.txt              143 pinned deps
├── installed_skills/             snapshot of ~/.claude/skills (46 skills, patches applied)
├── data/
│   ├── raw/                      titanic.csv, online_retail.csv/.xlsx  (immutable)
│   └── processed/                retail.db (4-table star schema), splits, intermediates
├── scripts/
│   ├── common.py                 paths, seeding, hashing, plot styling
│   ├── download_data.py          fetch + SHA-256-verify both raw datasets
│   ├── run_all.py                one-command full pipeline
│   ├── coverage_matrix.py        verifies all 46 skills have evidence
│   ├── p1_business_understanding.py
│   ├── p0_build_warehouse.py     flat file -> normalised SQLite star schema
│   ├── p2a_titanic_eda.py        p2b_retail_profile.py   p2c_warehouse_skills.py
│   ├── p3a_titanic_prep.py
│   ├── p4a_titanic_model.py      p4b_titanic_pytorch.py  p4c_retail_analytics.py
│   ├── p4d_rag_pipeline.py       p4e_llm_finetuning.py
│   ├── p5_evaluation.py
│   └── p6_model_serving.py       p6b_deployment_docs.py
├── outputs/
│   ├── figures/                  6 annotated PNGs
│   ├── tables/                   CSV/JSON results, mlruns.jsonl, coverage matrix
│   ├── reports/                  20+ markdown deliverables
│   └── models/                   sklearn pipeline, torch checkpoint, LoRA adapter, MLflow
└── patches/                      16 unified diffs fixing the upstream script bugs
```

## Key deliverables

| Report | Skill |
|---|---|
| [`executive_summary.md`](outputs/reports/executive_summary.md) | executive-summary-generator |
| [`metric_reconciliation.md`](outputs/reports/metric_reconciliation.md) | metric-reconciliation |
| [`schema_map.md`](outputs/reports/schema_map.md) | schema-mapper |
| [`methodology.md`](outputs/reports/methodology.md) | methodology-explainer |
| [`qa_signoff.md`](outputs/reports/qa_signoff.md) | analysis-qa-checklist |
| [`peer_review.md`](outputs/reports/peer_review.md) | peer-review-template |
| [`retrospective.md`](outputs/reports/retrospective.md) | analysis-retrospective |
| [`skill_coverage_matrix.md`](outputs/reports/skill_coverage_matrix.md) | (verification) |
