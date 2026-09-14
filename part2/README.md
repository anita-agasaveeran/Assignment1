# Assignment 1 — Part 2: Replicating the Data Science Experiments

This folder replicates a subset of the experiments from
[dlmastery/data_science_examples](https://github.com/dlmastery/data_science_examples)
using **Claude Code** as the coding assistant. Each project was built from the verbatim
prompt in the source repo's [`PROMPTS.md`](https://github.com/dlmastery/data_science_examples/blob/main/PROMPTS.md),
with follow-up prompts as needed to get a working, tested result.

**Video walkthrough:** [https://youtu.be/GApvUAs4Mow](https://youtu.be/GApvUAs4Mow)

Every experiment has its own README with full details (setup, results, CRISP-DM phases,
design decisions). This page is just the index.

---

## Experiments

| # | Project | Original prompt (abridged) | What was built |
|---|---|---|---|
| 0 | [`00_dynamic_todo_workspace`](00_dynamic_todo_workspace/) | *"build a modern end2end dynamic todo list application, industry best experience of ux and nice features"* | Full-stack todo app on stock Python 3 (no install): SQLite REST API + keyboard-first SPA, natural-language quick add, ⌘K palette, optimistic UI, offline queue, live sync over SSE, undo everywhere. Includes a [DESIGN.md](00_dynamic_todo_workspace/DESIGN.md). |
| 1 | [`01_nyc_taxi_trip_prediction`](01_nyc_taxi_trip_prediction/) | *"end2end data science project including data, training, deployment, crisp-dm framework, and an awesome front end… Kaggle NYC taxi challenge… interactive map and trip estimation"* | Gradient-boosted trip-duration model on 1.16M real 2016 trips (RMSLE 0.3065, MAE 3.1 min), p10–p90 uncertainty band from quantile models, interactive map with a 24-hour "when should I leave" sweep, and Data/Model tabs serving the CRISP-DM audit trail. 25 tests. |
| 2 | [`02_nano_llm_transformer`](02_nano_llm_transformer/) | *"build a simple llm and chatbot (state of art primitives but fit in my laptop gpu)… crisp-dm… admin dashboard… implement autoresearch to do hill climbing"* | A 16.9M-parameter language model + chatbot trained from scratch on an M4 Pro. **AutoResearch** starts from a 2019-era baseline and A/B-tests each modern primitive (Muon, RoPE, QK-norm, no dropout, …) against a measured seed-noise floor — 86 runs, −23% val loss. Pretrained PPL 4.42; SFT chat model served in a dashboard. |
| 3 | [`03_customer_segmentation_clustering`](03_customer_segmentation_clustering/) | *"clustering using popular kaggle data set… crisp-dm… admin dashboard… autoresearch to do hill climbing"* | Segmentation of ~9,000 credit-card holders (Kaggle `ccdata`) via a CASH-style hill-climbing search over 3.4M clustering pipelines, with random search as the control arm. Key finding: the objective-optimal model failed the business charter on stability, so the study loops back to Business Understanding and deploys a reproducible KMeans k=3 instead. 19 invariant tests. |
| 4 | [`04_associative_pattern_mining`](04_associative_pattern_mining/) | *"associative pattern mining using popular kaggle data set… crisp-dm… admin dashboard… autoresearch to do hill climbing"* | Market-basket mining on Online Retail (17,982 baskets, 2,547 products). Hyperparameters found by a hill climber scored on a temporal holdout; 500 rules → 403 replicate out of sample (80.6%). Statistical honesty layer: Fisher exact tests, FDR over 6.5M hypotheses, Webb's productivity filter, swap-randomised null. Dashboard cites the paper behind every panel. |
| 5 | [`05_data_science_skills_lab`](05_data_science_skills_lab/) | *"install param087 agent-ml-skills and nimrodfisher data-analytics-skills and demonstrate every skill on appropriate popular kaggle data set. also include crisp-dm steps"* | All 46 skills exercised across CRISP-DM on Titanic (15 ML skills) and Online Retail (31 analytics skills), with a coverage matrix that fails if any skill lacks an evidence artifact. Along the way, 16 of the 36 bundled skill scripts turned out to be broken and were patched (`patches/`). |

---

## Common threads

- **CRISP-DM throughout.** Projects 1–5 are organised as the six CRISP-DM phases, with
  success criteria fixed before modelling and the Evaluation → Business Understanding
  loop actually exercised when a result missed the bar (most visibly in Project 3).
- **AutoResearch / hill climbing with a control.** Projects 2, 3 and 4 replace hand-picked
  hyperparameters with a search, and each reports a baseline (seed-noise floor, random
  search, or swap-randomised null) so "the search helped" is a comparison, not a claim.
- **Dashboards rendered from run artifacts.** Each data-science project serves an admin
  dashboard built from a single run record, so numbers on screen match the numbers in the
  report.
- **Negative results are reported as such.** RMSNorm was a null on loss (P2); SMOTE and
  Optuna tuning were inside fold noise (P5); GQA and z-loss were rejected at small scale (P2).

## Running an experiment

Each project is self-contained with its own `requirements.txt` (or none, for Project 0)
and a Quickstart section in its README. In general:

```bash
cd 0X_project_name
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

then follow the project's `make` targets or scripts. Datasets are downloaded on first run
from public sources; no Kaggle credentials are needed.

## Not replicated

Projects 6–15 from the source repo (anomaly detection, AutoGluon AutoML, visual curriculum,
FlowForge, CRISP-DM curriculum, enterprise audit, time-series forecasting, NYC TLC audit
platform, AutoGluon multimodal, SPY forecasting) were out of scope for this submission.
