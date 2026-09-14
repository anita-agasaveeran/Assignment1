# Analysis documentation

*Produced by the `analysis-documentation` skill. Tiered for a mixed audience.*

## 1. Business context

**Question.** Two, in parallel. (A) Can a survival model be built whose reported number
would survive an audit? (B) Who are this retailer's valuable customers, and was the
December revenue fall real?

**Requested by.** Commercial VP (Track B); the data team (Track A).
**Decision informed.** Retention budget allocation; adoption of a modelling workflow.
**Success criteria.** See `brief_retail.md` and `brief_titanic.md`.

## 2. Data sources

| Source | Rows | Cols | Period | SHA-256 (first 16) |
|---|---|---|---|---|
| `data/raw/titanic.csv` | 891 | 12 | n/a | `see mlruns.jsonl` |
| `data/raw/online_retail.csv` | 541,909 | 8 | 2010-12-01 - 2011-12-09 | see `mlruns.jsonl` |
| `data/processed/retail.db` | 4 tables | - | derived | built by `p0_build_warehouse.py` |

**Exclusions applied and why:** see `metric_reconciliation.md`. The mart drops
cancellations and rows with no CustomerID; the resulting 8.58% revenue gap to the raw
feed is fully decomposed with a zero residual.

## 3. Methodology

Full write-up in `methodology.md` (three audience tiers). In brief:

- **Track A.** Split before cleaning; all cleaning statistics fit on train only.
  `ColumnTransformer` + `Pipeline` so preprocessing re-fits per CV fold. Target
  encoding cross-fitted (in-fold correlation 0.262
  vs out-of-fold 0.180; the difference is leakage).
  Optuna TPE, 60 trials. PR-AUC as the primary metric. Threshold selected on validation.
  Holdout scored once.
- **Track B.** SQLite star schema from the flat file. RFM + k-means (k=4,
  silhouette 0.381). Monthly cohorts. ADF + STL + ARIMA(7,1,1).
  Structured RCA with hypothesis rejection.

**Tools.** Python 3.14.7, pandas, scikit-learn, statsmodels, Optuna,
PyTorch, transformers/PEFT/TRL, sentence-transformers, MLflow. Exact versions pinned in
`requirements.txt`.

## 4. Results

| Result | Value |
|---|---|
| Titanic holdout PR-AUC | 0.791 (CV 0.867) |
| Titanic holdout ROC-AUC | 0.833 |
| Male-passenger recall | 0.417 |
| Champions | 12.9% of customers, 62.7% of revenue |
| Repeat rate | 65.6% |
| December verdict | partial-month artefact (+2.2% pro-rated) |
| ARIMA MAPE | 28.1% |
| RAG hybrid MRR | 0.739 (dense-only 0.562) |
| LoRA format accuracy | 0% -> 100% |
| Serving latency | p50 9.2ms, batch 0.069ms/row |

## 5. Insights and recommendations

See `executive_summary.md` (decisions) and `p5_evaluation_results.json` (all six
insights, scored and ranked).

## 6. Reproducibility

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/run_all.py        # runs every phase in order
```

- Seed 42 applied to python/random, numpy and torch.
- Both raw datasets SHA-256 hashed; the hash is logged with every experiment run.
- Git SHA recorded per run (`4ea6c51bee46`).
- Run history in `outputs/tables/mlruns.jsonl` and MLflow at `outputs/models/mlruns`.

## 7. Known limitations

Five, listed in `methodology.md`. The two that most constrain the conclusions: the
179-row Titanic holdout, and the 45% gross-margin assumption behind LTV.
