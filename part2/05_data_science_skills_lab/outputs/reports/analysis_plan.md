# Analysis plan

*Produced by the `analysis-planning` skill, after the briefs and before any data work.*

## Decomposition

Each sub-question must be answerable by one data pull or one calculation.

| # | Sub-question | Data needed | Availability |
|---|---|---|---|
| 1 | What is in each file, and is it trustworthy? | both raw files | confirmed |
| 2 | Do the two revenue definitions agree? | raw file + derived mart | confirmed |
| 3 | What customer groups exist? | customer-level RFM aggregate | derived from Q1 |
| 4 | Where do customers stop reordering? | order sequence per customer | derived from Q1 |
| 5 | Is December real? | daily revenue + dimensional splits | confirmed |
| 6 | Can survival be predicted honestly? | Titanic, train split only | confirmed |
| 7 | Does the model fail on any subgroup? | Titanic holdout + slice columns | derived from Q6 |
| 8 | What is the largest opportunity worth? | outputs of Q3 and Q4 | derived |

## Sequencing

```
Phase 1  briefs, plan                        <- you are here
Phase 2  profile both datasets, build the warehouse, reconcile revenue   [Q1, Q2]
Phase 3  clean and engineer features (train-only statistics)
Phase 4  Track A: pipeline -> imbalance -> tuning -> nets -> RAG -> LoRA  [Q6]
         Track B: cohorts -> segments -> funnel -> time series -> RCA     [Q3, Q4, Q5]
Phase 5  evaluate, synthesise, quantify, QA, peer review                  [Q7, Q8]
Phase 6  serve, document, specify the dashboard, retrospective
```

Track A and Track B are independent after Phase 2 and can run in parallel.
Q8 is blocked on both Q3 and Q4, so it is sequenced last.

## Effort estimate

| Step | Estimate |
|---|---|
| Phase 2 profiling + warehouse | 2h |
| Phase 3 preparation | 1h |
| Phase 4 Track A modelling | 3h |
| Phase 4 Track B analytics | 3h |
| Phase 5 evaluation + write-up | 2h |
| Phase 6 deployment + docs | 1h |
| **Total** | **12h** against an end-of-week deadline - feasible |

## Risks and dependencies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| The 25% of retail rows with no CustomerID bias customer-level conclusions | high | high | Reconcile mart vs raw explicitly and quantify the gap before drawing conclusions (Q2) |
| Titanic's 891 rows give a holdout too small to be precise | high | medium | Report the spread, not just the point estimate; state it as a limitation |
| Families share Titanic tickets, so a random split can split a family | medium | medium | Log as an open assumption; note GroupKFold as the conservative alternative |
| No marketing spend, so unit economics are incomputable | certain | medium | Declared out of scope at intake; refuse to proxy it |
| December is a partial month | medium | high | Check completeness before interpreting any month-on-month change |

## Data inventory

| Dataset | Rows | Source |
|---|---|---|
| Titanic | 891 | Kaggle `titanic` |
| Online Retail | 541,909 | Kaggle `carrie1/ecommerce-data` (= UCI 352) |
