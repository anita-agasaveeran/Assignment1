# Skill coverage matrix

All **46** installed skills, mapped to the CRISP-DM phase where they were used and the artifact that proves it.

## Phase 1 - Business Understanding

| Skill | Pack | Data | Script | What was demonstrated | Evidence |
|---|---|---|---|---|---|
| `analysis-planning` | data-analytics | Both | [`p1_business_understanding.py`](../../scripts/p1_business_understanding.py) | 8 sub-questions decomposed, sequenced and effort-estimated; 5 risks logged, 2 of which materialised as predicted | [`reports/analysis_plan.md`](../reports/analysis_plan.md) |
| `stakeholder-requirements-gathering` | data-analytics | Retail + Titanic | [`p1_business_understanding.py`](../../scripts/p1_business_understanding.py) | Two vague asks converted into scoped briefs with success criteria; CAC declared out of scope at intake | [`reports/brief_retail.md`](../reports/brief_retail.md) |

## Phase 2 - Data Understanding

| Skill | Pack | Data | Script | What was demonstrated | Evidence |
|---|---|---|---|---|---|
| `data-catalog-entry` | data-analytics | Retail | [`p2b_retail_profile.py`](../../scripts/p2b_retail_profile.py) | Real schema extracted from SQLite (397,884 rows, 7 cols) after fixing the extractor | [`reports/catalog_invoice_lines.md`](../reports/catalog_invoice_lines.md) |
| `data-quality-audit` | data-analytics | Retail | [`p2b_retail_profile.py`](../../scripts/p2b_retail_profile.py) | 5 checks: 5,268 dup rows, 10,624 sub-min quantities, 0% orphans, data stale by design | [`reports/p2b_skill_script_console_log.txt`](../reports/p2b_skill_script_console_log.txt) |
| `metric-reconciliation` | data-analytics | Retail | [`p2c_warehouse_skills.py`](../../scripts/p2c_warehouse_skills.py) | 8.58% raw-vs-mart gap decomposed to a ZERO residual | [`reports/metric_reconciliation.md`](../reports/metric_reconciliation.md) |
| `programmatic-eda` | data-analytics | Retail | [`p2b_retail_profile.py`](../../scripts/p2b_retail_profile.py) | All 5 bundled scripts run on 541,909 rows; CustomerID 24.93% null flagged | [`tables/retail_null_profile.csv`](../tables/retail_null_profile.csv) |
| `query-validation` | data-analytics | Retail | [`p2c_warehouse_skills.py`](../../scripts/p2c_warehouse_skills.py) | 5 anti-patterns found; sargability fix measured at 2.8x faster | [`reports/query_review.md`](../reports/query_review.md) |
| `schema-mapper` | data-analytics | Retail | [`p2c_warehouse_skills.py`](../../scripts/p2c_warehouse_skills.py) | 4 tables discovered, FKs inferred, data dictionary, 3 join paths, mermaid ERD | [`reports/schema_map.md`](../reports/schema_map.md) |
| `semantic-model-builder` | data-analytics | Retail | [`p2c_warehouse_skills.py`](../../scripts/p2c_warehouse_skills.py) | 3 entities, 3 dimensions, 3 metrics; validator-clean after a fix round | [`tables/semantic_model.yaml`](../tables/semantic_model.yaml) |
| `sql-to-business-logic` | data-analytics | Retail | [`p2c_warehouse_skills.py`](../../scripts/p2c_warehouse_skills.py) | Query translated to plain language with 5 validation questions for the author | [`reports/query_business_logic.md`](../reports/query_business_logic.md) |
| `exploratory-data-analysis` | agent-ml | Titanic | [`p2a_titanic_eda.py`](../../scripts/p2a_titanic_eda.py) | All 7 workflow steps; Cabin missingness proven predictive (+0.367); leakage scan clean | [`figures/p2a_titanic_eda.png`](../figures/p2a_titanic_eda.png) |
| `pandas-patterns` | agent-ml | Titanic | [`p2a_titanic_eda.py`](../../scripts/p2a_titanic_eda.py) | All 5 core rules applied; 26.2% memory cut via category dtype; validated m:1 merge | [`tables/titanic_pclass_sex_summary.csv`](../tables/titanic_pclass_sex_summary.csv) |

## Phase 3 - Data Preparation

| Skill | Pack | Data | Script | What was demonstrated | Evidence |
|---|---|---|---|---|---|
| `data-cleaning` | agent-ml | Titanic | [`p3a_titanic_prep.py`](../../scripts/p3a_titanic_prep.py) | Split BEFORE cleaning; 9 decisions logged; train-only medians and winsorisation bounds | [`tables/p3a_titanic_prep_decisions.json`](../tables/p3a_titanic_prep_decisions.json) |
| `feature-engineering` | agent-ml | Titanic | [`p3a_titanic_prep.py`](../../scripts/p3a_titanic_prep.py) | Title/FamilySize/Deck/log1p; cross-fitted target encoding, leakage measured at +0.081 corr | [`tables/p3a_titanic_prep_decisions.json`](../tables/p3a_titanic_prep_decisions.json) |

## Phase 4 - Modeling

| Skill | Pack | Data | Script | What was demonstrated | Evidence |
|---|---|---|---|---|---|
| `ab-test-analysis` | data-analytics | Retail | [`p4c_retail_analytics.py`](../../scripts/p4c_retail_analytics.py) | SRM chi-square, two-proportion z-test, CI and ship decision (assignment simulated, labelled) | [`tables/p4c_retail_analytics_results.json`](../tables/p4c_retail_analytics_results.json) |
| `business-metrics-calculator` | data-analytics | Retail | [`p4c_retail_analytics.py`](../../scripts/p4c_retail_analytics.py) | 9 e-commerce KPIs vs benchmarks; CAC declared incomputable rather than invented | [`tables/p4c_retail_analytics_results.json`](../tables/p4c_retail_analytics_results.json) |
| `cohort-analysis` | data-analytics | Retail | [`p4c_retail_analytics.py`](../../scripts/p4c_retail_analytics.py) | 13 monthly cohorts; month-1 retention 20.6% mean; 2010-12 identified as a non-cohort | [`figures/p4c_retention_heatmap.png`](../figures/p4c_retention_heatmap.png) |
| `funnel-analysis` | data-analytics | Retail | [`p4c_retail_analytics.py`](../../scripts/p4c_retail_analytics.py) | 5-step repeat-purchase funnel; biggest drop sized at GBP 717,933 | [`tables/funnel_repeat_purchase.csv`](../tables/funnel_repeat_purchase.csv) |
| `root-cause-investigation` | data-analytics | Retail | [`p4c_retail_analytics.py`](../../scripts/p4c_retail_analytics.py) | 70% December fall diagnosed as a partial-month artefact; 3 hypotheses rejected with evidence | [`tables/p4c_retail_analytics_results.json`](../tables/p4c_retail_analytics_results.json) |
| `segmentation-analysis` | data-analytics | Retail | [`p4c_retail_analytics.py`](../../scripts/p4c_retail_analytics.py) | RFM + k-means k=4 (silhouette 0.381); Champions = 12.9% of customers, 62.7% of revenue | [`tables/rfm_segment_profiles.csv`](../tables/rfm_segment_profiles.csv) |
| `time-series-analysis` | data-analytics | Retail | [`p4c_retail_analytics.py`](../../scripts/p4c_retail_analytics.py) | ADF non-stationary; STL decomposition; 2 anomalies; ARIMA(7,1,1) MAPE 28.1% | [`tables/p4c_retail_analytics_results.json`](../tables/p4c_retail_analytics_results.json) |
| `experiment-tracking` | agent-ml | Titanic | [`p4a_titanic_model.py`](../../scripts/p4a_titanic_model.py) | 7 runs logged to MLflow + JSONL with params, metrics, git SHA, data hash and env | [`tables/mlruns.jsonl`](../tables/mlruns.jsonl) |
| `hyperparameter-tuning` | agent-ml | Titanic | [`p4a_titanic_model.py`](../../scripts/p4a_titanic_model.py) | Optuna TPE 60 trials over the pipeline, log-scale spaces; learning_rate 68% of importance | [`tables/p4a_titanic_model_results.json`](../tables/p4a_titanic_model_results.json) |
| `imbalanced-data` | agent-ml | Titanic | [`p4a_titanic_model.py`](../../scripts/p4a_titanic_model.py) | PR-AUC over accuracy; SMOTE in-fold tested and REJECTED as inside noise; threshold tuned on validation | [`tables/p4a_titanic_model_results.json`](../tables/p4a_titanic_model_results.json) |
| `llm-finetuning` | agent-ml | Retail | [`p4e_llm_finetuning.py`](../../scripts/p4e_llm_finetuning.py) | Real LoRA on distilgpt2, 0.359% of params; format 0%->100%, facts 0% (the expected split) | [`tables/p4e_llm_finetuning_results.json`](../tables/p4e_llm_finetuning_results.json) |
| `ml-debugging` | agent-ml | Titanic | [`p4b_titanic_pytorch.py`](../../scripts/p4b_titanic_pytorch.py) | One-batch overfit PASS; LR sweep to divergence; injected leak caught by the 5-point checklist | [`tables/p4b_pytorch_debug_results.json`](../tables/p4b_pytorch_debug_results.json) |
| `model-evaluation` | agent-ml | Titanic | [`p4a_titanic_model.py`](../../scripts/p4a_titanic_model.py) | Holdout scored once; calibration; slice metrics exposed male recall 0.417; gain tested against fold noise | [`tables/titanic_slice_metrics.csv`](../tables/titanic_slice_metrics.csv) |
| `pytorch-training-loop` | agent-ml | Titanic | [`p4b_titanic_pytorch.py`](../../scripts/p4b_titanic_pytorch.py) | Canonical loop on MPS; forgetting .eval() shown to cost 0.050 PR-AUC silently | [`figures/p4b_pytorch_debug.png`](../figures/p4b_pytorch_debug.png) |
| `rag-pipeline` | agent-ml | Project docs | [`p4d_rag_pipeline.py`](../../scripts/p4d_rag_pipeline.py) | 346 chunks (pinned corpus), real embeddings + BM25 + cross-encoder; hybrid +0.177 MRR, reranker -0.028 (reported) | [`figures/p4d_rag_eval.png`](../figures/p4d_rag_eval.png) |
| `reproducible-ml` | agent-ml | Titanic | [`p4a_titanic_model.py`](../../scripts/p4a_titanic_model.py) | Seeded, SHA-256 data hashes, git SHA, 143 pinned deps, one-command rerun | [`requirements.txt`](../requirements.txt) |
| `sklearn-pipelines` | agent-ml | Titanic | [`p4a_titanic_model.py`](../../scripts/p4a_titanic_model.py) | ColumnTransformer over 10/4/3 cols; 4 baselines cross-validated with preprocessing inside each fold | [`figures/p4a_titanic_model.png`](../figures/p4a_titanic_model.png) |

## Phase 5 - Evaluation

| Skill | Pack | Data | Script | What was demonstrated | Evidence |
|---|---|---|---|---|---|
| `analysis-assumptions-log` | data-analytics | Both | [`p5_evaluation.py`](../../scripts/p5_evaluation.py) | 6 assumptions; 2 validated and closed; 2 critical (low confidence x high impact) gating conclusions | [`tables/assumptions_log.json`](../tables/assumptions_log.json) |
| `analysis-qa-checklist` | data-analytics | Both | [`p5_evaluation.py`](../../scripts/p5_evaluation.py) | 9-point gate: 1 FAIL (subgroup recall) blocks automated use; 2 WARN; delivered with caveats | [`reports/qa_signoff.md`](../reports/qa_signoff.md) |
| `context-packager` | data-analytics | Both | [`p5_evaluation.py`](../../scripts/p5_evaluation.py) | Layered bundle from 4 docs, token-counted against a 100k budget (3% used) | [`tables/p5_evaluation_results.json`](../tables/p5_evaluation_results.json) |
| `impact-quantification` | data-analytics | Retail | [`p5_evaluation.py`](../../scripts/p5_evaluation.py) | GBP 52K/104K/156K low-base-high with sensitivity on the one assumed input | [`tables/p5_evaluation_results.json`](../tables/p5_evaluation_results.json) |
| `insight-synthesis` | data-analytics | Both | [`p5_evaluation.py`](../../scripts/p5_evaluation.py) | 6 findings -> So What/Why/Now What, scored on impact x confidence x actionability | [`tables/p5_evaluation_results.json`](../tables/p5_evaluation_results.json) |
| `methodology-explainer` | data-analytics | Titanic | [`p5_evaluation.py`](../../scripts/p5_evaluation.py) | Three audience tiers plus five honest limitations | [`reports/methodology.md`](../reports/methodology.md) |
| `peer-review-template` | data-analytics | Both | [`p5_evaluation.py`](../../scripts/p5_evaluation.py) | 2 must-fix, 3 should-fix, 2 optional, each with an author response | [`reports/peer_review.md`](../reports/peer_review.md) |
| `technical-to-business-translator` | data-analytics | Titanic | [`p5_evaluation.py`](../../scripts/p5_evaluation.py) | Grade 14.3 -> 8.7 Flesch-Kincaid; 3 jargon terms -> 0 | [`reports/technical_to_business.md`](../reports/technical_to_business.md) |
| `visualization-builder` | data-analytics | Both | [`p5_evaluation.py`](../../scripts/p5_evaluation.py) | Message-first chart selection; annotated titles; Okabe-Ito colourblind-safe palette throughout | [`figures/p5_evaluation.png`](../figures/p5_evaluation.png) |

## Phase 6 - Deployment

| Skill | Pack | Data | Script | What was demonstrated | Evidence |
|---|---|---|---|---|---|
| `analysis-documentation` | data-analytics | Both | [`p6b_deployment_docs.py`](../../scripts/p6b_deployment_docs.py) | Context, sources with hashes, methodology, results, reproduction commands, limitations | [`reports/analysis_documentation.md`](../reports/analysis_documentation.md) |
| `analysis-retrospective` | data-analytics | Both | [`p6b_deployment_docs.py`](../../scripts/p6b_deployment_docs.py) | Start/Stop/Continue with 5-whys on the 16 broken scripts and 4 tracked actions | [`reports/retrospective.md`](../reports/retrospective.md) |
| `dashboard-specification` | data-analytics | Retail | [`p6b_deployment_docs.py`](../../scripts/p6b_deployment_docs.py) | One-sentence purpose, 2 audiences -> 2 tabs, 8 metrics, 2 filters rejected with reasons | [`reports/dashboard_spec.md`](../reports/dashboard_spec.md) |
| `data-narrative-builder` | data-analytics | Retail | [`p6b_deployment_docs.py`](../../scripts/p6b_deployment_docs.py) | Situation-Complication-Resolution, 6 beats with intended feeling and one visual each | [`reports/data_narrative.md`](../reports/data_narrative.md) |
| `executive-summary-generator` | data-analytics | Retail | [`p6b_deployment_docs.py`](../../scripts/p6b_deployment_docs.py) | One page, 3 quantified insights, a decision block, and a 'what this cannot tell you' section | [`reports/executive_summary.md`](../reports/executive_summary.md) |
| `model-serving` | agent-ml | Titanic | [`p6_model_serving.py`](../../scripts/p6_model_serving.py) | FastAPI over the whole pipeline; 422 on bad payloads; p50 20.9ms; batching 180x cheaper/row; drift z-scores | [`tables/p6_serving_results.json`](../tables/p6_serving_results.json) |
