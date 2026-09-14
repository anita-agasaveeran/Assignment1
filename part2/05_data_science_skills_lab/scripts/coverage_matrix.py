"""Build and verify the skill x CRISP-DM coverage matrix.

Every installed skill must map to (a) a CRISP-DM phase, (b) the script that
demonstrates it, and (c) a concrete artifact that exists on disk. This script
fails loudly if any of the 46 is unaccounted for or its evidence is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd

from common import FIG, REP, ROOT, TAB, banner, save_table

SKILLS = Path.home() / ".claude" / "skills"

# skill -> (pack, phase, dataset, script, evidence artifact, what was demonstrated)
COVERAGE = {
 # ---- Phase 1: Business Understanding
 "stakeholder-requirements-gathering": ("DA", 1, "Retail + Titanic", "p1_business_understanding.py",
   "reports/brief_retail.md", "Two vague asks converted into scoped briefs with success criteria; CAC declared out of scope at intake"),
 "analysis-planning": ("DA", 1, "Both", "p1_business_understanding.py",
   "reports/analysis_plan.md", "8 sub-questions decomposed, sequenced and effort-estimated; 5 risks logged, 2 of which materialised as predicted"),
 # ---- Phase 2: Data Understanding
 "exploratory-data-analysis": ("ML", 2, "Titanic", "p2a_titanic_eda.py",
   "figures/p2a_titanic_eda.png", "All 7 workflow steps; Cabin missingness proven predictive (+0.367); leakage scan clean"),
 "pandas-patterns": ("ML", 2, "Titanic", "p2a_titanic_eda.py",
   "tables/titanic_pclass_sex_summary.csv", "All 5 core rules applied; 26.2% memory cut via category dtype; validated m:1 merge"),
 "programmatic-eda": ("DA", 2, "Retail", "p2b_retail_profile.py",
   "tables/retail_null_profile.csv", "All 5 bundled scripts run on 541,909 rows; CustomerID 24.93% null flagged"),
 "data-quality-audit": ("DA", 2, "Retail", "p2b_retail_profile.py",
   "reports/p2b_skill_script_console_log.txt", "5 checks: 5,268 dup rows, 10,624 sub-min quantities, 0% orphans, data stale by design"),
 "data-catalog-entry": ("DA", 2, "Retail", "p2b_retail_profile.py",
   "reports/catalog_invoice_lines.md", "Real schema extracted from SQLite (397,884 rows, 7 cols) after fixing the extractor"),
 "schema-mapper": ("DA", 2, "Retail", "p2c_warehouse_skills.py",
   "reports/schema_map.md", "4 tables discovered, FKs inferred, data dictionary, 3 join paths, mermaid ERD"),
 "query-validation": ("DA", 2, "Retail", "p2c_warehouse_skills.py",
   "reports/query_review.md", "5 anti-patterns found; sargability fix measured at 2.8x faster"),
 "sql-to-business-logic": ("DA", 2, "Retail", "p2c_warehouse_skills.py",
   "reports/query_business_logic.md", "Query translated to plain language with 5 validation questions for the author"),
 "semantic-model-builder": ("DA", 2, "Retail", "p2c_warehouse_skills.py",
   "tables/semantic_model.yaml", "3 entities, 3 dimensions, 3 metrics; validator-clean after a fix round"),
 "metric-reconciliation": ("DA", 2, "Retail", "p2c_warehouse_skills.py",
   "reports/metric_reconciliation.md", "8.58% raw-vs-mart gap decomposed to a ZERO residual"),
 # ---- Phase 3: Data Preparation
 "data-cleaning": ("ML", 3, "Titanic", "p3a_titanic_prep.py",
   "tables/p3a_titanic_prep_decisions.json", "Split BEFORE cleaning; 9 decisions logged; train-only medians and winsorisation bounds"),
 "feature-engineering": ("ML", 3, "Titanic", "p3a_titanic_prep.py",
   "tables/p3a_titanic_prep_decisions.json", "Title/FamilySize/Deck/log1p; cross-fitted target encoding, leakage measured at +0.081 corr"),
 # ---- Phase 4: Modeling
 "reproducible-ml": ("ML", 4, "Titanic", "p4a_titanic_model.py",
   "requirements.txt", "Seeded, SHA-256 data hashes, git SHA, 143 pinned deps, one-command rerun"),
 "sklearn-pipelines": ("ML", 4, "Titanic", "p4a_titanic_model.py",
   "figures/p4a_titanic_model.png", "ColumnTransformer over 10/4/3 cols; 4 baselines cross-validated with preprocessing inside each fold"),
 "imbalanced-data": ("ML", 4, "Titanic", "p4a_titanic_model.py",
   "tables/p4a_titanic_model_results.json", "PR-AUC over accuracy; SMOTE in-fold tested and REJECTED as inside noise; threshold tuned on validation"),
 "hyperparameter-tuning": ("ML", 4, "Titanic", "p4a_titanic_model.py",
   "tables/p4a_titanic_model_results.json", "Optuna TPE 60 trials over the pipeline, log-scale spaces; learning_rate 68% of importance"),
 "experiment-tracking": ("ML", 4, "Titanic", "p4a_titanic_model.py",
   "tables/mlruns.jsonl", "7 runs logged to MLflow + JSONL with params, metrics, git SHA, data hash and env"),
 "model-evaluation": ("ML", 4, "Titanic", "p4a_titanic_model.py",
   "tables/titanic_slice_metrics.csv", "Holdout scored once; calibration; slice metrics exposed male recall 0.417; gain tested against fold noise"),
 "pytorch-training-loop": ("ML", 4, "Titanic", "p4b_titanic_pytorch.py",
   "figures/p4b_pytorch_debug.png", "Canonical loop on MPS; forgetting .eval() shown to cost 0.050 PR-AUC silently"),
 "ml-debugging": ("ML", 4, "Titanic", "p4b_titanic_pytorch.py",
   "tables/p4b_pytorch_debug_results.json", "One-batch overfit PASS; LR sweep to divergence; injected leak caught by the 5-point checklist"),
 "rag-pipeline": ("ML", 4, "Project docs", "p4d_rag_pipeline.py",
   "figures/p4d_rag_eval.png", "346 chunks (pinned corpus), real embeddings + BM25 + cross-encoder; hybrid +0.177 MRR, reranker -0.028 (reported)"),
 "llm-finetuning": ("ML", 4, "Retail", "p4e_llm_finetuning.py",
   "tables/p4e_llm_finetuning_results.json", "Real LoRA on distilgpt2, 0.359% of params; format 0%->100%, facts 0% (the expected split)"),
 "cohort-analysis": ("DA", 4, "Retail", "p4c_retail_analytics.py",
   "figures/p4c_retention_heatmap.png", "13 monthly cohorts; month-1 retention 20.6% mean; 2010-12 identified as a non-cohort"),
 "segmentation-analysis": ("DA", 4, "Retail", "p4c_retail_analytics.py",
   "tables/rfm_segment_profiles.csv", "RFM + k-means k=4 (silhouette 0.381); Champions = 12.9% of customers, 62.7% of revenue"),
 "funnel-analysis": ("DA", 4, "Retail", "p4c_retail_analytics.py",
   "tables/funnel_repeat_purchase.csv", "5-step repeat-purchase funnel; biggest drop sized at GBP 717,933"),
 "time-series-analysis": ("DA", 4, "Retail", "p4c_retail_analytics.py",
   "tables/p4c_retail_analytics_results.json", "ADF non-stationary; STL decomposition; 2 anomalies; ARIMA(7,1,1) MAPE 28.1%"),
 "root-cause-investigation": ("DA", 4, "Retail", "p4c_retail_analytics.py",
   "tables/p4c_retail_analytics_results.json", "70% December fall diagnosed as a partial-month artefact; 3 hypotheses rejected with evidence"),
 "ab-test-analysis": ("DA", 4, "Retail", "p4c_retail_analytics.py",
   "tables/p4c_retail_analytics_results.json", "SRM chi-square, two-proportion z-test, CI and ship decision (assignment simulated, labelled)"),
 "business-metrics-calculator": ("DA", 4, "Retail", "p4c_retail_analytics.py",
   "tables/p4c_retail_analytics_results.json", "9 e-commerce KPIs vs benchmarks; CAC declared incomputable rather than invented"),
 # ---- Phase 5: Evaluation
 "insight-synthesis": ("DA", 5, "Both", "p5_evaluation.py",
   "tables/p5_evaluation_results.json", "6 findings -> So What/Why/Now What, scored on impact x confidence x actionability"),
 "impact-quantification": ("DA", 5, "Retail", "p5_evaluation.py",
   "tables/p5_evaluation_results.json", "GBP 52K/104K/156K low-base-high with sensitivity on the one assumed input"),
 "visualization-builder": ("DA", 5, "Both", "p5_evaluation.py",
   "figures/p5_evaluation.png", "Message-first chart selection; annotated titles; Okabe-Ito colourblind-safe palette throughout"),
 "technical-to-business-translator": ("DA", 5, "Titanic", "p5_evaluation.py",
   "reports/technical_to_business.md", "Grade 14.3 -> 8.7 Flesch-Kincaid; 3 jargon terms -> 0"),
 "analysis-qa-checklist": ("DA", 5, "Both", "p5_evaluation.py",
   "reports/qa_signoff.md", "9-point gate: 1 FAIL (subgroup recall) blocks automated use; 2 WARN; delivered with caveats"),
 "methodology-explainer": ("DA", 5, "Titanic", "p5_evaluation.py",
   "reports/methodology.md", "Three audience tiers plus five honest limitations"),
 "peer-review-template": ("DA", 5, "Both", "p5_evaluation.py",
   "reports/peer_review.md", "2 must-fix, 3 should-fix, 2 optional, each with an author response"),
 "analysis-assumptions-log": ("DA", 5, "Both", "p5_evaluation.py",
   "tables/assumptions_log.json", "6 assumptions; 2 validated and closed; 2 critical (low confidence x high impact) gating conclusions"),
 "context-packager": ("DA", 5, "Both", "p5_evaluation.py",
   "tables/p5_evaluation_results.json", "Layered bundle from 4 docs, token-counted against a 100k budget (3% used)"),
 # ---- Phase 6: Deployment
 "model-serving": ("ML", 6, "Titanic", "p6_model_serving.py",
   "tables/p6_serving_results.json", "FastAPI over the whole pipeline; 422 on bad payloads; p50 20.9ms; batching 180x cheaper/row; drift z-scores"),
 "executive-summary-generator": ("DA", 6, "Retail", "p6b_deployment_docs.py",
   "reports/executive_summary.md", "One page, 3 quantified insights, a decision block, and a 'what this cannot tell you' section"),
 "data-narrative-builder": ("DA", 6, "Retail", "p6b_deployment_docs.py",
   "reports/data_narrative.md", "Situation-Complication-Resolution, 6 beats with intended feeling and one visual each"),
 "dashboard-specification": ("DA", 6, "Retail", "p6b_deployment_docs.py",
   "reports/dashboard_spec.md", "One-sentence purpose, 2 audiences -> 2 tabs, 8 metrics, 2 filters rejected with reasons"),
 "analysis-documentation": ("DA", 6, "Both", "p6b_deployment_docs.py",
   "reports/analysis_documentation.md", "Context, sources with hashes, methodology, results, reproduction commands, limitations"),
 "analysis-retrospective": ("DA", 6, "Both", "p6b_deployment_docs.py",
   "reports/retrospective.md", "Start/Stop/Continue with 5-whys on the 16 broken scripts and 4 tracked actions"),
}

PHASE_NAMES = {1: "Business Understanding", 2: "Data Understanding", 3: "Data Preparation",
               4: "Modeling", 5: "Evaluation", 6: "Deployment"}


def main() -> None:
    banner("(verification)", "all phases", "every installed skill must map to a phase and an artifact")
    installed = sorted(p.parent.name for p in SKILLS.glob("*/SKILL.md"))
    print(f"  installed skills : {len(installed)}")
    print(f"  mapped in matrix : {len(COVERAGE)}")

    missing = sorted(set(installed) - set(COVERAGE))
    extra = sorted(set(COVERAGE) - set(installed))
    if missing:
        print(f"  !! NOT DEMONSTRATED: {missing}")
    if extra:
        print(f"  !! MAPPED BUT NOT INSTALLED: {extra}")

    rows, absent = [], []
    for skill in installed:
        pack, phase, ds, script, art, what = COVERAGE[skill]
        p = ROOT / "outputs" / art if not art.endswith("requirements.txt") else ROOT / art
        ok = p.exists()
        if not ok:
            absent.append((skill, art))
        rows.append({"skill": skill, "pack": pack, "phase": phase,
                     "crisp_dm_phase": PHASE_NAMES[phase], "dataset": ds,
                     "script": script, "evidence": art, "evidence_exists": ok,
                     "demonstrated": what})

    df = pd.DataFrame(rows).sort_values(["phase", "pack", "skill"])
    save_table(df.set_index("skill"), "skill_coverage_matrix.csv")

    print(f"\n  by CRISP-DM phase:")
    for ph, n in df.groupby("crisp_dm_phase", sort=False).size().items():
        print(f"     {ph:<24} {n:>2} skills")
    print(f"\n  by pack:")
    print(f"     agent-ml-skills          {(df['pack']=='ML').sum():>2} / 15")
    print(f"     data-analytics-skills    {(df['pack']=='DA').sum():>2} / 31")

    if absent:
        print(f"\n  !! MISSING EVIDENCE FILES ({len(absent)}):")
        for s, a in absent:
            print(f"     {s:<36} {a}")
        sys.exit(1)
    if missing or extra:
        sys.exit(1)
    print(f"\n  VERIFIED: all {len(installed)} installed skills are mapped to a CRISP-DM phase")
    print(f"  and every one has its evidence artifact present on disk.")

    # markdown matrix for the README
    md = ["# Skill coverage matrix", "",
          f"All **{len(installed)}** installed skills, mapped to the CRISP-DM phase where they were "
          "used and the artifact that proves it.", ""]
    for ph in sorted(df["phase"].unique()):
        sub = df[df["phase"] == ph]
        md += [f"## Phase {ph} - {PHASE_NAMES[ph]}", "",
               "| Skill | Pack | Data | Script | What was demonstrated | Evidence |",
               "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            pack = "agent-ml" if r["pack"] == "ML" else "data-analytics"
            md.append(f"| `{r['skill']}` | {pack} | {r['dataset']} | "
                      f"[`{r['script']}`](../../scripts/{r['script']}) | {r['demonstrated']} | "
                      f"[`{r['evidence']}`](../{r['evidence']}) |")
        md.append("")
    (REP / "skill_coverage_matrix.md").write_text("\n".join(md))
    print(f"  -> wrote outputs/reports/skill_coverage_matrix.md")


if __name__ == "__main__":
    main()
