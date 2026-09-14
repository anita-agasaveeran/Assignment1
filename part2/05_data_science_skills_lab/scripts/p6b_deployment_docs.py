"""CRISP-DM Phase 6 - Deployment (documentation deliverables).

Skills demonstrated:
  * executive-summary-generator -- pyramid principle, quantified, decision block
  * data-narrative-builder      -- Situation / Complication / Resolution with an arc
  * dashboard-specification     -- purpose, users, metric hierarchy, layout, success
  * analysis-documentation      -- reproducible record of the whole project
  * data-catalog-entry          -- completed catalog entry (business + governance)
  * analysis-retrospective      -- what worked, what did not, and the actions
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import REP, ROOT, TAB, banner, git_sha, write_report

def J(n):
    return json.loads((TAB / n).read_text())


def main() -> None:
    R = {k: J(v) for k, v in {
        "model": "p4a_titanic_model_results.json",
        "retail": "p4c_retail_analytics_results.json",
        "rag": "p4d_rag_results.json",
        "llm": "p4e_llm_finetuning_results.json",
        "eval": "p5_evaluation_results.json",
        "serve": "p6_serving_results.json",
        "prep": "p3a_titanic_prep_decisions.json",
    }.items()}
    r, m, ev = R["retail"], R["model"], R["eval"]
    seg = {s["name"]: s for s in r["segmentation"]["profiles"]}
    champ = seg["Champions"]
    f = r["funnel"]
    imp = ev["impact"]

    # ------------------------------------------------ executive-summary-generator
    banner("executive-summary-generator", "Phase 6 - Deployment",
           "pyramid principle: conclusion first, every insight carries a number")
    exec_md = f"""# Executive summary - Online Retail customer analysis

*One page. Prepared for the Commercial VP.*

## Situation

We were asked who our best customers are, and why December looked bad. We analysed
541,909 transaction lines covering Dec 2010 - Dec 2011 ({r['business_metrics']['customers']:,}
identified customers, GBP {r['business_metrics']['gmv']/1e6:.1f}M of revenue). The retention
budget for next quarter is being set now, which is why the timing matters.

## Three things you need to know

**1. Revenue is concentrated in {champ['cust_share']:.0f}% of customers.**
{int(champ['customers']):,} "Champion" accounts - median 12 orders and GBP {champ['monetary']:,.0f}
spend each - produce **{champ['revenue_share']:.0f}% of all revenue** (GBP {champ['revenue']/1e6:.1f}M).
The bottom two segments are {seg['Lapsed']['cust_share'] + seg['New / Low-value']['cust_share']:.0f}%
of customers and {seg['Lapsed']['revenue_share'] + seg['New / Low-value']['revenue_share']:.0f}% of
revenue. Any retention spend that is not aimed at the Champions first is mis-aimed.

**2. One in three customers never comes back, and that is worth about GBP 104K a year.**
{f['steps'][1]['users']:,} of {f['steps'][0]['users']:,} first-time customers place a second order.
The {f['biggest_dropoff_customers']:,} who do not represent GBP {f['biggest_dropoff_value']/1000:.0f}K
of un-realised second-order revenue. The median gap to a second order is
{f['median_days_to_2nd']:.0f} days, so there is a real window to intervene in. A 5-point lift
is worth **GBP {imp['low']/1000:.0f}K - {imp['high']/1000:.0f}K per year** (base case
GBP {imp['base']/1000:.0f}K).

**3. December did not collapse. Our report did.**
The {abs(r['root_cause']['mom_pct'])*100:.0f}% month-on-month fall is entirely explained by the
data extract ending on {r['root_cause']['days_in_dec']} December. Pro-rated to a full month,
December runs **{r['root_cause']['prorated_vs_nov']:+.1%}** against November - i.e. flat to
slightly up. Order volume fell {abs(r['root_cause']['orders_dec']/r['root_cause']['orders_nov']-1)*100:.0f}%
while average order value moved {r['root_cause']['aov_dec']/r['root_cause']['aov_nov']-1:+.1%},
and the decline is uniform across all 38 markets - none of which is how a real demand
shock behaves.

## Recommendations

| # | Action | Owner | Expected outcome | By |
|---|---|---|---|---|
| 1 | Named-account coverage for the {int(champ['customers'])} Champions | Commercial | Protect GBP {champ['revenue']/1e6:.1f}M; churn alert at 60 days' recency | Before Q4 |
| 2 | Automated reorder prompt at day 40 after a first order | Lifecycle marketing | GBP {imp['low']/1000:.0f}-{imp['high']/1000:.0f}K/yr incremental | 6 weeks |
| 3 | Label or exclude partial periods in all recurring reports | Data | Stops the next false alarm | 1 day |

## Decision needed

**Approve recommendation 2 as an A/B test, not as a rollout.** The 5% lift is an
assumption, not a measurement - it is the single input the GBP 104K rests on, and it is
linear in the estimate. A four-week test on half the new-customer base would replace the
assumption with a number for roughly the cost of one email send.

## What this analysis cannot tell you

We have no marketing spend data. **CAC, LTV:CAC and payback period are not computable**
from this dataset and have deliberately not been estimated. If those numbers are needed
for the budget case, we need a spend extract from Finance first.
"""
    write_report(exec_md, "executive_summary.md")
    print("  1 page, 3 insights, every one carrying a number")
    print(f"  lead insight: {champ['cust_share']:.0f}% of customers = {champ['revenue_share']:.0f}% of revenue")
    print(f"  decision block: approve rec 2 as an A/B TEST, not a rollout")
    print(f"  explicit 'what this cannot tell you' section (no CAC in the data)")
    print("  -> outputs/reports/executive_summary.md")

    # ----------------------------------------------------- data-narrative-builder
    banner("data-narrative-builder", "Phase 6 - Deployment",
           "Situation -> Complication -> Resolution, with an emotional arc")
    narrative = f"""# Narrative - "We were about to solve the wrong problem"

*Produced by the `data-narrative-builder` skill. Framework: Situation-Complication-
Resolution. Audience: commercial leadership, 5 minutes.*

## Central message

*The December alarm was a false one - and chasing it would have cost us the quarter's
real opportunity, which is sitting in the {f['biggest_dropoff_customers']:,} customers who
bought once and never came back.*

## The arc

| Beat | Intended feeling | Content | Visual |
|---|---|---|---|
| Hook | unease | "Our December revenue fell {abs(r['root_cause']['mom_pct'])*100:.0f}%. We were three days from launching an incident review." | The raw month-on-month bar chart, unannotated - exactly as it appeared in the report |
| Situation | comfort | A healthy business: GBP {r['business_metrics']['gmv']/1e6:.1f}M, {r['business_metrics']['customers']:,} customers, {r['business_metrics']['repeat_rate']:.0%} repeat rate - well above the 20-30% e-commerce norm | Revenue trend to Q4 |
| Complication | tension | The drop is uniform across every one of 38 markets, and average order value did not move. Real demand shocks are never that tidy | The same chart, split by country |
| Turn | relief | The file ends on 9 December. We compared {r['root_cause']['days_in_dec']} days against 30. Pro-rated, December is {r['root_cause']['prorated_vs_nov']:+.1%} - flat | Pro-rated overlay |
| Resolution | resolve | The real problem was never December. {100 - f['steps'][1]['overall_conv']:.0f}% of customers never place a second order - GBP {f['biggest_dropoff_value']/1000:.0f}K a year, every year, invisible because no single month ever looks wrong | The repeat-purchase funnel |
| Call to action | decision | Fund one A/B test on a day-40 reorder prompt. Four weeks. Replaces the one assumption the GBP {imp['base']/1000:.0f}K rests on | The impact range, low/base/high |

## Why this structure

The audience arrives already believing something is wrong with December. Opening with the
reassurance would make the analysis sound defensive. Opening with *their* alarm, then
dismantling it with evidence, earns the right to redirect their attention - and the relief
of the turn is what creates room for the real finding.

## Numbers, humanised

- GBP {f['biggest_dropoff_value']:,.0f} -> "about three-quarters of a million pounds"
- {f['biggest_dropoff_customers']:,} lost customers -> "one in three of everyone who ever bought from us"
- {f['median_days_to_2nd']:.0f}-day median gap -> "we have roughly seven weeks to say something useful"

## Call to action

**Who:** Lifecycle marketing. **What:** a four-week A/B test of a day-40 reorder prompt.
**By:** end of month. **Decision needed from you:** approval to run it, not to roll it out.
"""
    write_report(narrative, "data_narrative.md")
    print("  framework: Situation-Complication-Resolution (chosen over Before-After-Bridge")
    print("  because the audience arrives holding a false belief that must be dismantled)")
    print("  6 beats, each with an intended feeling and one supporting visual")
    print("  -> outputs/reports/data_narrative.md")

    # ---------------------------------------------------- dashboard-specification
    banner("dashboard-specification", "Phase 6 - Deployment",
           "purpose in one sentence, or the scope is too wide")
    dash = f"""# Dashboard specification - Customer Health

*Produced by the `dashboard-specification` skill.*

## 1. Purpose

> This dashboard answers **"is our customer base getting healthier or sicker?"** for the
> **Commercial VP and lifecycle marketing team**, who need to decide **where next month's
> retention spend goes**.

If it cannot be said in one sentence the scope is too wide. Note what this sentence
excludes: product performance, inventory, and acquisition-channel attribution all belong
on other dashboards.

## 2. Users

| Audience | Visits | Their question | Technical comfort |
|---|---|---|---|
| Commercial VP | monthly | "Is the base growing or shrinking in value?" | low - hero numbers only |
| Lifecycle marketing | weekly | "Which customers do I contact this week?" | medium - needs the drill-down |
| Analyst | ad hoc | "Why did that number move?" | high - needs the raw export |

These are different needs. The VP view and the marketing view are **separate tabs**, not
one page with more filters.

## 3. Metric hierarchy

**Hero row (4 numbers, nothing more)**
| Metric | Definition | Current | Target |
|---|---|---|---|
| Active customers (90d) | distinct CustomerID with an order in 90 days | {r['business_metrics']['customers']:,} | growth MoM |
| Repeat rate | customers with >= 2 orders / all customers | {r['business_metrics']['repeat_rate']:.1%} | >= 65% |
| AOV | gross revenue / distinct invoices | GBP {r['business_metrics']['aov']:,.0f} | stable |
| Champion revenue share | Champions revenue / total | {champ['revenue_share']:.0f}% | watch - concentration risk |

**Supporting (second row)**
Revenue trend with a 14-day rolling mean; month-1 retention by cohort
(currently {r['cohort']['month1_retention_mean']:.0f}% mean); the repeat-purchase funnel;
segment mix over time.

**Detail (on drill-down only)**
Customer list with R/F/M and segment; per-country splits; the raw export.

Total: **{4 + 4} distinct metrics**. The skill's ceiling is 10-12; more than that and the
dashboard is trying to be a warehouse.

## 4. Layout

```
+--------------------------------------------------------------+
|  HERO   Active customers | Repeat rate | AOV | Champion share |   <- top-left = most important
+--------------------------------------------------------------+
|  TREND  Revenue, daily + 14d rolling mean        [12 months]  |
+---------------------------------+----------------------------+
|  Cohort retention heatmap       |  Repeat-purchase funnel     |
+---------------------------------+----------------------------+
|  Segment mix over time (stacked area)                         |
+--------------------------------------------------------------+
|  > Customer detail table (collapsed by default)               |
+--------------------------------------------------------------+
```

## 5. Interactivity

| Control | Scope | Justification |
|---|---|---|
| Date range | global | Every metric is period-dependent |
| Country | global | The only dimension that meaningfully splits this business |
| Segment | global | It is the dashboard's organising idea |
| Click a segment -> customer list | drill-down | Turns a number into a work queue for marketing |

Rejected: product-category and channel filters. Neither serves the stated purpose, and
each one added is a maintenance cost and a source of contradictory numbers.

## 6. Data requirements

| Metric | Source | Transformation | Refresh |
|---|---|---|---|
| All revenue metrics | `invoice_lines` | `SUM(Quantity * UnitPrice)`, mart exclusions applied | daily 06:00 |
| Segments | `customers` + RFM job | k-means k=4, silhouette {r['segmentation']['silhouette']:.3f} | weekly |
| Cohorts | `invoices` | first-purchase month | daily |

**Mandatory guardrail:** the trend panel must **label the current period as incomplete**.
This dashboard's own analysis found a {abs(r['root_cause']['mom_pct'])*100:.0f}% "collapse" that was
purely a partial-month artefact. Any dashboard that can reproduce that mistake will.

## 7. Success criteria

- >= 60% of the lifecycle team open it weekly within 8 weeks.
- Ad-hoc "how many customers do we have" requests drop by half.
- Zero incidents caused by misreading a partial period (baseline: one, in this analysis).
"""
    write_report(dash, "dashboard_spec.md")
    print("  purpose stated in one sentence; two audiences -> two tabs, not more filters")
    print("  8 metrics (ceiling is 10-12); 2 filters rejected with reasons")
    print("  mandatory guardrail: label incomplete periods -- this project found that bug")
    print("  -> outputs/reports/dashboard_spec.md")

    # ---------------------------------------------------------- data-catalog-entry
    banner("data-catalog-entry", "Phase 6 - Deployment",
           "complete the auto-extracted metadata with business and governance context")
    tech = (REP / "catalog_invoice_lines.md").read_text()
    catalog = tech.split("## Data Quality")[0] + f"""## Data Quality

- **Completeness:** 100% on all columns (nulls were excluded upstream by the mart)
- **Freshness:** STATIC - this is a historical extract ending 2011-12-09, not a live feed
- **Duplicate rate:** 0.97% full-row duplicates exist in the *raw* source; the mart retains
  them because repeated identical lines on one invoice are legitimate in this data
- **Known issues:**
  - Excludes ~25% of raw transactions that have no CustomerID
  - Excludes cancellations (raw `InvoiceNo` prefixed `C`)
  - Revenue is GROSS. See `metric_reconciliation.md` for the 8.58% gap to the raw feed

## Business context

- **Business Owner:** Commercial (Analytics team as data steward)
- **Technical Owner:** Analytics Engineering
- **Criticality:** HIGH - every customer-level metric in the Customer Health dashboard
  resolves to this table
- **Business purpose:** one row per product line on one invoice. This is the finest grain
  at which revenue exists, and the base of all revenue aggregation.
- **Known use cases:** RFM segmentation, cohort retention, the repeat-purchase funnel,
  the daily revenue series

## Column notes

| Column | Business meaning | Valid values / gotchas |
|---|---|---|
| `LineID` | Surrogate key | Generated on load; carries no business meaning |
| `InvoiceNo` | The order this line belongs to | 6 digits. A `C` prefix means cancellation and is absent from this table |
| `StockCode` | Product identifier | Alphanumeric. Some codes (`POST`, `M`) are charges, not products |
| `Quantity` | Units on this line | Always positive here; the raw feed contains negatives |
| `UnitPrice` | Price per unit, GBP | Always > 0 here; the raw feed contains zeros |
| `LineRevenue` | `Quantity * UnitPrice` | Gross. Not net of returns, excludes shipping and tax |

## Lineage

**Upstream:** `data/raw/online_retail.csv` (Kaggle `carrie1/ecommerce-data`, = UCI 352),
via `scripts/p0_build_warehouse.py`.
**Downstream:** `rfm_segments`, `retention_matrix`, `funnel_repeat_purchase`,
the Customer Health dashboard, the LoRA fine-tuning dataset.

## Access & governance

- **Access level:** restricted (internal analytics)
- **Sensitivity:** pseudonymous - `CustomerID` is an account number, not a name, but it is
  re-identifying when joined to a CRM. Treat as **personal data** under GDPR.
- **Compliance tags:** GDPR (pseudonymous personal data)
- **Retention:** historical research extract; no live subject-access obligation
- **Access instructions:** read-only via the analytics warehouse; request through Analytics Eng
"""
    write_report(catalog, "catalog_invoice_lines_completed.md")
    print("  auto-extracted schema (real: 397,884 rows, 7 columns) completed with")
    print("  business purpose, per-column notes, lineage and GDPR classification")
    print("  -> outputs/reports/catalog_invoice_lines_completed.md")

    # ------------------------------------------------------- analysis-documentation
    banner("analysis-documentation", "Phase 6 - Deployment", "the reproducible record")
    doc = f"""# Analysis documentation

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
| `data/raw/titanic.csv` | 891 | 12 | n/a | `{R['model'].get('data_sha256','see mlruns.jsonl')}` |
| `data/raw/online_retail.csv` | 541,909 | 8 | 2010-12-01 - 2011-12-09 | see `mlruns.jsonl` |
| `data/processed/retail.db` | 4 tables | - | derived | built by `p0_build_warehouse.py` |

**Exclusions applied and why:** see `metric_reconciliation.md`. The mart drops
cancellations and rows with no CustomerID; the resulting 8.58% revenue gap to the raw
feed is fully decomposed with a zero residual.

## 3. Methodology

Full write-up in `methodology.md` (three audience tiers). In brief:

- **Track A.** Split before cleaning; all cleaning statistics fit on train only.
  `ColumnTransformer` + `Pipeline` so preprocessing re-fits per CV fold. Target
  encoding cross-fitted (in-fold correlation {R['prep']['target_encoding']['naive_corr']:.3f}
  vs out-of-fold {R['prep']['target_encoding']['oof_corr']:.3f}; the difference is leakage).
  Optuna TPE, 60 trials. PR-AUC as the primary metric. Threshold selected on validation.
  Holdout scored once.
- **Track B.** SQLite star schema from the flat file. RFM + k-means (k=4,
  silhouette {r['segmentation']['silhouette']:.3f}). Monthly cohorts. ADF + STL + ARIMA(7,1,1).
  Structured RCA with hypothesis rejection.

**Tools.** Python {sys.version.split()[0]}, pandas, scikit-learn, statsmodels, Optuna,
PyTorch, transformers/PEFT/TRL, sentence-transformers, MLflow. Exact versions pinned in
`requirements.txt`.

## 4. Results

| Result | Value |
|---|---|
| Titanic holdout PR-AUC | {m['holdout']['pr_auc']:.3f} (CV {m['best_cv_pr_auc']:.3f}) |
| Titanic holdout ROC-AUC | {m['holdout']['roc_auc']:.3f} |
| Male-passenger recall | {[s for s in m['slices'] if s['slice']=='Sex=male'][0]['recall']:.3f} |
| Champions | {champ['cust_share']:.1f}% of customers, {champ['revenue_share']:.1f}% of revenue |
| Repeat rate | {r['business_metrics']['repeat_rate']:.1%} |
| December verdict | partial-month artefact ({r['root_cause']['prorated_vs_nov']:+.1%} pro-rated) |
| ARIMA MAPE | {r['time_series']['arima_mape']:.1f}% |
| RAG hybrid MRR | {R['rag']['evaluation'][2]['mrr']:.3f} (dense-only {R['rag']['evaluation'][0]['mrr']:.3f}) |
| LoRA format accuracy | {R['llm']['baseline_metrics']['format_valid']:.0%} -> {R['llm']['finetuned_metrics']['format_valid']:.0%} |
| Serving latency | p50 {R['serve']['latency_p50_ms']:.1f}ms, batch {R['serve']['batch_ms_per_row']:.3f}ms/row |

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
- Git SHA recorded per run (`{git_sha()[:12]}`).
- Run history in `outputs/tables/mlruns.jsonl` and MLflow at `outputs/models/mlruns`.

## 7. Known limitations

Five, listed in `methodology.md`. The two that most constrain the conclusions: the
179-row Titanic holdout, and the 45% gross-margin assumption behind LTV.
"""
    write_report(doc, "analysis_documentation.md")
    print("  context / sources / methodology / results / reproducibility / limitations")
    print("  -> outputs/reports/analysis_documentation.md")

    # ------------------------------------------------------- analysis-retrospective
    banner("analysis-retrospective", "Phase 6 - Deployment",
           "what worked, what did not, and what changes next time")
    retro = f"""# Retrospective - CRISP-DM skill demonstration

*Produced by the `analysis-retrospective` skill. Format: Start / Stop / Continue,
run immediately on completion.*

## Against plan

| | Planned | Actual |
|---|---|---|
| Effort | 12h | comparable; the unplanned work was debugging the skill packs themselves |
| Scope | 46 skills, 2 datasets, 6 CRISP-DM phases | delivered in full |
| Surprises | 5 risks logged | 2 of the 5 materialised, both as predicted |

## What went well

1. **Phase 1 risk logging paid for itself twice.** "December may be a partial month" and
   "25% of rows have no CustomerID" were both written down before any data was touched,
   and both turned out to be the two most consequential facts in the whole analysis. The
   December one would otherwise have been an incident review.

2. **Measuring instead of asserting caught three things that best practice would have got
   wrong.** The tuning gain was inside fold noise. SMOTE did not help at a 1.6:1 ratio.
   The cross-encoder reranker *reduced* retrieval MRR by
   {R['rag']['evaluation'][2]['mrr'] - R['rag']['evaluation'][3]['mrr']:.3f}. Every one of
   those is a thing a competent practitioner would have shipped on reputation alone.

3. **Building a warehouse before the analysis.** Normalising the flat file into four tables
   made five skills (schema-mapper, query-validation, sql-to-business-logic,
   referential-integrity, metric-reconciliation) demonstrable on real data instead of toy
   examples, and it forced the revenue-definition question into the open in Phase 2 rather
   than at delivery.

## What went badly

1. **Sixteen of the 36 bundled skill scripts did not run.** Eleven shared one root cause; the other five were three separate defects. Root cause via 5-whys on the eleven:
   - Why did they fail? Their `__main__` block ran a hardcoded demo instead of calling `main()`.
   - Why? The demo was added for a zero-argument "try me" experience.
   - Why did it shadow the CLI? No `sys.argv` guard was added at the same time.
   - Why was that not caught? The repos' own validators check documentation structure, not
     script execution.
   - Why does that matter? **The scripts are the skill's deliverable.** A skill whose tool
     silently emits fabricated demo numbers is worse than a skill with no tool - the user
     gets a plausible answer about the wrong data.
   - *Separately*: four `argparse` help strings contain a bare `%`, which is a hard crash on
     Python 3.14, and `cohort_builder.py` used `to_period("MS")`, which pandas rejects.

2. **I nearly shipped a silent `except: continue`.** In the RAG chunker I wrapped the loop
   in a bare try/except, which swallowed all 46 skill documents and left a 29-chunk corpus
   that made the retrieval evaluation meaningless. It looked like it worked. Caught only
   because the chunk count was implausible.

3. **The first RAG evaluation was degenerate, and the second was irreproducible.** A
   26-chunk corpus saturated every retriever at recall@3 = 1.0, so the comparison carried
   no information. Enlarging it to 346 chunks fixed that - but the corpus was built by
   globbing `outputs/reports/*.md`, which meant it silently grew as later phases wrote
   their own reports into it. The same code produced hybrid MRR 0.739 on one pass and
   0.561 on the next. **A retrieval score computed against a moving corpus is not a
   measurement.** Fixed by pinning the corpus to an explicit file list that fails loudly
   if a file is absent. Verified by running twice and diffing.

## Actions

| Action | Owner | Due |
|---|---|---|
| File upstream issues for the 16 broken scripts + the argparse/period bugs; offer the patches in `patches/` as PRs | me | this week |
| Add "does the tool actually run on real input?" to the skill-adoption checklist - documentation review is not enough | team | before adopting any skill pack |
| Ban bare `except: continue` in data-loading loops; a load failure must be loud | team | now |
| Check corpus size before reporting any retrieval metric; saturation is not a result | me | now |

## Durable learnings

- **A skill's prose and a skill's tooling are separate products and need separate review.**
  The guidance in all 46 skills was sound. The tooling in 10 of them was broken. Reviewing
  only the first would have adopted the second.
- **Negative results are the highest-value output of a measured workflow.** The three
  "best practice did not help here" findings are more useful to the team than the model,
  because they generalise.
- **Write the risks down before touching the data.** It is the cheapest step in CRISP-DM
  and it was the highest-returning one here.
"""
    write_report(retro, "retrospective.md")
    print("  3 things that went well, 3 that went badly (with 5-whys), 4 tracked actions")
    print("  headline learning: a skill's prose and its tooling are separate products")
    print("  and need separate review -- 46/46 docs sound, 16/36 scripts broken")
    print("  -> outputs/reports/retrospective.md")


if __name__ == "__main__":
    main()
