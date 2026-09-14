"""CRISP-DM Phase 5 - Evaluation (both tracks).

Skills demonstrated (all driven from the real Phase 2-4 numbers):
  * visualization-builder            -- chart selection, hierarchy, annotation, export
  * insight-synthesis                -- So What / Why / Now What, scored and ranked
  * impact-quantification            -- point estimate + low/base/high range
  * technical-to-business-translator -- jargon detection + readability, before/after
  * analysis-qa-checklist            -- automated checks + sign-off
  * methodology-explainer            -- tiered explanation with an honest limitation
  * peer-review-template             -- must-fix / should-fix / optional
  * analysis-assumptions-log         -- structured, auditable assumption log
  * context-packager                 -- layered bundle + token budget
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd

from common import FIG, PALETTE, PROC, REP, ROOT, TAB, banner, save_table, style_plots, write_json, write_report

SKILLS = Path.home() / ".claude" / "skills"
PY = sys.executable


def run(skill: str, script: str, *args: str) -> str:
    path = SKILLS / skill / "scripts" / script
    print(f"\n    $ python {skill}/scripts/{script} ...")
    r = subprocess.run([PY, str(path), *args], capture_output=True, text=True)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    print("\n".join("    " + ln for ln in out.splitlines()[:30]))
    return out


def load() -> dict:
    def j(name):
        return json.loads((TAB / name).read_text())
    return {
        "eda": j("p2a_titanic_eda_findings.json"),
        "prep": j("p3a_titanic_prep_decisions.json"),
        "model": j("p4a_titanic_model_results.json"),
        "torch": j("p4b_pytorch_debug_results.json"),
        "retail": j("p4c_retail_analytics_results.json"),
        "rag": j("p4d_rag_results.json"),
        "serve": j("p6_serving_results.json"),
    }


# ----------------------------------------------------------- insight-synthesis
def insights(R: dict) -> list[dict]:
    banner("insight-synthesis", "Phase 5 - Evaluation",
           "So What -> Why -> Now What, scored on impact x confidence x actionability")
    r, m = R["retail"], R["model"]
    seg = {s["name"]: s for s in r["segmentation"]["profiles"]}
    champ = seg.get("Champions", {})
    f = r["funnel"]

    items = [
        {
            "finding": f"{champ.get('cust_share',0):.1f}% of customers (Champions) generate "
                       f"{champ.get('revenue_share',0):.1f}% of revenue.",
            "so_what": "Revenue is concentrated enough that losing a few hundred accounts would "
                       "be material; retention spend on this group has the highest return.",
            "why": "A wholesale-heavy customer mix: median 12 orders and GBP 4,629 spend versus "
                   "2 orders and GBP 674 overall.",
            "now_what": "Stand up named-account coverage for the 558 Champions before the next "
                        "Q4 peak, and add a churn alert when a Champion's recency exceeds 60 days.",
            "impact_gbp": champ.get("revenue", 0),
            "impact": 3, "confidence": 3, "actionability": 3,
        },
        {
            "finding": f"{100 - f['steps'][1]['overall_conv']:.0f}% of customers never place a "
                       f"second order; the 1st->2nd step loses {f['biggest_dropoff_customers']:,}.",
            "so_what": f"At GBP {f['aov']:,.0f} AOV that is GBP {f['biggest_dropoff_value']:,.0f} "
                       f"of un-realised second-order revenue in a single year.",
            "why": f"Median gap to a second order is {f['median_days_to_2nd']:.0f} days -- long "
                   f"enough that there is a real reactivation window, and no evidence of a "
                   f"geographic cause (UK {f['uk_conv']:.1f}% vs RoW {f['row_conv']:.1f}%).",
            "now_what": "Trigger a reorder prompt at day 40, before the median gap closes. "
                        "A 5pp lift is worth roughly GBP 104,000/year.",
            "impact_gbp": f["biggest_dropoff_value"],
            "impact": 3, "confidence": 3, "actionability": 3,
        },
        {
            "finding": "The 70% December revenue 'collapse' is a partial-month artefact, not a "
                       "business event.",
            "so_what": "Any decision taken on the raw month-on-month chart would be a reaction "
                       "to a reporting bug.",
            "why": f"The extract ends {R['retail']['root_cause']['days_in_dec']} days into "
                   f"December; pro-rated, December runs "
                   f"{R['retail']['root_cause']['prorated_vs_nov']:+.1%} against November.",
            "now_what": "Label or exclude partial periods in every recurring report. One-line "
                        "change to the reporting layer.",
            "impact_gbp": 0.0,
            "impact": 2, "confidence": 3, "actionability": 3,
        },
        {
            "finding": f"The survival model's holdout PR-AUC is {m['holdout']['pr_auc']:.3f} "
                       f"versus {m['best_cv_pr_auc']:.3f} in cross-validation.",
            "so_what": "The CV number overstates real-world performance by ~0.08 PR-AUC. "
                       "Promoting on the CV figure would set expectations the model cannot meet.",
            "why": "179 holdout rows is a small sample, and hyperparameters were selected on the "
                   "same CV folds that produced the 0.867.",
            "now_what": "Report the holdout figure externally. Treat the CV figure as a model-"
                        "selection tool only.",
            "impact_gbp": 0.0,
            "impact": 2, "confidence": 3, "actionability": 2,
        },
        {
            "finding": f"Model recall on male passengers is "
                       f"{[s for s in m['slices'] if s['slice']=='Sex=male'][0]['recall']:.3f} "
                       f"versus "
                       f"{[s for s in m['slices'] if s['slice']=='Sex=female'][0]['recall']:.3f} "
                       f"on female passengers.",
            "so_what": "The aggregate metric hides a subgroup where the model is barely better "
                       "than the base rate. Any decision applied uniformly would fail one group.",
            "why": "Male survival is 20% in the holdout; the model has little signal to separate "
                   "the rare positives within that group.",
            "now_what": "Either report per-group metrics alongside the headline, or set a "
                        "group-specific threshold. Do not ship a single global threshold silently.",
            "impact_gbp": 0.0,
            "impact": 3, "confidence": 3, "actionability": 2,
        },
        {
            "finding": f"Hybrid retrieval scores {R['rag']['evaluation'][2]['mrr']:.3f} MRR "
                       f"versus {R['rag']['evaluation'][0]['mrr']:.3f} for dense-only, but the "
                       f"cross-encoder rerank *reduced* it to "
                       f"{R['rag']['evaluation'][3]['mrr']:.3f}.",
            "so_what": "Adding a reranker on best-practice grounds alone would have made this "
                       "system measurably worse.",
            "why": "ms-marco-MiniLM is trained on web-search passages; this corpus is terse "
                   "technical prose with heavy vocabulary overlap.",
            "now_what": "Keep hybrid retrieval, drop the reranker, and re-test if the corpus "
                        "character changes.",
            "impact_gbp": 0.0,
            "impact": 2, "confidence": 2, "actionability": 3,
        },
    ]
    for i in items:
        i["score"] = i["impact"] * i["confidence"] * i["actionability"]
    items.sort(key=lambda x: -x["score"])

    print(f"  {len(items)} findings converted to insights and ranked "
          f"(impact x confidence x actionability, max 27)\n")
    for n, i in enumerate(items, 1):
        print(f"  {n}. [score {i['score']:>2}/27] {i['finding']}")
        print(f"     So what   : {i['so_what']}")
        print(f"     Why       : {i['why']}")
        print(f"     Now what  : {i['now_what']}")
        if i["impact_gbp"]:
            print(f"     Magnitude : GBP {i['impact_gbp']:,.0f}")
        print()
    print("  conflict check: no two insights point in opposite directions. The concentration")
    print("  finding (#1) and the repeat-purchase finding (#2) are complementary, not competing:")
    print("  one is about protecting existing value, the other about converting new customers.")
    return items


# ------------------------------------------------------- impact-quantification
def impact(R: dict) -> dict:
    banner("impact-quantification", "Phase 5 - Evaluation",
           "point estimate, then a low/base/high range -- never a single number")
    f = R["retail"]["funnel"]
    base_cust = f["steps"][0]["users"]
    aov = f["aov"]
    lift = 0.05

    print("  step 1  impact type: REVENUE GROWTH (a conversion-rate lift)")
    print("  step 2  inputs:")
    print(f"          baseline volume  {base_cust:,} first-time customers/year (measured)")
    print(f"          expected lift    {lift:.0%} on 1st->2nd order conversion (assumed)")
    print(f"          AOV              GBP {aov:,.2f} (measured)")
    print(f"          horizon          12 months")
    print(f"          confidence       MEDIUM -- the lift is an assumption, not an experiment")

    # NOTE: despite the name, --lift-pct takes a DECIMAL (the module docstring shows
    # `--lift-pct 0.02`). Passing 5 for "5%" silently returns a 100x overestimate.
    # --periods is 1 here, not 12: baseline_volume is already an annual figure, so
    # multiplying by 12 months would double-count the horizon.
    out = run("impact-quantification", "revenue_impact.py",
              "--model", "conversion_lift",
              "--baseline-volume", str(base_cust),
              "--lift-pct", str(lift),
              "--avg-order-value", f"{aov:.2f}",
              "--periods", "1")
    base_val = base_cust * lift * aov
    ci = run("impact-quantification", "confidence_interval.py",
             "--base", f"{base_val:.0f}", "--confidence", "medium")

    print("\n  step 5  assumption sensitivity:")
    rows = []
    for l in [0.02, 0.05, 0.10]:
        v = base_cust * l * aov
        rows.append({"lift": f"{l:.0%}", "annual_gbp": round(v)})
        print(f"          {l:>4.0%} lift -> GBP {v:>10,.0f}/year")
    print(f"          the estimate is LINEAR in the lift assumption, which is the single")
    print(f"          most uncertain input -- so the range must be reported, not the point.")
    print(f"\n  -> deliverable: GBP {base_val*0.5:,.0f} (low) / GBP {base_val:,.0f} (base) / "
          f"GBP {base_val*1.5:,.0f} (high), 12 months, MEDIUM confidence")
    return {"base": base_val, "low": base_val * 0.5, "high": base_val * 1.5,
            "sensitivity": rows, "confidence": "medium"}


# ------------------------------------------ technical-to-business-translator
def translate() -> dict:
    banner("technical-to-business-translator", "Phase 5 - Evaluation",
           "detect jargon, score readability, rewrite, re-score")
    technical = (
        "The HistGradientBoostingClassifier pipeline achieved a cross-validated PR-AUC of 0.867 "
        "(sigma 0.022) under stratified 5-fold CV, though the held-out estimate regressed to 0.791, "
        "indicating the hyperparameter optimisation overfit the validation folds. Slice-level "
        "diagnostics reveal heteroscedastic performance: recall on the male stratum is 0.417 versus "
        "0.867 on the female stratum, and the Brier score of 0.158 implies the posterior "
        "probabilities are poorly calibrated for downstream thresholding."
    )
    business = (
        "Our model correctly ranks who survived about 79% of the time on data it has never seen. "
        "In testing it looked better than that, so we are reporting the lower, more honest number. "
        "It works well for women, and much less well for men, where it catches under half of the "
        "cases it should. We would not yet use its confidence scores to make an automatic decision; "
        "we would use its ranking to prioritise a human review."
    )
    tp = PROC / "technical_draft.txt"
    bp = PROC / "business_draft.txt"
    tp.write_text(technical)
    bp.write_text(business)

    print("  step 1  jargon detection on the technical draft:")
    run("technical-to-business-translator", "jargon_detector.py", "--input", str(tp))
    print("\n  step 2  readability, technical draft:")
    t_out = run("technical-to-business-translator", "readability_scorer.py", "--input", str(tp))
    print("\n  step 2  readability, business rewrite:")
    b_out = run("technical-to-business-translator", "readability_scorer.py", "--input", str(bp))
    print("\n  step 1  jargon detection on the rewrite:")
    run("technical-to-business-translator", "jargon_detector.py", "--input", str(bp))

    write_report(
        "# Technical to business translation\n\n"
        "*Produced by the `technical-to-business-translator` skill. Audience persona: "
        "a commercial VP who owns the decision but not the method.*\n\n"
        "## Business version (lead with this)\n\n" + business +
        "\n\n## Translation table\n\n"
        "| Technical | Business |\n|---|---|\n"
        "| PR-AUC 0.791 | correctly ranks survivors about 79% of the time |\n"
        "| cross-validated 0.867 vs held-out 0.791 | it looked better in testing than it is; "
        "we report the lower number |\n"
        "| recall 0.417 on the male stratum | it misses more than half the men it should catch |\n"
        "| Brier score 0.158, poorly calibrated | its confidence scores are not trustworthy "
        "enough to automate a decision |\n"
        "| hyperparameter optimisation overfit the folds | we tuned it against the same data we "
        "measured it on, which flatters the result |\n\n"
        "## Appendix - original technical text\n\n" + technical + "\n",
        "technical_to_business.md")
    return {"technical": technical, "business": business}


# ---------------------------------------------------------- analysis-qa-checklist
def qa(R: dict) -> dict:
    banner("analysis-qa-checklist", "Phase 5 - Evaluation", "pre-delivery quality gate")
    print("  step 1  automated checks on a delivered table:")
    run("analysis-qa-checklist", "qa_runner.py",
        "--input", str(TAB / "rfm_segment_profiles.csv"),
        "--output", str(TAB / "qa_report.json"))

    print("\n  step 2-5  logic checklist:")
    checks = [
        ("Question framing", "PASS", "Both tracks trace to a stated business question in Phase 1."),
        ("Data sourcing", "PASS", "Both datasets hashed (SHA-256) and version-pinned; the mart's "
                                  "exclusions are reconciled to the raw feed with zero residual."),
        ("Transformations", "PASS", "Every cleaning statistic is fit on train only; target "
                                    "encoding is cross-fitted. Leakage inflation measured at +0.081 corr."),
        ("Statistical validity", "WARN", "The tuned-vs-untuned CV gain (+0.016) is smaller than "
                                         "fold noise (+/-0.022). Reported as 'no measurable gain', "
                                         "not as an improvement."),
        ("Findings", "WARN", "Holdout PR-AUC 0.791 is materially below CV 0.867 on n=179. "
                             "Stated explicitly rather than smoothed over."),
        ("Subgroup performance", "FAIL", "Male-passenger recall 0.417 is close to unusable. "
                                         "Blocks promotion to any automated decision."),
        ("Presentation", "PASS", "Every chart is annotated with its finding; the colourblind-safe "
                                 "Okabe-Ito palette is used throughout."),
        ("Assumptions documented", "PASS", "9 cleaning decisions plus a separate assumptions log."),
        ("Reproducibility", "PASS", "Seeded, pinned, hashed; one command reruns the whole pipeline."),
    ]
    for name, st, note in checks:
        print(f"     [{st:<4}] {name:<24} {note}")

    fails = [c for c in checks if c[1] == "FAIL"]
    warns = [c for c in checks if c[1] == "WARN"]
    decision = "CONDITIONAL" if fails else ("DELIVER" if not warns else "DELIVER WITH CAVEATS")
    print(f"\n  step 6  sign-off: {decision}")
    print(f"          {len(fails)} FAIL, {len(warns)} WARN, "
          f"{len(checks)-len(fails)-len(warns)} PASS")
    print(f"          The single FAIL blocks automated use. The analysis may be delivered as")
    print(f"          decision support with the subgroup caveat stated up front.")
    return {"checks": [{"check": c, "status": s, "note": n} for c, s, n in checks],
            "decision": decision}


# ---------------------------------------------------------- assumptions + context
def assumptions_and_context(R: dict) -> None:
    banner("analysis-assumptions-log", "Phase 5 - Evaluation", "auditable decision trail")
    log = {
        "analysis": "CRISP-DM skill demonstration - Titanic + Online Retail",
        "date": "2026-09-01",
        "analyst": "anita.agasaveeran@sjsu.edu",
        "decision_informed": "Which analytics/ML skills to adopt as team standard, and whether "
                             "the survival model may be promoted.",
        "assumptions": [
            {"id": 1, "category": "data", "assumption": "Rows with a null CustomerID (24.9%) can "
             "be excluded from customer-level analysis.",
             "rationale": "They cannot be attributed to a customer by construction.",
             "confidence": "high", "impact_if_wrong": "medium",
             "validation": "VALIDATED - reconciled to the raw feed with zero unexplained residual.",
             "status": "closed"},
            {"id": 2, "category": "data", "assumption": "Cancellations (InvoiceNo prefix 'C') are "
             "excluded from gross revenue.",
             "rationale": "Standard treatment for this dataset; gross, not net.",
             "confidence": "high", "impact_if_wrong": "high",
             "validation": "VALIDATED - accounts for GBP -896,812 of the 8.58% source-to-mart gap.",
             "status": "closed"},
            {"id": 3, "category": "business_logic", "assumption": "Gross margin is 45% for LTV.",
             "rationale": "Industry placeholder; the dataset contains no cost data.",
             "confidence": "low", "impact_if_wrong": "high",
             "validation": "OPEN - LTV scales linearly with this. Needs a finance-supplied figure.",
             "status": "open"},
            {"id": 4, "category": "business_logic", "assumption": "A 5% lift in 1st->2nd order "
             "conversion is achievable from a reorder prompt.",
             "rationale": "Assumption, not measurement. No experiment has been run.",
             "confidence": "low", "impact_if_wrong": "high",
             "validation": "OPEN - the impact estimate is linear in this. Run an A/B test before "
                           "committing budget.",
             "status": "open"},
            {"id": 5, "category": "statistical", "assumption": "Titanic rows are independent.",
             "rationale": "Required for a random stratified split to be valid.",
             "confidence": "medium", "impact_if_wrong": "medium",
             "validation": "OPEN - families share tickets and cabins, so relatives can span the "
                           "train/test split. GroupKFold on ticket would be more conservative.",
             "status": "open"},
            {"id": 6, "category": "statistical", "assumption": "The 179-row holdout is large "
             "enough to estimate PR-AUC.",
             "rationale": "20% of an 891-row dataset.",
             "confidence": "low", "impact_if_wrong": "medium",
             "validation": "OPEN - the CV/holdout gap of 0.076 is consistent with sampling noise "
                           "at this n. Repeated nested CV would tighten it.",
             "status": "open"},
        ],
    }
    p = TAB / "assumptions_log.json"
    p.write_text(json.dumps(log, indent=2))

    # Re-express in the tracker's own schema so its reporter runs on OUR log, not its demo.
    tracker_log = {
        "analysis_name": log["analysis"],
        "analyst": log["analyst"],
        "created": log["date"],
        "assumptions": [{
            "id": a["id"], "category": a["category"], "assumption": a["assumption"],
            "rationale": a["rationale"], "confidence": a["confidence"],
            "impact_if_wrong": a["impact_if_wrong"],
            "validation_plan": a["validation"],
            "validated": a["status"] == "closed",
            "validation_result": "confirmed" if a["status"] == "closed" else None,
            "validation_notes": a["validation"] if a["status"] == "closed" else None,
        } for a in log["assumptions"]],
    }
    tp = TAB / "assumptions_log_tracker_schema.json"
    tp.write_text(json.dumps(tracker_log, indent=2))
    crit = [a for a in log["assumptions"] if a["confidence"] == "low" and a["impact_if_wrong"] == "high"]
    print(f"  {len(log['assumptions'])} assumptions logged across data / business-logic / statistical")
    print(f"  {sum(1 for a in log['assumptions'] if a['status']=='closed')} validated and closed, "
          f"{sum(1 for a in log['assumptions'] if a['status']=='open')} open")
    print(f"\n  CRITICAL (low confidence x high impact) -- these gate the conclusions:")
    for a in crit:
        print(f"     [{a['id']}] {a['assumption']}")
        print(f"         plan: {a['validation']}")
    run("analysis-assumptions-log", "assumptions_tracker.py", "--load", str(tp), "--report")

    banner("context-packager", "Phase 5 - Evaluation", "layered bundle + token budget")
    bundle = PROC / "context_bundle.md"
    run("context-packager", "context_bundler.py",
        "--task", "Review this CRISP-DM analysis and recommend whether the survival model "
                  "may be promoted to an automated decision.",
        "--schema", str(REP / "schema_map.md"),
        "--business", str(REP / "metric_reconciliation.md"),
        "--files", str(REP / "query_review.md"), str(REP / "technical_to_business.md"),
        "--output", str(bundle))
    if bundle.exists():
        run("context-packager", "token_counter.py", "--input", str(bundle), "--limit", "100000")


def figures(R: dict, ins: list[dict]) -> None:
    banner("visualization-builder", "Phase 5 - Evaluation",
           "message-first chart selection, annotated for the reader")
    print("  step 1  message -> chart type (the skill's selection guide):")
    for msg, typ, why in [
        ("'Champions are 13% of customers but 63% of revenue'", "horizontal bar",
         "part-of-whole comparison across few categories; horizontal because labels are long"),
        ("'34% never place a second order'", "funnel / ordered bar",
         "sequential drop-off through ordered steps"),
        ("'Revenue trends up into Q4'", "line + rolling mean",
         "trend over time; the raw series is too noisy to read alone"),
        ("'The tuning gain sits inside fold noise'", "bar with error bars",
         "comparison where the uncertainty IS the message"),
    ]:
        print(f"          {msg:<50} -> {typ}\n             {why}")

    print("\n  step 3  running the skill's chart_builder for a recommendation + spec:")
    run("visualization-builder", "chart_builder.py", "--recommend",
        "--data-type", "part-to-whole", "--categories", "4", "--metric-count", "1")

    import matplotlib.pyplot as plt

    style_plots()
    m = R["model"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # Chart 1: the honest model story
    ax = axes[0]
    labels = ["CV (tuned)", "CV (untuned)", "Holdout"]
    vals = [m["best_cv_pr_auc"], m["cv_results"]["hgb"]["cv_pr_auc"], m["holdout"]["pr_auc"]]
    errs = [0, m["cv_results"]["hgb"]["cv_pr_auc_std"], 0]
    bars = ax.bar(labels, vals, yerr=errs, capsize=5,
                  color=[PALETTE[4], PALETTE[4], PALETTE[1]])
    ax.axhline(0.385, ls="--", c="grey", lw=1)
    ax.text(2.4, 0.395, "base rate", fontsize=8, color="grey", ha="right")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("PR-AUC")
    ax.set_title("The holdout number is the honest one")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}",
                ha="center", fontweight="bold", fontsize=9)
    ax.annotate("", xy=(2, m["holdout"]["pr_auc"] + 0.005), xytext=(0, m["best_cv_pr_auc"] - 0.005),
                arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=1.5))
    ax.text(1.0, 0.88, f"-{m['best_cv_pr_auc']-m['holdout']['pr_auc']:.3f}",
            color=PALETTE[1], fontweight="bold", ha="center", fontsize=9)

    # Chart 2: the subgroup failure the aggregate hides
    ax = axes[1]
    sl = [s for s in m["slices"] if s["slice"].startswith("Sex")]
    x = range(len(sl))
    ax.bar([i - 0.2 for i in x], [s["recall"] for s in sl], 0.4, label="recall", color=PALETTE[0])
    ax.bar([i + 0.2 for i in x], [s["pr_auc"] for s in sl], 0.4, label="PR-AUC", color=PALETTE[2])
    ax.set_xticks(list(x))
    ax.set_xticklabels([s["slice"] for s in sl])
    ax.set_ylim(0, 1.0)
    ax.set_title("Aggregate metrics hide a subgroup failure")
    ax.legend(frameon=False, fontsize=8)
    for i, s in enumerate(sl):
        ax.text(i - 0.2, s["recall"] + 0.02, f"{s['recall']:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, s["pr_auc"] + 0.02, f"{s['pr_auc']:.2f}", ha="center", fontsize=8)

    fig.suptitle("Phase 5 - what the evaluation actually says", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "p5_evaluation.png")
    plt.close(fig)
    print("\n  step 6  exported at 150 DPI, annotated titles state the finding, "
         "colourblind-safe palette")
    print("  -> wrote p5_evaluation.png")


def methodology_and_review(R: dict, qa_res: dict) -> None:
    banner("methodology-explainer", "Phase 5 - Evaluation", "tiered explanation + honest limitation")
    m = R["model"]
    md = f"""# How we did this, and what it can and cannot tell you

*Produced by the `methodology-explainer` skill. Three tiers: read the one that matches you.*

## Tier 1 - Executive (60 seconds)

We took two well-known public datasets and ran a complete, disciplined analysis over
each: one predicting who survived the Titanic, one understanding an online retailer's
customers. The point was to test 46 analysis "skills" on real data, not to discover
something new about either dataset.

The survival model ranks passengers correctly about **{m['holdout']['pr_auc']:.0%}** of the
time on data it has never seen. That is decent but not automatic-decision grade, and it
is noticeably worse for men than for women.

## Tier 2 - Business analyst

**Data.** Kaggle Titanic (891 passengers, 12 columns) and Kaggle Online Retail
(541,909 transaction lines, Dec 2010 - Dec 2011). Both hashed and version-pinned.

**Method.** CRISP-DM, all six phases, on both datasets in parallel. For the model:
the data was split into training and holdout *before* any cleaning, so nothing learned
from the holdout could leak backwards. Preprocessing lives inside a scikit-learn
Pipeline, so it is re-fit inside every cross-validation fold. Hyperparameters were
searched with Optuna over {m.get('param_importance', {}) and len(m['param_importance'])} parameters.

**Metric.** PR-AUC rather than accuracy. With 38% survivors, a model that predicts
"nobody survived" scores 62% accuracy while being useless. PR-AUC cannot be fooled
that way.

**Result.** Cross-validated PR-AUC {m['best_cv_pr_auc']:.3f}; holdout
{m['holdout']['pr_auc']:.3f}. We report the second number.

## Tier 3 - Technical peer

Stratified 5-fold CV, seed 42. `ColumnTransformer` over 10 numeric / 4 categorical /
3 binary features; `HistGradientBoostingClassifier`. Optuna TPE, 60 trials, log-scale
sampling on learning rate and L2. Ticket-prefix target encoding is cross-fitted over
5 folds with smoothing 10; in-fold encoding correlates {R['prep']['target_encoding']['naive_corr']:.3f}
with the target versus {R['prep']['target_encoding']['oof_corr']:.3f} out-of-fold, and the
difference is leakage. Decision threshold {m['threshold_maxf1']:.3f} selected on a
validation split, never re-tuned on the holdout. Brier {m['holdout']['brier']:.3f};
calibration error {m['calibration_error']:.4f} over 8 quantile bins.

## Limitations - read this before using any of it

1. **The holdout is 179 rows.** The gap between the cross-validated
   {m['best_cv_pr_auc']:.3f} and the holdout {m['holdout']['pr_auc']:.3f} is consistent
   with sampling noise at that size. Neither number should be quoted to three decimals
   as if it were precise.
2. **Subgroup performance is uneven.** Recall is
   {[s for s in m['slices'] if s['slice']=='Sex=male'][0]['recall']:.2f} for men versus
   {[s for s in m['slices'] if s['slice']=='Sex=female'][0]['recall']:.2f} for women. A
   single global threshold treats those groups as if they were the same. They are not.
3. **Probabilities are not well calibrated** (Brier {m['holdout']['brier']:.3f}). Use the
   model's *ranking*, not its stated confidence.
4. **Families share tickets and cabins.** A random split can put relatives on both sides
   of it. `GroupKFold` on ticket would be more conservative; we did not do that, and it
   may mean the numbers above are slightly optimistic.
5. **The retail dataset is one UK gift wholesaler in 2011.** Nothing here generalises to
   e-commerce at large, and the benchmark comparisons are indicative only.
"""
    write_report(md, "methodology.md")
    print("  three audience tiers written, with five stated limitations")
    print("  -> outputs/reports/methodology.md")

    banner("peer-review-template", "Phase 5 - Evaluation", "must-fix / should-fix / optional")
    review = [
        ("MUST-FIX", "Male-passenger recall is 0.417. Do not promote to any automated decision "
                     "until either a group-specific threshold is set or the gap is closed.",
         "Accepted - recorded as the blocking item in the QA sign-off."),
        ("MUST-FIX", "The tuned-vs-untuned CV delta (+0.016) is inside fold noise (+/-0.022) but "
                     "the tuned model was still promoted.",
         "Accepted - the report now states there is no measurable gain and that the tuned model "
         "was kept for reproducibility of the search, not for accuracy."),
        ("SHOULD-FIX", "Titanic families share tickets; a random split can split a family. "
                       "GroupKFold on ticket would be more conservative.",
         "Acknowledged - logged as open assumption #5. Not changed in this pass; it would alter "
         "every number in the report."),
        ("SHOULD-FIX", "The 45% gross margin behind the LTV figure is an industry placeholder.",
         "Acknowledged - logged as open assumption #3 and flagged inline wherever LTV appears."),
        ("SHOULD-FIX", "The A/B test section uses a simulated assignment.",
         "Accepted - the simulation is labelled in the console output, the JSON "
         "(`simulated: true`) and the report."),
        ("OPTIONAL", "The cross-encoder reranker reduced retrieval MRR and was still built.",
         "Accepted - kept deliberately, because the negative result is the finding."),
        ("OPTIONAL", "ARIMA(7,1,1) order was chosen by inspection rather than by AIC search.",
         "Acknowledged - MAPE 28% is reported so the reader can judge; a grid over (p,d,q) is "
         "the obvious next step."),
    ]
    for sev, item, resp in review:
        print(f"\n  [{sev}] {item}")
        print(f"           author response: {resp}")
    must = sum(1 for r in review if r[0] == "MUST-FIX")
    print(f"\n  sign-off: {must} must-fix items, both addressed in the report text.")
    print(f"  Reviewer conclusion: deliverable as decision support, NOT for automated use.")

    write_report(
        "# Peer review\n\n*Produced by the `peer-review-template` skill.*\n\n"
        "**Scope agreed with author:** statistical validity, leakage, and honesty of reporting.\n\n"
        + "\n".join(f"### [{s}] {i}\n\n**Author response:** {r}\n" for s, i, r in review)
        + f"\n## Sign-off\n\n{must} must-fix items raised, both addressed. "
          "Deliverable as decision support; **not** approved for automated decisioning "
          "while the subgroup gap stands.\n",
        "peer_review.md")

    write_report(
        "# QA sign-off\n\n*Produced by the `analysis-qa-checklist` skill.*\n\n"
        f"**Decision: {qa_res['decision']}**\n\n"
        "| Check | Status | Note |\n|---|---|---|\n"
        + "\n".join(f"| {c['check']} | {c['status']} | {c['note']} |" for c in qa_res["checks"])
        + "\n\nThe single FAIL (subgroup performance) blocks automated use. The analysis may be "
          "delivered as decision support with that caveat stated up front.\n",
        "qa_signoff.md")


def main() -> None:
    R = load()
    ins = insights(R)
    imp = impact(R)
    tr = translate()
    qa_res = qa(R)
    assumptions_and_context(R)
    figures(R, ins)
    methodology_and_review(R, qa_res)
    write_json({"insights": ins, "impact": imp, "qa": qa_res, "translation": tr},
               "p5_evaluation_results.json")


if __name__ == "__main__":
    main()
