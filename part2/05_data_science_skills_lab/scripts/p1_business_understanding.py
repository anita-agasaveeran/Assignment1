"""CRISP-DM Phase 1 - Business Understanding (both tracks).

Skills demonstrated:
  * stakeholder-requirements-gathering -- intake interview, decision type, scope,
                                          success criteria, sign-off, analysis brief
  * analysis-planning                  -- decompose, sequence, estimate, log risks

Phase 1 precedes data work, so this script establishes the questions the rest of
the pipeline answers. It reads only enough of each dataset to state scope honestly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd

from common import RETAIL, TITANIC, banner, sha256, write_report

BRIEF_RETAIL = """# Analysis brief - Online Retail

*Produced by the `stakeholder-requirements-gathering` skill.*

## 1. The request as received

> "Can you look into our customers? I want to know who the good ones are and why
> December looked so bad."

Two questions in one sentence, at different altitudes, with no success criteria.
This is exactly the vague ask the skill exists to convert into a scoped brief.

## 2. Intake interview - what was established

| Question | Answer |
|---|---|
| What decision does this inform? | Where to spend a fixed retention budget next quarter, and whether the December drop needs an incident response |
| Who is the audience? | Commercial VP (decides), analytics team (executes) |
| What does "done" look like? | A ranked, costed list of customer groups, and a yes/no on whether December was a real business event |
| Decision type | **Operational** (budget allocation) + **Tactical** (incident triage) - not strategic, so speed matters more than exhaustiveness |
| Deadline | End of week |
| What is explicitly out of scope? | Product-level assortment analysis; anything requiring marketing-spend data (we do not have it) |

## 3. Business questions, restated answerably

- **Q1.** Which customer groups exist, how large is each, and what share of revenue does each hold?
- **Q2.** How many customers return after a first order, and where in that sequence do we lose them?
- **Q3.** Is the December revenue fall a genuine business event or a data artefact?
- **Q4.** What is the single largest quantifiable opportunity, with a range?

## 4. Success criteria

1. Every group is defined by rules a marketer can act on, not by an opaque cluster id.
2. Every number traces to a query that can be re-run.
3. The December question gets a yes/no with evidence, not a hedge.
4. The opportunity is stated as a range with its assumptions named.

## 5. Scope

**In:** the 12-month transaction file; customer-level aggregation; revenue, orders, recency.
**Out:** CAC, LTV:CAC, payback - **the dataset contains no marketing spend**. This was
raised at intake rather than discovered at delivery, and no proxy will be invented.

## 6. Sign-off

Requirements confirmed before any data was touched. Any change to Q1-Q4 restarts this brief.
"""

BRIEF_TITANIC = """# Analysis brief - Titanic survival

*Produced by the `stakeholder-requirements-gathering` skill.*

## 1. The request as received

> "Build a model that predicts who survived the Titanic."

## 2. Intake interview

| Question | Answer |
|---|---|
| What decision does this inform? | Whether this modelling *workflow* is fit to be the team standard. The Titanic outcome itself is 113 years settled - there is no live decision riding on the prediction |
| Who is the audience? | The data team, plus a reviewer deciding whether to adopt the workflow |
| What does "done" look like? | A model whose reported number would survive an audit, with leakage, imbalance, threshold and subgroup performance all explicitly handled |
| Decision type | **Strategic** (tooling standard) - so rigour matters more than speed |
| Deadline | Same week |

## 3. Reframing the question

Because there is no live decision, the interesting question is **not** "how high can the
score go?" but **"is the score honest?"**. That reframing changes the whole design:

- accuracy is rejected as the headline metric before any model is fit
- a holdout is set aside and touched exactly once
- the tuning gain is tested against fold noise rather than reported as a win
- subgroup metrics are computed even though the aggregate looks acceptable

## 4. Success criteria

1. No statistic learned from data outside the training fold. Demonstrated, not asserted.
2. Metric choice justified against class balance before modelling starts.
3. A decision threshold chosen on validation and never re-tuned on the holdout.
4. Failure modes reported as prominently as headline performance.

## 5. Out of scope

Kaggle leaderboard position. Optimising against a public leaderboard is a form of
test-set tuning and would contradict criterion 1.
"""


def plan(rows_t: int, rows_r: int) -> str:
    return f"""# Analysis plan

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
| Titanic | {rows_t:,} | Kaggle `titanic` |
| Online Retail | {rows_r:,} | Kaggle `carrie1/ecommerce-data` (= UCI 352) |
"""


def main() -> None:
    banner("stakeholder-requirements-gathering", "Phase 1 - Business Understanding",
           "turn two vague asks into scoped, signed-off briefs")
    t = pd.read_csv(TITANIC)
    n_retail = sum(1 for _ in open(RETAIL)) - 1

    print("  Track B request: \"Can you look into our customers? I want to know who the good")
    print("                    ones are and why December looked so bad.\"")
    print("  -> two questions at different altitudes, no success criteria, no decision named.")
    print()
    print("  intake interview established:")
    for k, v in [("decision", "where to spend a fixed retention budget; whether December needs triage"),
                 ("audience", "commercial VP (decides) + analytics team (executes)"),
                 ("decision type", "operational + tactical -> speed over exhaustiveness"),
                 ("done looks like", "ranked costed customer groups; a yes/no on December"),
                 ("out of scope", "anything needing marketing spend -- WE DO NOT HAVE IT")]:
        print(f"     {k:<16}: {v}")
    print()
    print("  The out-of-scope line matters most: CAC, LTV:CAC and payback are the metrics a")
    print("  stakeholder will ask for by reflex, and this dataset cannot produce them. Saying")
    print("  so at intake costs one sentence. Discovering it at delivery costs the deliverable.")

    print("\n  Track A request: \"Build a model that predicts who survived the Titanic.\"")
    print("  -> no live decision rides on the prediction, so the real question is not")
    print("     'how high can the score go' but 'is the score honest'. That reframing is")
    print("     what makes accuracy-rejection, the untouched holdout and the noise test")
    print("     design decisions rather than afterthoughts.")

    write_report(BRIEF_RETAIL, "brief_retail.md")
    write_report(BRIEF_TITANIC, "brief_titanic.md")

    banner("analysis-planning", "Phase 1 - Business Understanding",
           "decompose -> sequence -> estimate -> log risks")
    print("  8 sub-questions, each answerable by one pull or one calculation")
    print("  Track A and Track B are independent after Phase 2 -> can run in parallel")
    print("  Q8 (opportunity sizing) is blocked on Q3 and Q4 -> sequenced last")
    print("  effort estimate 12h against an end-of-week deadline -> feasible")
    print("\n  risks logged (5), highest-impact first:")
    for r, m in [
        ("25% of retail rows have no CustomerID -> biased customer conclusions",
         "reconcile mart vs raw and quantify the gap BEFORE concluding"),
        ("December may be a partial month -> a fake incident",
         "check completeness before interpreting any month-on-month change"),
        ("891 Titanic rows -> holdout too small to be precise",
         "report spread, not just the point estimate"),
        ("families share tickets -> a random split can split a family",
         "log as an open assumption; note GroupKFold as the conservative option"),
        ("no marketing spend -> unit economics incomputable",
         "declared out of scope at intake; refuse to proxy it"),
    ]:
        print(f"     - {r}\n       mitigation: {m}")
    print("\n  Note: two of these risks (partial December, the CustomerID gap) were predicted")
    print("  here in Phase 1 and both were confirmed in Phase 2/4. That is the return on")
    print("  planning: the surprises were budgeted for instead of derailing the analysis.")

    write_report(plan(len(t), n_retail), "analysis_plan.md")


if __name__ == "__main__":
    main()
