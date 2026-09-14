# Retrospective - CRISP-DM skill demonstration

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
   0.028. Every one of
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
