# ARMLab — Associative Pattern Mining on Online Retail

A complete CRISP-DM project on the **Online Retail** market-basket dataset
(Kaggle `carrie1/ecommerce-data`; originally UCI ML Repository 352), with two
things a typical market-basket write-up leaves out:

1. **AutoResearch** — the mining hyperparameters are not hand-picked. A
   stochastic hill climber searches them against an explicit, literature-grounded
   objective that only rewards rules surviving an untouched temporal holdout.
2. **Statistical honesty** — Fisher exact tests, FDR correction against the
   *full searched hypothesis space*, Webb's productivity filter, and a
   swap-randomised null baseline.

Everything is served by a data-science admin dashboard where each panel states
the measure it shows, the paper it comes from, and the caveat that applies.

```bash
make setup     # venv + dependencies
make run       # full pipeline, ~2-3 min (downloads the data on first run)
make serve     # dashboard at http://127.0.0.1:8931
```

---

## Headline results

Run fingerprint `9ac2d9f86318` · 541,909 invoice lines → 17,982 baskets over 2,547 products.

| | |
|---|---|
| Rules discovered | **500** (from 1,534 frequent itemsets) |
| Holdout-validated | **403** — 80.6% replication on future baskets |
| Holdout coverage | **81.7%** of unseen baskets fire at least one rule |
| Lift rank stability | Spearman **ρ = 0.862** in-sample vs holdout |
| Confidence drift | **−0.033** (MAE 0.075) — mild selection optimism |
| Objective, baseline → champion | 0.7157 → **0.8029** (**+11.9%**) over 60 evaluations |
| Hypotheses corrected for | **6,484,662** |
| Swap-randomised null | **0 rules** from 2,933,270 margin-preserving swaps |

The strongest rules are the kind a merchandiser recognises immediately:
`{HERB MARKER THYME} ⇒ {HERB MARKER ROSEMARY}` (confidence 0.93, lift 62.6),
`{JUMBO BAG PINK POLKADOT} ⇒ {JUMBO BAG RED RETROSPOT}` (confidence 0.67, lift 5.8).

---

## The six CRISP-DM phases

### Phase 1 — Business understanding

A UK online giftware retailer wants larger baskets. Three decisions depend on
co-purchase structure: the cross-sell surface, which bundles to discount, and
catalogue adjacency. Success criteria were fixed **before** modelling, so the
objective function is not reverse-engineered from whatever the model produced:
rules must replicate out of sample (≥60%), the rule set must fire on real
baskets (≥40% coverage), rules must survive multiplicity correction, beat a
structure-free null, and stay legible (≤3 antecedent items).

Explicitly out of scope: causal claims, individual-level targeting, and any
market outside the training period without a refit.

### Phase 2 — Data understanding

Descriptive only — nothing is dropped here, the point is to *quantify* what is
broken. Nine named quality gates run over the raw log; several are **expected**
to warn, and each warning maps to a specific Phase 3 drop step:

| Gate | Finding |
|---|---|
| `description_complete` | 1,454 rows with no product name |
| `customer_id_complete` | 135,080 rows (24.9%) — guest checkouts |
| `quantity_positive` | 10,624 rows with Quantity ≤ 0 (returns) |
| `no_credit_notes` | 9,288 cancellation invoices (prefix `C`) |
| `no_exact_duplicates` | 5,268 fully duplicated rows |

### Phase 3 — Data preparation

An auditable cleaning ledger: every step is one named predicate with a rationale,
and the row count before and after. 541,909 → 522,286 lines (−3.6%).

Two decisions matter more than the rest:

**Baskets are sets, not counts.** Buying six of an item is one piece of
co-occurrence evidence, not six (Agrawal & Srikant 1994). De-duplicating line
items is what makes support a probability rather than a sales volume.

**The holdout is split on time, not at random.** Webb (2007) shows that testing a
rule on the data that suggested it inflates the false-discovery rate without
bound. For retail we go further: a random split would let one December basket
vouch for a rule found in another December basket. The temporal split (explore
ends 2011-09-27, holdout runs to 2011-12-09) is the *harder* test — the holdout
spans the Christmas peak the training window never sees.

### Phase 4 — Modeling

**FP-Growth, Apriori and Eclat implemented from scratch** (`src/armlab/mining/`).
Not for novelty: the AutoResearch loop needs exact support counts, a reusable
itemset lattice, and instrumentation (tree nodes, conditional trees, candidates
touched) that library APIs do not expose. They are exact algorithms, so
`tests/test_mining.py` asserts all three return **identical** itemset→count maps
as a brute-force oracle — that agreement is the correctness proof.

At the champion threshold (min count 125):

| Algorithm | Time | Itemsets | Agrees |
|---|---|---|---|
| FP-Growth | **0.67 s** | 1,534 | — |
| Eclat | 0.92 s | 1,534 | identical |
| Apriori | 80.70 s | 1,534 | identical |

Every rule carries **32 interestingness measures**, an exact Fisher p-value, a
search-space-corrected q-value, Webb's productivity verdict, and its holdout
behaviour.

### Phase 5 — Evaluation

Association rules have no accuracy score, which is why so many market-basket
projects ship rules that do not replicate. The rule set is treated as a
predictive artefact and scored on data the miner never saw — replication,
lift-rank stability, coverage, firing precision, drift — plus:

* **Fisher's exact test** on each 2×2 table (exact, not asymptotic: retail cell
  counts are small).
* **Benjamini–Hochberg FDR** against all **6,484,662** rules the search *could*
  have examined, not the 530 that cleared the thresholds (Webb 2007, §4).
* **Webb's productivity filter** — a rule must beat every one of its own
  generalisations.
* **Swap randomisation** (Gionis et al. 2007) — 2.9M swaps preserving basket
  sizes and item frequencies while destroying co-occurrence structure. Mining
  the identical thresholds on the surrogate yields **zero** rules: an empirical
  false-discovery rate of 0.
* **Local sensitivity sweeps** around the champion, so a reviewer can see whether
  it sits on a robust plateau or a lucky spike.

### Phase 6 — Deployment

A model card (Mitchell et al., FAT* 2019) adapted to an unsupervised artefact,
full run provenance, and the rules actually **served**:

```bash
curl -X POST http://127.0.0.1:8931/api/recommend \
  -H 'content-type: application/json' \
  -d '{"items":["HERB MARKER THYME"],"top_k":5}'
```

returns ranked next items with the evidence attached to each (confidence, lift,
Fisher p, holdout confidence, validation status).

---

## AutoResearch

### The objective

Mining has no natural loss, so "lower min-support until the output looks nice" is
the norm. That is not a methodology. Following the multi-objective framing of
Ghosh & Nath (*Information Sciences* 163, 2004), the target is explicit; every
term is normalised to [0,1], higher is better.

| Term | Weight | What it measures | Why |
|---|---|---|---|
| Validated yield | 0.35 | log-saturated count of rules that are replicated **and** FDR-significant **and** productive | Counting raw rules rewards threshold-lowering; counting survivors does not |
| Effect size | 0.25 | median Kulczynski blended with 1 − 1/lift | Kulczynski is null-invariant, so it does not inflate as the catalogue grows |
| Coverage | 0.20 | share of holdout baskets a rule fires on | A precise but inapplicable rule set cannot drive a surface |
| Parsimony | 0.10 | penalises long antecedents and duplicate consequents | 200 variations of one insight is one insight with a reporting problem |
| Compute economy | 0.10 | 1 − normalised wall clock, calibrated to 5× the baseline | The champion must be re-mineable nightly |

### The optimiser

Stochastic hill climbing with an adaptive step, a tabu set over discretised
coordinates (Glover 1989) and random restarts (Selman et al., AAAI 1992).

**Why not Bayesian optimisation?** The landscape is riddled with plateaus —
nudging min-support by 0.001 usually changes nothing, then crosses a support
threshold and changes everything. A GP surrogate fits that badly, and at ~60
evaluations over 5 dimensions most of the budget would go into learning the
surrogate. Stochastic local search is the standard answer for this shape of
landscape (Hoos & Stützle 2004).

Each dimension is mapped to a normalised [0,1] coordinate — support on a log
scale, lift linear, lengths as small integers — so a step of 0.1 means a
comparable move in every dimension. Perturbations **reflect** at the bounds
rather than clipping, which otherwise piles probability mass on the edges.

Every evaluation is appended to `artifacts/autoresearch_ledger.jsonl`
before the next one starts, so a crashed search is still analysable.

### What it found — and the interesting part

The champion mines **pairwise rules only**: support ≥ 0.99%, confidence ≥ 0.36,
lift ≥ 2.91, single-item antecedents.

The search did **not** ignore longer rules — it evaluated all six
(itemset-length, antecedent-length) shapes:

| max itemset | max antecedent | trials | best objective | validated rules at best |
|---|---|---|---|---|
| 2 | 1 | 41 | **0.8012** | 403 |
| 3 | 1 | 1 | 0.7725 | 265 |
| 3 | 2 | 6 | 0.7517 | **420** |
| 4 | 1 | 3 | 0.7237 | 176 |
| 4 | 2 | 6 | 0.7509 | 415 |
| 4 | 3 | 3 | 0.7157 | 184 |

The `3/2` configuration found **more** holdout-validated rules (420 vs 403) and
still lost, because the parsimony and compute terms charge for longer antecedents
and slower mines. **Whether that is the right trade is a business decision
encoded in the weights, not a fact about the data** — raise
`w_validated_yield` and the champion changes shape. The dashboard says so on the
AutoResearch page rather than presenting the champion as inevitable.

---

## Honest caveats

These are stated in the dashboard too, not buried here:

* **Productivity is vacuous for this champion.** Webb's test compares a rule to
  its proper generalisations, and a one-item antecedent has none. "530/530
  productive" means the test did not apply, not that every rule passed a
  demanding filter.
* **Every rule clears the FDR correction**, because the median Fisher p-value is
  1.9 × 10⁻¹⁸⁶ — at these support counts the hypergeometric tail vanishes and
  many p-values underflow to exactly 0 in double precision. The correction bites
  at lower support thresholds, which the sensitivity sweep reaches.
* **No restart was consumed** in this run: the 60-evaluation budget was spent
  inside the first search. The restart machinery is exercised by
  `tests/test_autoresearch.py`, not by this particular run.
* **The population is overwhelmingly UK.** Rules describe the UK market; the
  config exposes a `countries` filter to mine a homogeneous market instead.
* **The support floor excludes the long tail by construction** — the rule set is
  silent about most of the catalogue, by design.
* Wholesale-sized baskets were removed, so rules do not describe B2B purchasing.

---

## Interestingness measures and their properties

32 measures are computed for every rule. The dashboard renders a **property
matrix** over Piatetsky-Shapiro's axioms (P1–P3) and the invariance properties of
Tan, Kumar & Srivastava (KDD 2002) — O1 symmetry, O2 row/column scaling
invariance, O3 inversion invariance, O4 null invariance.

The matrix is **derived numerically**, not transcribed: each cell is a randomised
experiment over 2×2 contingency tables (`src/armlab/metrics/properties.py`).
Re-running `make properties` reproduces it exactly, and it recovers the published
findings — only the odds ratio and Yule's Q/Y are scaling-invariant; the
null-invariant family is cosine, Jaccard, all-confidence, max-confidence and
Kulczynski. Support and lift are *not* null-invariant, which is precisely why a
lift-only ranking degrades as a catalogue grows.

`tests/test_properties.py` asserts those literature facts, so a drift in any
measure's implementation fails the build.

---

## Repository layout

```
04_associative_pattern_mining/
├── config/experiment.yaml        every knob; its SHA-256 is the run identity
├── src/armlab/
│   ├── config.py                 typed config + provenance fingerprinting
│   ├── pipeline.py               the six-phase orchestration
│   ├── cli.py                    run | serve | properties | profile
│   ├── data/                     acquire · profile · prepare  (Phases 2-3)
│   ├── mining/                   fpgrowth · apriori · eclat · rules  (Phase 4)
│   ├── metrics/                  interestingness (32) · properties (derived)
│   ├── stats/                    Fisher · BH-FDR · productivity · swap null
│   ├── evaluation/               temporal holdout scoring  (Phase 5)
│   ├── autoresearch/             space · objective · hillclimb · ledger
│   ├── api/server.py             FastAPI: artifacts + /api/recommend
│   └── web/                      the dashboard (no CDN; runs offline)
├── tests/                        60 tests
├── tools/validate_palette.py     chart-palette accessibility validator
└── artifacts/                    generated; rebuild with `make run`
```

## Reproducibility

Every artifact embeds a provenance block: a SHA-256 fingerprint of the full
config, the git SHA, Python version, platform, and both random seeds. Two result
sets are comparable only if their fingerprints match. The raw `.xlsx` is cached
as Parquet on first load (40 s → 0.2 s), and AutoResearch memoises FP-Growth by
`(min_count, max_len)` — 27 of 60 trials were cache hits here, with the original
wall clock replayed so the cost term still reflects real cost.

## Testing

```bash
make test     # 60 tests
```

> The folder is self-contained. `make` and the shell scripts use a local `.venv`
> if one exists, otherwise the shared `../.venv` in `part2/`.

The suite checks the things that would otherwise be taken on trust: the three
miners against a brute-force oracle and each other, downward closure, every
measure against hand-computed values and edge cases, the property matrix against
the published literature, Fisher's test against `scipy.stats.fisher_exact`,
swap randomisation actually preserving both margins, the hill climber finding a
known synthetic optimum and being reproducible, that no artifact contains
`NaN`/`Infinity` (invalid JSON that silently breaks the dashboard), and that the
recommender never suggests an item already in the basket.

## Dashboard

Twelve views, one per CRISP-DM concern, served by FastAPI. No CDN dependency —
charts are hand-built SVG so the dashboard runs offline. Light and dark themes;
the categorical palette is validated for colour-vision deficiency by
`tools/validate_palette.py` (a Python port of the reference validator: worst
adjacent CVD ΔE 9.1 light / 8.4 dark, normal-vision floor 19.6 / 19.3), and every
chart ships a table view for the values colour alone cannot carry.

## References

* Agrawal & Srikant (1994). *Fast Algorithms for Mining Association Rules.* VLDB.
* Han, Pei & Yin (2000). *Mining Frequent Patterns without Candidate Generation.* SIGMOD.
* Zaki (2000). *Scalable Algorithms for Association Mining.* IEEE TKDE.
* Brin, Motwani, Ullman & Tsur (1997). *Dynamic Itemset Counting and Implication Rules.* SIGMOD.
* Piatetsky-Shapiro (1991). *Discovery, Analysis and Presentation of Strong Rules.*
* Tan, Kumar & Srivastava (2002). *Selecting the Right Interestingness Measure for Association Patterns.* KDD.
* Geng & Hamilton (2006). *Interestingness Measures for Data Mining: A Survey.* ACM Computing Surveys.
* Webb (2007). *Discovering Significant Patterns.* Machine Learning 68:1–33.
* Benjamini & Hochberg (1995). *Controlling the False Discovery Rate.* JRSS-B.
* Gionis, Mannila, Mielikäinen & Tsaparas (2007). *Assessing Data Mining Results via Swap Randomization.* ACM TKDD.
* Wu, Chen & Han (2010). *Re-examination of Interestingness Measures in Pattern Mining.* ACM TKDD.
* Omiecinski (2003). *Alternative Interest Measures for Mining Associations.* IEEE TKDE.
* Ghosh & Nath (2004). *Multi-objective Rule Mining using Genetic Algorithms.* Information Sciences 163.
* Hoos & Stützle (2004). *Stochastic Local Search: Foundations and Applications.*
* Selman, Levesque & Mitchell (1992). *A New Method for Solving Hard Satisfiability Problems.* AAAI.
* Mitchell et al. (2019). *Model Cards for Model Reporting.* FAT*.
* Chen, Sain & Guo (2012). *Data mining for the online retail industry.* J. Database Marketing 19:197–208.
