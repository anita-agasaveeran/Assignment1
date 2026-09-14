# AutoResearch — automated hill climbing over the literature

`python -m slm.autoresearch --study hillclimb-v1`

## The idea

Every candidate move in the search is a **claim from a published paper**, and a
trial is an attempt to reproduce that claim at our scale on our data. The search
starts from a deliberately dated 2019-era baseline — LayerNorm, learned absolute
positions, GELU feed-forward, full multi-head attention, untied embeddings,
dropout 0.1, AdamW with cosine decay — and climbs.

A successful climb therefore **rediscovers** the modern recipe from the
literature rather than assuming it. That is the whole point: it converts the
architecture section from a list of assertions into a set of measurements.

## The procedure

**Round 0 — establish the noise floor.** Train the baseline with *n* different
seeds and record the standard deviation σ of held-out loss. Nothing is judged
before this exists. Without it, "intervention X improved loss by 0.004" is
unfalsifiable.

**Rounds 1..R — greedy expansion.** Re-measure the incumbent at this round's
seed, then train every remaining candidate on top of the incumbent using that
same seed and the same fixed evaluation batches. Accept the single best candidate
**only if** its improvement exceeds `max(min_delta, z_accept·σ)`. Otherwise the
climb has converged and the search stops.

**Final — champion validation.** Retrain champion and baseline across fresh
seeds and report the mean improvement and Cohen's *d*. This checks that the climb
found a real effect rather than fitting its own search path.

## Why each design choice is there

| Choice | Reason |
|---|---|
| Noise floor measured first | An improvement smaller than seed variance is not an improvement |
| Paired trials (same seed within a round) | The variance under test is seed variance, not batch luck |
| Fixed evaluation batches, run-seed-independent | Two trials are scored on identical held-out windows |
| Low-fidelity proxy (4L × d256, 300 steps) | Successive-halving logic (arXiv:1603.06560) reduced to its simplest usable form |
| Rejected trials retained in the DB | Null results are results; they are the honest half of the output |
| Champion re-validated across seeds | Guards against over-fitting the search |
| `depends_on` between interventions | e.g. RoPE base only tested once RoPE is in the incumbent |

## Statistical reasoning

Acceptance requires improvement > `z_accept · σ` where σ is measured
seed-to-seed standard deviation of the baseline. With `z_accept = 1.0` this is a
deliberately permissive bar for a *screening* procedure, not a confirmatory
test — and the champion validation step is what supplies the confirmatory
evidence.

This is a multiple-comparison setting: ~17 candidates per round means some
apparent winners will be noise. Two things mitigate it: the effect sizes that
actually get accepted are typically many σ (z of −8 or larger), far outside
anything multiple comparisons would manufacture; and the champion is re-validated
on fresh seeds at the end.

## Known limitations — stated, not hidden

- **Greedy search cannot see interactions.** Techniques that only pay off in
  combination are invisible to it. The accepted *order* is published because that
  is the axis along which this bias acts.
- **Proxy-scale results may not transfer.** A 3M-parameter, 300-step proxy is not
  a 9M-parameter, 5,000-step run, let alone a 7B one. The champion is retrained at
  full scale, but a technique rejected on the proxy is never re-tested at scale.
- **Several candidates are stability interventions.** QK-norm, z-loss, and logit
  soft-capping exist to prevent divergence that appears at large scale. At this
  scale there is no instability for them to prevent, so they can only show their
  cost and none of their benefit. Rejecting them here says *nothing* about their
  value at scale — and this is precisely the sort of conclusion an automated
  search will state confidently and wrongly if a human does not annotate it.
- **One corpus, one scale, one seed family.** No claim of generality.

## Reading the output

On the AutoResearch dashboard tab:

- **Bars inside the grey band** are within seed noise — explicitly *not* evidence.
- **"significant, not best"** means the move cleared the bar but another move
  cleared it by more this round; it stays in the candidate pool for later rounds.
- **"worse"** means the move cleared the bar in the wrong direction. `hi_lr`
  under AdamW (round 1, z = +14) is the intended demonstration of a genuine risk
  trial: the same move became a win once Muon was the optimizer (round 5).
- **"diverged"** means a non-finite loss. That is recorded as a result, not
  swallowed as a crash.
- Each row carries the **paper's claim** and the **predicted signal** written
  *before* the trial ran, next to the measured delta. That pairing is the point
  of the whole exercise.
