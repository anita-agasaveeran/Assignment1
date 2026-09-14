# Analysis brief - Titanic survival

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
