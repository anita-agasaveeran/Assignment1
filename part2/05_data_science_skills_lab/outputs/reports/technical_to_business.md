# Technical to business translation

*Produced by the `technical-to-business-translator` skill. Audience persona: a commercial VP who owns the decision but not the method.*

## Business version (lead with this)

Our model correctly ranks who survived about 79% of the time on data it has never seen. In testing it looked better than that, so we are reporting the lower, more honest number. It works well for women, and much less well for men, where it catches under half of the cases it should. We would not yet use its confidence scores to make an automatic decision; we would use its ranking to prioritise a human review.

## Translation table

| Technical | Business |
|---|---|
| PR-AUC 0.791 | correctly ranks survivors about 79% of the time |
| cross-validated 0.867 vs held-out 0.791 | it looked better in testing than it is; we report the lower number |
| recall 0.417 on the male stratum | it misses more than half the men it should catch |
| Brier score 0.158, poorly calibrated | its confidence scores are not trustworthy enough to automate a decision |
| hyperparameter optimisation overfit the folds | we tuned it against the same data we measured it on, which flatters the result |

## Appendix - original technical text

The HistGradientBoostingClassifier pipeline achieved a cross-validated PR-AUC of 0.867 (sigma 0.022) under stratified 5-fold CV, though the held-out estimate regressed to 0.791, indicating the hyperparameter optimisation overfit the validation folds. Slice-level diagnostics reveal heteroscedastic performance: recall on the male stratum is 0.417 versus 0.867 on the female stratum, and the Brier score of 0.158 implies the posterior probabilities are poorly calibrated for downstream thresholding.
