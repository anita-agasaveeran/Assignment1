# How we did this, and what it can and cannot tell you

*Produced by the `methodology-explainer` skill. Three tiers: read the one that matches you.*

## Tier 1 - Executive (60 seconds)

We took two well-known public datasets and ran a complete, disciplined analysis over
each: one predicting who survived the Titanic, one understanding an online retailer's
customers. The point was to test 46 analysis "skills" on real data, not to discover
something new about either dataset.

The survival model ranks passengers correctly about **79%** of the
time on data it has never seen. That is decent but not automatic-decision grade, and it
is noticeably worse for men than for women.

## Tier 2 - Business analyst

**Data.** Kaggle Titanic (891 passengers, 12 columns) and Kaggle Online Retail
(541,909 transaction lines, Dec 2010 - Dec 2011). Both hashed and version-pinned.

**Method.** CRISP-DM, all six phases, on both datasets in parallel. For the model:
the data was split into training and holdout *before* any cleaning, so nothing learned
from the holdout could leak backwards. Preprocessing lives inside a scikit-learn
Pipeline, so it is re-fit inside every cross-validation fold. Hyperparameters were
searched with Optuna over 5 parameters.

**Metric.** PR-AUC rather than accuracy. With 38% survivors, a model that predicts
"nobody survived" scores 62% accuracy while being useless. PR-AUC cannot be fooled
that way.

**Result.** Cross-validated PR-AUC 0.867; holdout
0.791. We report the second number.

## Tier 3 - Technical peer

Stratified 5-fold CV, seed 42. `ColumnTransformer` over 10 numeric / 4 categorical /
3 binary features; `HistGradientBoostingClassifier`. Optuna TPE, 60 trials, log-scale
sampling on learning rate and L2. Ticket-prefix target encoding is cross-fitted over
5 folds with smoothing 10; in-fold encoding correlates 0.262
with the target versus 0.180 out-of-fold, and the
difference is leakage. Decision threshold 0.465 selected on a
validation split, never re-tuned on the holdout. Brier 0.158;
calibration error 0.1024 over 8 quantile bins.

## Limitations - read this before using any of it

1. **The holdout is 179 rows.** The gap between the cross-validated
   0.867 and the holdout 0.791 is consistent
   with sampling noise at that size. Neither number should be quoted to three decimals
   as if it were precise.
2. **Subgroup performance is uneven.** Recall is
   0.42 for men versus
   0.87 for women. A
   single global threshold treats those groups as if they were the same. They are not.
3. **Probabilities are not well calibrated** (Brier 0.158). Use the
   model's *ranking*, not its stated confidence.
4. **Families share tickets and cabins.** A random split can put relatives on both sides
   of it. `GroupKFold` on ticket would be more conservative; we did not do that, and it
   may mean the numbers above are slightly optimistic.
5. **The retail dataset is one UK gift wholesaler in 2011.** Nothing here generalises to
   e-commerce at large, and the benchmark comparisons are indicative only.
