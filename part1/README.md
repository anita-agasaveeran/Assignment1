# Assignment 1 — Exoplanet Habitable Zone Analysis

## Predicting Radiative Habitable-Zone Membership from Stellar and Orbital Features

A CRISP-DM data science study of 4,575 confirmed exoplanets, conducted end-to-end through a staged, chunked ChatGPT session (GPT-5 Thinking).

> **Scope note:** This is a radiative habitable-zone *screening* analysis. It does not determine whether any exoplanet is actually habitable or inhabited. It predicts whether a detected planet receives a broadly Earth-like amount of stellar energy — a first-pass condition, not proof of habitability.

---

## Project Links

- **Full analysis transcript:** [`transcript/chat_history.pdf`](./transcript/chat_history.pdf) — the complete, chunk-by-chunk ChatGPT conversation this project was built from
- **Medium article:** [Predicting Radiative Habitable-Zone Membership from Stellar and Orbital Features](https://medium.com/@anita.agasaveeran/predicting-radiative-habitable-zone-membership-from-stellar-and-orbital-features-3481e68d6a66)
- **Source dataset:** [Kaggle — All Exoplanets Dataset](https://www.kaggle.com/datasets/shivamb/all-exoplanets-dataset) (`all_exoplanets_2021.csv`, sourced from the NASA Exoplanet Archive)

---

## Executive Summary

Stellar temperature, radius, and mass had unstable *individual* coefficients once entered together, because they describe overlapping aspects of the same star — hotter stars tend to be larger and more massive. Yet removing all three as a group caused a large drop in out-of-host predictive performance. The individual coefficients were uncertain; the joint stellar-structure signal was not.

More flexible machine-learning models also performed *worse*, not better. A physics-informed, regularized logistic model reached an out-of-host PR-AUC of **0.596** [0.447, 0.789], compared with **0.475** for a constrained random forest and **0.366** for constrained gradient boosting. With only 26 host systems supplying positive habitable-zone examples, the simpler model used the available information more efficiently than either tree ensemble.

The scientific pattern was more nuanced than "longer orbits are better." Very short orbits were usually too hot; very long orbits could be too cold. The favorable region was an intermediate orbital band whose location depended on the host star — cooler stars could support moderate irradiation at shorter periods, while hotter, larger stars generally required more distant orbits.

Finally, the large raw gap between transit and radial-velocity discoveries was mostly a **covariate-shift** story: the two methods sample different kinds of systems. Once orbital and stellar measurements were known, the discovery-method label itself added no predictive value — it did not eliminate detection bias, it showed *where* the bias entered the data.

---

## Research Questions

1. Which stellar and orbital features predict radiative habitable-zone membership?
2. Are planets with moderate orbital periods around cooler stars more likely to be candidates?
3. Does an explicit log-period × stellar-temperature interaction improve prediction?
4. How do transit, radial-velocity, and imaging selection effects influence the observed relationship?
5. Do nonlinear tree ensembles (random forest, gradient boosting) outperform a regularized, physics-informed logistic model?

---

## What "Habitable Zone" Means Here

A planet is labeled a **radiative habitable-zone (HZ) candidate** when its reported insolation flux falls between 0.35 and 1.75 times the flux Earth receives from the Sun — the orbital region where a planet is neither roasted (too close) nor frozen (too far). This label does **not** measure atmosphere, surface composition, water, climate stability, or biological activity. It is a screening condition, not a habitability determination.

---

## Dataset

`data/raw/all_exoplanets_2021.csv` contains 4,575 confirmed exoplanets and 23 columns, including orbital period, orbital size, planet mass and eccentricity, reported insolation and equilibrium temperature, stellar temperature/radius/mass/metallicity/surface gravity, and discovery method/year/facility.

Only 370 planets have a reported insolation value. The primary modeling cohort — after requiring complete core stellar predictors — contains **366 planets from 264 host systems**, including **32 positive planets belonging to only 26 positive host systems**.

**Key data limitations:**
- Reported insolation flux is missing for 91.9% of planets.
- Planet radius is absent from the catalog entirely, preventing a conventional Earth Similarity Index.
- Planet mass and eccentricity are heavily missing and discovery-method dependent.
- No directly imaged planet has the complete labeled data needed to enter the primary model.
- The catalog contains confirmed discoveries, not a random sample of all planetary systems.

---

## Methodology (CRISP-DM)

The analysis was run as a staged, chunked ChatGPT (GPT-5 Thinking) session, prompted to work step-by-step through the CRISP-DM framework and to critique its own choices before proceeding at each stage. The full back-and-forth is preserved in [`transcript/chat_history.pdf`](./transcript/chat_history.pdf).

| Phase | Summary |
|---|---|
| **Business Understanding** | Defined a narrow, measurable radiative screening target instead of claiming actual habitability. |
| **Data Understanding** | Audited every column, unit, missingness pattern, and discovery-method composition; established that only 26 host systems carry a positive label. |
| **Data Preparation** | Preserved unknown outcome values rather than back-filling them as negative; avoided broad statistical imputation for high-missingness physical fields; built leakage-safe feature sets. |
| **Exploratory Analysis** | Examined the raw period–temperature pattern, missingness structure, feature collinearity, physics-aware outlier handling, and mass–period clustering. |
| **Modeling** | Naive baselines → targeted period–temperature logistic → full stellar-structure logistic → method-adjusted logistic → constrained random forest → constrained gradient boosting → continuous log-insolation regression cross-check. |
| **Evaluation** | Repeated (15×6) host-grouped cross-validation, whole-host bootstrap uncertainty intervals, and paired comparisons on identical held-out hosts throughout, using PR-AUC/recall/F1/MCC/log-loss rather than raw accuracy. |

**Leakage controls:** the primary HZ classifier never sees reported or derived insolation, the HZ label itself, equilibrium temperature, or any feature directly constructed from the target boundaries.

**Validation design:** 15 repeats × 6 host-grouped outer folds = 90 outer test folds, with all preprocessing, configuration selection, and thresholding nested inside training folds only. Grouping by host star prevents planets sharing a star from leaking across train/test.

---

## Model Results

| Model | PR-AUC | Recall | Precision | F1 | MCC | Log-loss |
|---|---:|---:|---:|---:|---:|---:|
| Prevalence baseline | 0.077 | 0.000 | 0.000 | 0.000 | 0.000 | 0.271 |
| **Full stellar ridge logistic (selected model)** | **0.596** | **0.802** | **0.542** | **0.647** | **0.626** | **0.140** |
| Constrained random forest | 0.475 | 0.691 | 0.535 | 0.603 | 0.571 | 0.162 |
| Constrained gradient boosting | 0.366 | 0.786 | 0.337 | 0.472 | 0.456 | 0.183 |

All metrics are out-of-host, from the same 90 repeated grouped folds; full uncertainty intervals are in `results/chunk5_full_stellar_model_and_method_adjustment/model_performance_host_bootstrap.csv` and `results/chunk7_gradient_boosting_and_flux_regression/`.

**Selected model:** the full stellar-structure ridge-logistic model (log orbital period + period curvature + stellar temperature + stellar radius + stellar mass + a targeted period × temperature term), *without* a discovery-method term — adding one slightly worsened both PR-AUC and log-loss. Both tree ensembles were deliberately constrained (shallow depth, minimum leaf size scaled to positive-host count) and selected via nested one-standard-error rules, not an open hyperparameter search — the negative result for both is treated as a genuine finding, not a failed attempt to find a winner.

---

## Key Findings

- **Orbital period** was the strongest individual predictor and behaved almost independently of the stellar features (correlation ≈ 0.00–0.02).
- **Stellar temperature, radius, and mass** were extremely collinear (pairwise correlations 0.95–0.98). Individually their coefficients were unstable — but removing all three as a block reduced PR-AUC by ~0.295 and worsened log-loss by ~0.058, confirming the joint signal was real even when individual attribution was not.
- The **period × temperature interaction** moved in the expected direction (favorable period shifts outward for hotter stars) but its 95% uncertainty interval touched zero — reported as *suggestive, not conclusive*.
- **Discovery-method covariate shift:** radial-velocity discoveries had a much higher raw HZ rate (20.3%) than transit discoveries (5.6%), but a model given only physical features (no method label) reproduced almost the same gap. Adding a method indicator did not improve the model. Direct imaging could not be evaluated at all — no imaging planet has the labeled data required.
- **Simpler beat more flexible:** with only 26 positive host systems, the regularized logistic model out-generalized both a constrained random forest and constrained gradient boosting on identical held-out hosts.
- **Continuous flux regression** reconstructed reported insolation with R² ≈ 0.997 — but this is a leakage-adjacent physical consistency check (flux is calculated from closely related predictors), not independent evidence of habitability prediction.

---

## Repository Structure

```text
.
├── README.md
├── data/
│   ├── raw/
│   │   └── all_exoplanets_2021.csv                         # Source Kaggle/NASA Exoplanet Archive catalog
│   └── processed/
│       ├── exoplanets_prepared_no_statistical_imputation.csv   # Cleaned + feature-engineered, no global imputation
│       └── exoplanets_chunk3_feature_outlier_cluster.csv       # + outlier flags and mass–period cluster assignments
├── transcript/
│   └── chat_history.pdf                                     # Full staged ChatGPT (GPT-5 Thinking) analysis session
├── figures/
│   └── public/                                              # Presentation-ready infographics for the Medium article
├── results/
│   ├── chunk2_data_preparation/                             # Raw missingness/EDA baseline, imputation decision register
│   ├── chunk3_features_outliers_clustering/                 # Collinearity, physics-aware outliers, mass–period clusters
│   ├── chunk4_period_temperature_logistic/                  # Targeted period × temperature interaction model
│   ├── chunk5_full_stellar_model_and_method_adjustment/     # Selected model, coefficient shifts, method-bias test
│   ├── chunk6_random_forest/                                # Constrained random forest, permutation importance
│   └── chunk7_gradient_boosting_and_flux_regression/        # Constrained gradient boosting + continuous flux regression
```

Each `results/chunkN_*/` folder holds that stage's CSV result tables and, where generated, the corresponding chart PDFs — mirroring the order the analysis was actually run in, as documented in the transcript.

---

## Figures

`figures/public/` contains five presentation-ready graphics built for the Medium article:

- **Radiative Habitable-Zone Likelihood vs. Orbital Period** — the moderate-period ("inverted-U") hypothesis explained visually
- **Discovery-Method Covariate Shift** — raw HZ prevalence by method vs. physically-adjusted prevalence
- **Typical Orbits and Host Stars: Candidates vs. Non-Candidates** — median period/temperature comparison table
- **How Stellar and Orbital Features Relate to Each Other** — the stellar-collinearity correlation table
- **Coefficient redistribution and joint stellar-block effect** — the central data-science finding of the project

Additional analysis-stage figures (raw scatter panels, collinearity heatmaps, cluster plots, classifier comparisons, response surfaces) are in each `results/chunkN_*/` folder as PDFs.

---

## Main Conclusions

- Orbital period is the strongest individual predictor; the relationship is curved, not monotonic — very short and very long periods are both less favorable than an intermediate band.
- The favorable period depends on the host star; stellar temperature, radius, and mass matter collectively even though individual coefficients are unstable.
- The period × temperature interaction is directionally consistent with expectation but statistically suggestive rather than proven.
- Discovery method creates strong covariate shift but adds little residual predictive information once physical features are known; imaging remains unsupported by the labeled data.
- The physics-informed ridge-logistic model outperforms both constrained random forest and constrained gradient boosting.
- Continuous flux regression is highly accurate but leakage-adjacent — a consistency check, not independent proof of habitability.
- The results predict radiative HZ *candidacy*, not actual habitability.

## Claims This Project Does Not Make

This project does not claim that any listed planet is inhabited or genuinely habitable, that it estimates how common habitable planets are in the galaxy, that radial-velocity surveys intrinsically find more habitable planets, that directly imaged planets have a lower HZ rate, that any one stellar feature has a uniquely causal effect, that the period × temperature interaction is definitively proven, or that the log-flux regression's R² of 0.997 represents near-perfect habitability prediction.

**Editorial test:** before publishing any sentence containing the word "habitable," swap it for "receives between 0.35 and 1.75 times Earth's stellar flux." If the sentence becomes false or misleading, it's overclaiming.

---

## Author

**Anita Agasaveeran**
Master's Program in Data Science (MSSE), CMPE 255
San José State University — Fall 2026
