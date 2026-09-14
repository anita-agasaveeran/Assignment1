# CMPE 255 — Assignment 1

**Anita Agasaveeran** · San José State University · MSSE, Fall 2026

This repository contains both parts of Assignment 1 for CMPE 255 (Data Mining). Each part
has its own README with full details; this page is the entry point.

| Part | Folder | What it is | Deliverables |
|---|---|---|---|
| **1** | [`part1/`](part1/) | A CRISP-DM data science study on the Kaggle *All Exoplanets* dataset, run end-to-end through a staged ChatGPT (GPT-5 Thinking) session | [README](part1/README.md) · [chat transcript](part1/transcript/chat_history.pdf) · [Medium article](https://medium.com/@anita.agasaveeran/predicting-radiative-habitable-zone-membership-from-stellar-and-orbital-features-3481e68d6a66) |
| **2** | [`part2/`](part2/) | Six data science experiments from [dlmastery/data_science_examples](https://github.com/dlmastery/data_science_examples) replicated with Claude Code from the repo's verbatim prompts | [README](part2/README.md) · [YouTube walkthrough](https://youtu.be/GApvUAs4Mow) |

---

## Part 1 — Exoplanet Habitable-Zone Analysis

Predicts whether a confirmed exoplanet receives a broadly Earth-like amount of stellar energy
(a *radiative habitable-zone* screening label) from stellar and orbital features, across 4,575
planets. The analysis walks through all six CRISP-DM phases with host-grouped cross-validation
and leakage controls, and compares a physics-informed regularized logistic model against random
forest and gradient boosting.

Headline result: the simpler logistic model won (out-of-host PR-AUC **0.596** vs 0.475 for random
forest and 0.366 for gradient boosting) — with only 26 host systems carrying positive labels, the
flexible models had too little signal to exploit. Details, figures and per-phase results are in
[`part1/README.md`](part1/README.md).

## Part 2 — Replicated Data Science Experiments

Six projects rebuilt in Claude Code from the prompts in the source repo's `PROMPTS.md`:

| # | Project | One-liner |
|---|---|---|
| 0 | [Dynamic todo workspace](part2/00_dynamic_todo_workspace/) | Full-stack todo app on stock Python — SQLite API, keyboard-first SPA, live sync over SSE |
| 1 | [NYC taxi trip prediction](part2/01_nyc_taxi_trip_prediction/) | Trip-duration model on 1.16M real trips with interactive map and uncertainty bands |
| 2 | [Nano LLM transformer](part2/02_nano_llm_transformer/) | 16.9M-param language model + chatbot trained from scratch on a laptop GPU, with AutoResearch A/B search over architectural primitives |
| 3 | [Customer segmentation clustering](part2/03_customer_segmentation_clustering/) | Hill-climbing search over 3.4M clustering pipelines on credit-card data, with random-search control |
| 4 | [Associative pattern mining](part2/04_associative_pattern_mining/) | Market-basket rules on Online Retail, validated on a temporal holdout with FDR correction and a swap-randomised null |
| 5 | [Data science skills lab](part2/05_data_science_skills_lab/) | All 46 skills from two agent-skill packs exercised across CRISP-DM on Titanic and Online Retail |

Every project follows CRISP-DM, ships a dashboard rendered from its run artifacts, and reports
negative results as such. See [`part2/README.md`](part2/README.md) for the prompt-to-project
mapping, and the [video walkthrough](https://youtu.be/GApvUAs4Mow) for a tour of each project.

---

## Repository layout

```
Assignment1/
├── README.md          ← this file
├── part1/             Exoplanet CRISP-DM study (data, figures, results, transcript)
└── part2/             Six replicated experiments, one folder each
```

Raw datasets, model checkpoints and virtual environments are git-ignored; each project's README
explains how to re-download or rebuild them.
