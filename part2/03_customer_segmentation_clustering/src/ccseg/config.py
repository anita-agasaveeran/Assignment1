"""CRISP-DM Phase 1 - Business Understanding, plus global run configuration.

Everything a reviewer needs to judge *why* this project exists lives here, so the
dashboard can render the charter verbatim instead of paraphrasing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DASHBOARD = ROOT / "dashboard"
for _p in (DATA_RAW, DATA_PROCESSED, ARTIFACTS, REPORTS, DASHBOARD):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 20260901

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
KAGGLE_DATASET = "arjunbhasin2013/ccdata"
KAGGLE_FILE = "CC GENERAL.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/arjunbhasin2013/ccdata"
ID_COLUMN = "CUST_ID"

# --------------------------------------------------------------------------- #
# Phase 1: Business Understanding
# --------------------------------------------------------------------------- #
BUSINESS_CHARTER = {
    "problem": (
        "A card issuer runs one undifferentiated marketing programme across its entire "
        "active book. Campaign response is measured in aggregate, so spend is allocated "
        "uniformly to ~9k cardholders whose behaviour is anything but uniform."
    ),
    "business_objective": (
        "Discover a small, stable set of behavioural segments in six months of card "
        "activity, so credit-line, retention and cross-sell offers can be targeted per "
        "segment rather than broadcast."
    ),
    "data_mining_goal": (
        "Unsupervised partitioning of cardholders into k segments (2 <= k <= 12) that are "
        "internally compact, well separated, reproducible under resampling, and "
        "describable in plain language to a marketing stakeholder."
    ),
    "why_unsupervised": (
        "No labelled 'segment' target exists and none can be manufactured without begging "
        "the question. Success is therefore judged by internal validity, replication "
        "stability and business interpretability - never by accuracy."
    ),
    "success_criteria": [
        {"id": "SC-1", "criterion": "Silhouette (Rousseeuw 1987) >= 0.25 on held-out full data",
         "rationale": "Above the 0.25 'weak but real structure' band; behavioural data rarely exceeds 0.5."},
        {"id": "SC-2", "criterion": "Bootstrap cluster-wise Jaccard >= 0.60 for every retained segment",
         "rationale": "Hennig (2007) calls <0.60 'dissolved'; a segment we cannot reproduce cannot be budgeted against."},
        {"id": "SC-3", "criterion": "Mean bootstrap ARI (Hubert & Arabie 1985) >= 0.60 for the partition",
         "rationale": "Partition-level replication; guards against a lucky seed."},
        {"id": "SC-4", "criterion": "No segment below 3% of the book",
         "rationale": "A segment too small to staff a campaign for is an outlier bucket, not a market."},
        {"id": "SC-5", "criterion": "Surrogate decision tree (depth<=4) reproduces >= 85% of assignments",
         "rationale": "If a 4-rule tree can restate the model, marketing can act on it without the model."},
        {"id": "SC-6", "criterion": "Inductive scoring path < 5 ms/record at p95",
         "rationale": "Segment must be assignable in the offer-decisioning service, not just in a notebook."},
    ],
    "stakeholders": [
        {"role": "Head of Portfolio Marketing", "needs": "Named, sized, actionable segments and an offer per segment."},
        {"role": "Credit Risk", "needs": "Assurance that segments are not a proxy for a protected attribute."},
        {"role": "ML Platform", "needs": "A deterministic, versioned, monitorable scoring artifact."},
    ],
    "constraints_and_risks": [
        "Six months of behaviour only - no seasonality can be modelled, and TENURE is near-constant (87% at 12).",
        "The file carries no demographics; segments are behavioural, and must not be presented as customer identity.",
        "Internal validity indices are biased toward small k (Milligan & Cooper 1985), so the search objective is "
        "explicitly regularised for balance and minimum segment size rather than left to maximise silhouette alone.",
        "Monetary features are heavy-tailed and unbounded; without a variance-stabilising transform the solution "
        "collapses to 'one whale cluster plus everyone else'.",
    ],
}

# --------------------------------------------------------------------------- #
# Search / evaluation budget
# --------------------------------------------------------------------------- #
@dataclass
class SearchConfig:
    """Budget and hyperparameters for the AutoResearch hill-climbing optimiser."""
    max_evals: int = 420              # total pipeline evaluations (shared by HC and the RS control)
    restarts: int = 8                 # random restarts -> iterated local search
    neighbours_per_step: int = 8      # neighbours sampled per local step
    strategy: str = "stochastic"      # 'stochastic' (first-improvement) | 'steepest'
    sideways_patience: int = 2        # accepted plateau moves before declaring a local optimum
    tabu_size: int = 48               # short-term memory of visited configs (Glover 1986)
    search_subsample: int = 4000      # AutoML4Clust: optimise on a subsample, refit champion on full data
    silhouette_sample: int = 2500     # cap on silhouette pairwise cost per evaluation
    min_cluster_frac: float = 0.03    # SC-4; configs violating it are penalised, not discarded
    max_noise_frac: float = 0.35      # density methods labelling >35% noise are penalised
    k_range: tuple[int, int] = (2, 12)
    min_k_business: int = 3           # a binary split is already available from the existing
                                      # high/low-spend rule; the programme needs >=3 treatments
    time_budget_s: float = 1200.0
    seed: int = RANDOM_SEED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["k_range"] = list(self.k_range)
        return d


@dataclass
class EvalConfig:
    """Budget for CRISP-DM Phase 5 (Evaluation)."""
    bootstrap_rounds: int = 40        # Ben-Hur (2002) / Hennig (2007) resampling replicates
    bootstrap_frac: float = 0.80
    gap_b_refs: int = 12              # Monte-Carlo reference draws for the gap statistic
    k_sweep: tuple[int, int] = (2, 12)
    surrogate_depth: int = 4
    seed: int = RANDOM_SEED


@dataclass
class RunConfig:
    search: SearchConfig = field(default_factory=SearchConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    objective: str = "composite"      # 'composite' | 'silhouette' | 'calinski_harabasz' | 'davies_bouldin'
