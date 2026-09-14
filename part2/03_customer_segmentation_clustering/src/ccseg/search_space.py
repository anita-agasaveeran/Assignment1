"""The CASH search space and its neighbourhood structure.

CASH - Combined Algorithm Selection and Hyperparameter optimisation - is the
formulation introduced by Thornton, Hutter, Hoos & Leyton-Brown (Auto-WEKA,
KDD 2013) for supervised learning, and carried over to clustering by
AutoML4Clust (Tschechlov, Fritz & Schwarz, EDBT 2021) and ML2DAC (Treder-Tschechlov
et al., SIGMOD 2023). The space here is wider than the AutoML4Clust space because
it also selects the *preprocessing* chain, which on heavy-tailed financial data
matters more than the choice of clustering algorithm.

The space is conditional: ``k`` is meaningless for HDBSCAN, ``covariance`` only
exists for the mixture model, and so on. ``canonical`` blanks every inactive
dimension before hashing, which is what makes the memoisation cache and the tabu
list correct - without it the optimiser would treat two identical pipelines as
different points and waste budget re-evaluating them.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterator

import numpy as np

# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #
SPACE: dict[str, list[Any]] = {
    # ---- preprocessing (CRISP-DM Phase 3 decisions exposed to the optimiser) ---
    "feature_view": ["raw17", "engineered", "core8", "behaviour"],
    "imputer":      ["median", "knn5", "iterative"],
    "winsorize":    ["none", "0.99", "0.995"],
    "transform":    ["none", "log1p", "yeojohnson", "quantile_normal"],
    "scaler":       ["standard", "robust", "minmax"],
    "reducer":      ["none", "pca_var80", "pca_var90", "pca_var95",
                     "pca_n2", "pca_n3", "pca_n4", "pca_n5", "pca_n6", "pca_n8", "pca_n10"],
    # ---- algorithm selection --------------------------------------------------
    "algorithm":    ["kmeans", "minibatch_kmeans", "bisecting_kmeans", "gmm",
                     "agglo_ward", "agglo_average", "agglo_complete", "birch",
                     "spectral", "hdbscan"],
    "k":            list(range(2, 13)),
    # ---- conditional algorithm hyperparameters --------------------------------
    "n_init":           [5, 10, 20],                       # centroid family
    "init":             ["k-means++", "random"],           # kmeans
    "bisecting":        ["biggest_inertia", "largest_cluster"],
    "covariance":       ["full", "tied", "diag", "spherical"],   # gmm
    "reg_covar":        [1e-6, 1e-4, 1e-2],                # gmm
    "metric":           ["euclidean", "manhattan", "cosine"],    # agglo (non-ward)
    "birch_threshold":  [0.2, 0.5, 1.0],
    "affinity":         ["nearest_neighbors", "rbf"],      # spectral
    "n_neighbors":      [10, 15, 30],
    "min_cluster_size": [50, 100, 150, 250, 400],          # hdbscan
    "min_samples":      [5, 10, 25],
    "selection":        ["eom", "leaf"],
}

# Ordinal dimensions get local ±1/±2 moves as well as arbitrary jumps: for these,
# neighbouring values really are near-neighbours in objective space.
ORDINAL = {"k", "n_init", "reg_covar", "birch_threshold", "n_neighbors",
           "min_cluster_size", "min_samples",
           }

# dimension -> predicate deciding whether it is active for a given configuration
CONDITIONS = {
    "k":                lambda c: c["algorithm"] != "hdbscan",
    "n_init":           lambda c: c["algorithm"] in {"kmeans", "minibatch_kmeans", "gmm"},
    "init":             lambda c: c["algorithm"] == "kmeans",
    "bisecting":        lambda c: c["algorithm"] == "bisecting_kmeans",
    "covariance":       lambda c: c["algorithm"] == "gmm",
    "reg_covar":        lambda c: c["algorithm"] == "gmm",
    "metric":           lambda c: c["algorithm"] in {"agglo_average", "agglo_complete"},
    "birch_threshold":  lambda c: c["algorithm"] == "birch",
    "affinity":         lambda c: c["algorithm"] == "spectral",
    "n_neighbors":      lambda c: c["algorithm"] == "spectral" and c.get("affinity") == "nearest_neighbors",
    "min_cluster_size": lambda c: c["algorithm"] == "hdbscan",
    "min_samples":      lambda c: c["algorithm"] == "hdbscan",
    "selection":        lambda c: c["algorithm"] == "hdbscan",
}

PREPROCESSING_DIMS = ["feature_view", "imputer", "winsorize", "transform", "scaler", "reducer"]
MODEL_DIMS = [d for d in SPACE if d not in PREPROCESSING_DIMS]


def active_dims(cfg: dict) -> list[str]:
    return [d for d in SPACE if d not in CONDITIONS or CONDITIONS[d](cfg)]


def canonical(cfg: dict) -> dict:
    """Drop inactive dimensions so equivalent pipelines hash identically."""
    return {d: cfg[d] for d in active_dims(cfg) if d in cfg}


def config_key(cfg: dict) -> str:
    blob = json.dumps(canonical(cfg), sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


def random_config(rng: np.random.Generator, k_range: tuple[int, int] = (2, 12)) -> dict:
    cfg = {d: _pick(rng, vals) for d, vals in SPACE.items()}
    cfg["k"] = int(rng.integers(k_range[0], k_range[1] + 1))
    return cfg


def _pick(rng: np.random.Generator, vals: list):
    return vals[int(rng.integers(len(vals)))]


def neighbours(cfg: dict, rng: np.random.Generator, n: int,
               k_range: tuple[int, int] = (2, 12),
               dim_weights: dict[str, float] | None = None) -> list[tuple[dict, str]]:
    """Sample up to ``n`` distinct one-dimension mutations of ``cfg``.

    A move changes exactly one *active* dimension, which keeps the neighbourhood
    small and interpretable and makes the trajectory in the dashboard readable as
    a sequence of single decisions. Ordinal dimensions get a step move half the
    time (±1 index) and a jump move otherwise, so the climber can both refine and
    escape.

    ``dim_weights`` lets the optimiser bias sampling toward dimensions that have
    historically produced improvements - a cheap adaptive-neighbourhood heuristic
    in the spirit of variable-neighbourhood search (Mladenovic & Hansen 1997).
    """
    dims = [d for d in active_dims(cfg) if len(SPACE[d]) > 1]
    if dim_weights:
        w = np.array([max(dim_weights.get(d, 1.0), 0.05) for d in dims], dtype=float)
        w /= w.sum()
    else:
        w = None

    seen, out = {config_key(cfg)}, []
    for _ in range(n * 12):
        if len(out) >= n:
            break
        d = str(rng.choice(dims, p=w))
        vals = SPACE[d]
        cur = cfg.get(d)
        if d in ORDINAL and cur in vals and rng.random() < 0.5:
            i = vals.index(cur)
            step = int(rng.choice([-2, -1, 1, 2]))
            j = min(max(i + step, 0), len(vals) - 1)
            if j == i:
                continue
            new_val, kind = vals[j], "step"
        else:
            alt = [v for v in vals if v != cur]
            if not alt:
                continue
            new_val, kind = alt[int(rng.integers(len(alt)))], "jump"
        cand = dict(cfg)
        cand[d] = new_val
        if d == "k":
            cand["k"] = int(min(max(int(new_val), k_range[0]), k_range[1]))
        key = config_key(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append((cand, f"{d}:{kind}"))
    return out


def space_size() -> dict:
    """Cardinality of the conditional space - the honest denominator for
    'how much of the space did the search actually look at?'."""
    total = 0
    pre = math.prod(len(SPACE[d]) for d in PREPROCESSING_DIMS)
    for algo in SPACE["algorithm"]:
        probe = {"algorithm": algo, "affinity": "nearest_neighbors"}
        n = 1
        for d in MODEL_DIMS:
            if d == "algorithm":
                continue
            if d in CONDITIONS and not CONDITIONS[d](probe):
                continue
            n *= len(SPACE[d])
        total += n
    return {"preprocessing_combinations": pre,
            "model_combinations": total,
            "total_configurations": pre * total,
            "n_dimensions": len(SPACE),
            "n_conditional_dimensions": len(CONDITIONS)}


def describe(cfg: dict) -> str:
    """One-line human-readable rendering of a pipeline, used in the leaderboard."""
    c = canonical(cfg)
    pre = f"{c['feature_view']} | {c['imputer']} | wins={c['winsorize']} | {c['transform']} | {c['scaler']} | {c['reducer']}"
    tail = " ".join(f"{d}={c[d]}" for d in MODEL_DIMS if d in c and d != "algorithm")
    return f"{pre}  ->  {c['algorithm']} {tail}".strip()
