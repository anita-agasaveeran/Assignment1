"""Cluster validity indices and the composite search objective.

Index selection follows Arbelaitz et al. (2013), an empirical comparison of 30
relative validity indices over 720 synthetic and 20 real datasets, in which
Silhouette, Davies-Bouldin* and Calinski-Harabasz are among the consistently
strongest performers. AutoML4Clust (Tschechlov, Fritz & Schwarz, EDBT 2021)
optimises exactly these three, one at a time, as the CASH objective.

Optimising any single index is known to be biased - Milligan & Cooper (1985) and
Vendramin, Campello & Hruschka (2010) both document systematic preference for
small k and for spherical, equal-sized clusters. This module therefore also
provides a *calibrated composite*: the three indices are mapped onto a common
[0,1] scale with a stationary transform, averaged, and then penalised for
solutions that are unusable in the business sense (a segment too small to run a
campaign for, or a density model that calls most of the book noise).

References
----------
Rousseeuw (1987)   J. Comput. Appl. Math. 20:53-65      - silhouette
Calinski & Harabasz (1974) Commun. Stat. 3(1):1-27      - variance ratio criterion
Davies & Bouldin (1979) IEEE TPAMI 1(2):224-227         - within/between ratio
Hubert & Arabie (1985) J. Classification 2:193-218      - adjusted Rand index
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)

from . import config


# --------------------------------------------------------------------------- #
# Raw indices
# --------------------------------------------------------------------------- #
def cluster_shape(labels: np.ndarray) -> dict:
    """Structural description of a partition, independent of any index."""
    lab = np.asarray(labels)
    noise = int((lab == -1).sum())
    core = lab[lab != -1]
    uniq, counts = np.unique(core, return_counts=True)
    n = len(lab)
    frac = counts / max(len(core), 1)
    # Normalised Shannon entropy of the size distribution: 1.0 = perfectly balanced.
    ent = float(-(frac * np.log(frac + 1e-12)).sum() / np.log(len(uniq))) if len(uniq) > 1 else 0.0
    return {
        "k": int(len(uniq)),
        "sizes": {int(u): int(c) for u, c in zip(uniq, counts)},
        "min_cluster_frac": float(counts.min() / n) if len(counts) else 0.0,
        "max_cluster_frac": float(counts.max() / n) if len(counts) else 1.0,
        "balance_entropy": round(ent, 4),
        "noise_frac": round(noise / n, 4),
    }


def internal_indices(X: np.ndarray, labels: np.ndarray, seed: int = config.RANDOM_SEED,
                     sample: int = 2500) -> dict:
    """Silhouette / CH / DB, computed on core points only (noise excluded).

    Silhouette is O(n^2); it is estimated on a seeded subsample above ``sample``
    points, which is the standard practice and keeps a single evaluation in the
    tens of milliseconds rather than seconds.
    """
    lab = np.asarray(labels)
    mask = lab != -1
    Xc, lc = X[mask], lab[mask]
    if len(np.unique(lc)) < 2 or len(lc) < 10:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan"), "n_scored": int(len(lc))}
    try:
        sil = float(silhouette_score(Xc, lc, metric="euclidean",
                                     sample_size=min(sample, len(lc)), random_state=seed))
        ch = float(calinski_harabasz_score(Xc, lc))
        db = float(davies_bouldin_score(Xc, lc))
    except Exception:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan"), "n_scored": int(len(lc))}
    return {"silhouette": round(sil, 5), "calinski_harabasz": round(ch, 2),
            "davies_bouldin": round(db, 5), "n_scored": int(len(lc))}


def silhouette_profile(X: np.ndarray, labels: np.ndarray, seed: int = config.RANDOM_SEED,
                       sample: int = 4000) -> dict:
    """Per-cluster silhouette distribution - Rousseeuw's original diagnostic plot.

    A high mean can hide a cluster whose members mostly sit on the wrong side of
    the boundary, so the dashboard shows the per-cluster breakdown, not just the mean.
    """
    from sklearn.metrics import silhouette_samples
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    mask = lab != -1
    Xc, lc = X[mask], lab[mask]
    if len(Xc) > sample:
        idx = rng.choice(len(Xc), sample, replace=False)
        Xc, lc = Xc[idx], lc[idx]
    if len(np.unique(lc)) < 2:
        return {"per_cluster": [], "overall": float("nan")}
    s = silhouette_samples(Xc, lc)
    out = []
    for c in np.unique(lc):
        v = np.sort(s[lc == c])[::-1]
        out.append({
            "cluster": int(c), "n": int(len(v)),
            "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4),
            "pct_negative": round(100 * float((v < 0).mean()), 2),
            # thinned silhouette curve for plotting
            "curve": [round(float(x), 3) for x in v[:: max(1, len(v) // 60)]],
        })
    return {"per_cluster": out, "overall": round(float(s.mean()), 4)}


# --------------------------------------------------------------------------- #
# Composite objective
# --------------------------------------------------------------------------- #
class ObjectiveCalibration:
    """Stationary [0,1] rescaling for the three indices.

    A hill climber compares scores across iterations, so the objective must not move
    under it. Min-max normalising against the running history would do exactly that,
    so the CH scale constant ``kappa`` is estimated once, from a random calibration
    draw of configurations, and then frozen for the whole study.

        silhouette in [-1,1]  ->  (s+1)/2
        davies-bouldin >= 0   ->  1/(1+db)          (lower is better -> inverted)
        calinski-harabasz >=0 ->  ch/(ch+kappa)     (unbounded -> saturating)
    """

    def __init__(self, kappa: float = 500.0):
        self.kappa = float(kappa)

    @classmethod
    def from_samples(cls, ch_values: list[float]) -> "ObjectiveCalibration":
        vals = [v for v in ch_values if np.isfinite(v) and v > 0]
        kappa = float(np.median(vals)) if vals else 500.0
        return cls(kappa=max(kappa, 1.0))

    def normalise(self, m: dict) -> dict:
        sil, ch, db = m["silhouette"], m["calinski_harabasz"], m["davies_bouldin"]
        if not all(np.isfinite([sil, ch, db])):
            return {"n_sil": 0.0, "n_ch": 0.0, "n_db": 0.0}
        return {"n_sil": (sil + 1) / 2, "n_ch": ch / (ch + self.kappa), "n_db": 1 / (1 + db)}

    def to_dict(self) -> dict:
        return {"kappa_ch": round(self.kappa, 2),
                "sil_map": "(s+1)/2", "ch_map": "ch/(ch+kappa)", "db_map": "1/(1+db)"}


# Weights: silhouette carries the most weight because it is the only one of the
# three that is bounded, per-sample decomposable, and directly interpretable to a
# stakeholder ("how well does this customer fit its segment?").
COMPOSITE_WEIGHTS = {"n_sil": 0.50, "n_db": 0.30, "n_ch": 0.20}

# Additive shift applied to any configuration that violates a business constraint.
# Equal to the maximum attainable raw objective (1.0), which makes the ordering
# lexicographic: feasibility first, index quality second.
INFEASIBLE_SHIFT = 1.0


def score_partition(X: np.ndarray, labels: np.ndarray, calib: ObjectiveCalibration,
                    objective: str = "composite", search_cfg: config.SearchConfig | None = None,
                    seed: int = config.RANDOM_SEED) -> dict:
    """Full scoring of one candidate partition: indices, shape, penalties, objective."""
    sc = search_cfg or config.SearchConfig()
    shape = cluster_shape(labels)
    idx = internal_indices(X, labels, seed=seed, sample=sc.silhouette_sample)

    if shape["k"] < 2 or not np.isfinite(idx["silhouette"]):
        return {**idx, **shape, "penalty": 1.0, "raw_objective": 0.0, "objective": -1.0,
                "feasible": False, "violations": ["degenerate partition (k<2 or unscorable)"]}

    norm = calib.normalise(idx)
    if objective == "silhouette":
        raw = norm["n_sil"]
    elif objective == "calinski_harabasz":
        raw = norm["n_ch"]
    elif objective == "davies_bouldin":
        raw = norm["n_db"]
    else:
        raw = sum(COMPOSITE_WEIGHTS[k] * norm[k] for k in COMPOSITE_WEIGHTS)

    # --- business feasibility constraints ------------------------------------ #
    # Constraint handling follows the *static penalty* formulation (Coello Coello,
    # Comput. Methods Appl. Mech. Eng. 191, 2002): every feasible solution strictly
    # dominates every infeasible one, while a graded penalty inside the infeasible
    # region still gives the climber a direction back toward feasibility.
    #
    # This matters here rather than being decoration. Left unconstrained, the
    # objective is maximised by a degenerate partition - one blob holding most of
    # the book plus a handful of far-flung outlier pockets - which scores a
    # silhouette of 1.0 and is worth nothing to a campaign manager.
    violations: list[str] = []
    penalty = 0.0
    if shape["min_cluster_frac"] < sc.min_cluster_frac:
        deficit = (sc.min_cluster_frac - shape["min_cluster_frac"]) / sc.min_cluster_frac
        penalty += 0.35 * deficit
        violations.append(f"smallest segment is {100*shape['min_cluster_frac']:.1f}% "
                          f"(< {100*sc.min_cluster_frac:.0f}% floor, SC-4)")
    if shape["max_cluster_frac"] > 0.60:
        penalty += 0.30 * (shape["max_cluster_frac"] - 0.60) / 0.40
        violations.append(f"one segment absorbs {100*shape['max_cluster_frac']:.0f}% of the book")
    if shape["noise_frac"] > sc.max_noise_frac:
        penalty += 0.5 * (shape["noise_frac"] - sc.max_noise_frac)
        violations.append(f"{100*shape['noise_frac']:.0f}% of the book labelled noise")
    if shape["k"] < sc.min_k_business:
        penalty += 0.08 * (sc.min_k_business - shape["k"])
        violations.append(f"k={shape['k']} is below the {sc.min_k_business}-treatment minimum the "
                          f"campaign programme requires")
    if shape["balance_entropy"] < 0.55 and shape["k"] > 2:
        penalty += 0.10 * (0.55 - shape["balance_entropy"])
        violations.append(f"size distribution is lopsided (entropy {shape['balance_entropy']:.2f})")

    feasible = len(violations) == 0
    # INFEASIBLE_SHIFT >= max attainable raw objective, so no infeasible solution can
    # ever outrank a feasible one however good its indices look.
    objective = raw - penalty - (0.0 if feasible else INFEASIBLE_SHIFT)
    return {**idx, **shape, **{k: round(v, 5) for k, v in norm.items()},
            "raw_objective": round(float(raw), 5),
            "penalty": round(float(penalty), 5),
            "objective": round(float(objective), 5),
            "feasible": feasible,
            "violations": violations}
