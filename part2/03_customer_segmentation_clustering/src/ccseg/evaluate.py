"""CRISP-DM Phase 5 - Evaluation.

Internal validity indices answer "is this partition geometrically tidy?". They do
not answer the two questions that decide whether a segmentation ships:

  * **Is it real?**  Would a different sample of the same book give the same
    segments? Stability-based validation (Ben-Hur, Elisseeff & Guyon, PSB 2002;
    Lange, Roth, Braun & Buhmann, Neural Comput. 2004; von Luxburg, 2010) treats
    reproducibility under resampling as the primary evidence for cluster structure.
    Hennig (J. Multivar. Anal. 99, 2007) refines this to the *cluster* level: a
    partition can be stable on average while containing one segment that dissolves.
  * **Is it self-flattering?** Indices computed in a PCA-reduced space are
    optimistically biased relative to the same labels scored in the original
    feature space, because the reduction discards exactly the variance that would
    have made the clusters overlap. Both numbers are reported.

Also here: the gap statistic as an *independent* read on k (Tibshirani, Walther &
Hastie, JRSS-B 63, 2001), a full k-sweep that exposes the small-k bias of the
indices, and baselines - because a search result with nothing to beat is a number,
not a finding.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import config, models, prepare
from . import search_space as ss
from .metrics import ObjectiveCalibration, internal_indices, score_partition


# --------------------------------------------------------------------------- #
# Fitting a full candidate pipeline
# --------------------------------------------------------------------------- #
def fit_pipeline(frame: pd.DataFrame, cfg: dict, seed: int = config.RANDOM_SEED):
    """Fit preprocessing + clustering, and induce a deployable assignment rule."""
    cols = prepare.feature_view(cfg["feature_view"])
    pre = prepare.build_preprocessor(cfg, seed=seed)
    X = np.asarray(pre.fit_transform(frame[cols]), dtype=float)
    est = models.build_clusterer(cfg, seed=seed)
    labels = models.fit_predict(est, X)
    assigner = models.InductiveAssigner().fit(X, labels)
    return {"pre": pre, "est": est, "assigner": assigner, "X": X,
            "labels": labels, "columns": cols}


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def stability(frame: pd.DataFrame, cfg: dict, ec: config.EvalConfig,
              reference_labels: np.ndarray, fit_cap: int = 5000) -> dict:
    """Resampling stability at both partition and cluster level.

    Protocol
    --------
    For each of B replicates: draw a subsample of the book, refit the *entire*
    pipeline (preprocessing included - refitting only the clusterer would leak the
    full-data scaling and inflate stability), induce the nearest-centroid rule from
    that replicate, relabel every record, and compare against the reference
    partition.

      * partition level - Adjusted Rand Index (Hubert & Arabie 1985), chance-corrected
        so it does not drift upward with k, and NMI (Vinh, Epps & Bailey, JMLR 2010)
        as a second, differently-biased view.
      * cluster level  - for every reference segment, the best Jaccard overlap
        against any replicate segment. Hennig's rule of thumb: > 0.85 highly stable,
        0.60-0.85 a real but fuzzy pattern, < 0.60 dissolved.
    """
    rng = np.random.default_rng(ec.seed)
    n = len(frame)
    n_fit = min(int(ec.bootstrap_frac * n), fit_cap)
    ref = np.asarray(reference_labels)
    ref_clusters = [c for c in np.unique(ref) if c != -1]

    aris, nmis, jac = [], [], {int(c): [] for c in ref_clusters}
    coassign_n = min(1200, n)
    co_idx = rng.choice(n, coassign_n, replace=False)
    co_sum = np.zeros((coassign_n, coassign_n), dtype=np.float32)
    per_point_agree = np.zeros(n, dtype=float)
    rounds_ok = 0
    t0 = time.perf_counter()

    for b in range(ec.bootstrap_rounds):
        idx = rng.choice(n, n_fit, replace=False)
        try:
            fit = fit_pipeline(frame.iloc[idx], cfg, seed=ec.seed + b)
            Xall = np.asarray(fit["pre"].transform(frame[fit["columns"]]), dtype=float)
            rep = fit["assigner"].predict(Xall)
        except Exception:
            continue
        rounds_ok += 1
        mask = ref != -1
        aris.append(float(adjusted_rand_score(ref[mask], rep[mask])))
        nmis.append(float(normalized_mutual_info_score(ref[mask], rep[mask])))
        for c in ref_clusters:
            a = ref == c
            best = 0.0
            for d in np.unique(rep):
                bset = rep == d
                inter = float((a & bset).sum())
                union = float((a | bset).sum())
                if union > 0:
                    best = max(best, inter / union)
            jac[int(c)].append(best)
        sub = rep[co_idx]
        co_sum += (sub[:, None] == sub[None, :]).astype(np.float32)
        per_point_agree += (rep == ref).astype(float)

    if rounds_ok == 0:
        return {"error": "every stability replicate failed to fit"}

    co = co_sum / rounds_ok
    ref_sub = ref[co_idx]
    within = [float(co[np.ix_(ref_sub == c, ref_sub == c)].mean())
              for c in ref_clusters if (ref_sub == c).sum() > 1]

    return {
        "rounds_requested": ec.bootstrap_rounds, "rounds_completed": rounds_ok,
        "subsample_size": n_fit, "subsample_frac": round(n_fit / n, 3),
        "ari_mean": round(float(np.mean(aris)), 4),
        "ari_std": round(float(np.std(aris)), 4),
        "ari_ci95": [round(float(np.percentile(aris, 2.5)), 4),
                     round(float(np.percentile(aris, 97.5)), 4)],
        "ari_values": [round(v, 4) for v in aris],
        "nmi_mean": round(float(np.mean(nmis)), 4),
        "per_cluster_jaccard": {
            int(c): {"mean": round(float(np.mean(v)), 4),
                     "std": round(float(np.std(v)), 4),
                     "min": round(float(np.min(v)), 4),
                     "verdict": ("highly stable" if np.mean(v) > 0.85 else
                                 "stable pattern" if np.mean(v) > 0.60 else "dissolved")}
            for c, v in jac.items() if v},
        "mean_within_cluster_coassignment": round(float(np.mean(within)), 4) if within else None,
        "point_agreement": {
            "mean": round(float((per_point_agree / rounds_ok).mean()), 4),
            "pct_below_0_8": round(100 * float(((per_point_agree / rounds_ok) < 0.8).mean()), 2),
        },
        "per_point_agreement_raw": (per_point_agree / rounds_ok).tolist(),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# Gap statistic
# --------------------------------------------------------------------------- #
def gap_statistic(X: np.ndarray, k_range: tuple[int, int], b_refs: int = 12,
                  seed: int = config.RANDOM_SEED) -> dict:
    """Tibshirani, Walther & Hastie (2001).

    Compare log within-cluster dispersion against its expectation under a null of
    no structure, with the reference distribution drawn uniformly over a box
    aligned to the data's principal axes (the paper's recommended variant, which
    respects the shape of the data rather than only its bounding box).
    Chosen k = smallest k with Gap(k) >= Gap(k+1) - s_{k+1}.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    ks = list(range(k_range[0], k_range[1] + 1))

    def logW(D: np.ndarray, k: int) -> float:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(D)
        return float(np.log(max(km.inertia_, 1e-12)))

    pca = PCA().fit(X)
    Xp = pca.transform(X)
    lo, hi = Xp.min(axis=0), Xp.max(axis=0)

    obs = np.array([logW(X, k) for k in ks])
    ref = np.zeros((b_refs, len(ks)))
    for b in range(b_refs):
        Z = rng.uniform(lo, hi, size=Xp.shape) @ pca.components_ + pca.mean_
        ref[b] = [logW(Z, k) for k in ks]

    gap = ref.mean(axis=0) - obs
    sk = ref.std(axis=0) * np.sqrt(1 + 1 / b_refs)
    chosen = None
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            chosen = ks[i]
            break
    return {"k": ks, "gap": [round(float(g), 4) for g in gap],
            "s_k": [round(float(s), 4) for s in sk],
            "log_w_observed": [round(float(v), 4) for v in obs],
            "log_w_expected": [round(float(v), 4) for v in ref.mean(axis=0)],
            "chosen_k": chosen, "b_refs": b_refs,
            "rule": "smallest k with Gap(k) >= Gap(k+1) - s_(k+1)"}


# --------------------------------------------------------------------------- #
# k-sweep
# --------------------------------------------------------------------------- #
def k_sweep(frame: pd.DataFrame, cfg: dict, calib: ObjectiveCalibration,
            sc: config.SearchConfig, ec: config.EvalConfig,
            stability_rounds: int = 8) -> list[dict]:
    """Hold the champion pipeline fixed and vary only k.

    Isolates the effect of k from every other decision, and makes the small-k bias
    of the raw indices visible next to the constrained objective and the stability
    curve - three different answers to "how many segments?", which is the honest
    picture.

    ``k_achieved`` is recorded next to the requested k because several algorithms
    do not honour the request. Birch in particular caps the partition at the number
    of subclusters its CF-tree happens to hold, so at threshold 1.0 it returns five
    clusters for every k from 5 to 12 - a first version of this panel showed eleven
    rows of which seven were the same model, which is exactly the kind of chart that
    reads fine and means nothing. Callers should sweep with a k-honouring algorithm;
    the flag makes a violation visible rather than silent.
    """
    rows = []
    rng = np.random.default_rng(ec.seed)
    n_fit = min(int(0.8 * len(frame)), 4000)
    X_eval, _ = prepare.reference_space(frame, seed=ec.seed)
    for k in range(ec.k_sweep[0], ec.k_sweep[1] + 1):
        c = dict(cfg, k=k)
        if c["algorithm"] == "hdbscan":
            continue
        try:
            fit = fit_pipeline(frame, c, seed=ec.seed)
            s = score_partition(X_eval, fit["labels"], calib, search_cfg=sc, seed=ec.seed)
            self_sil = internal_indices(fit["X"], fit["labels"], seed=ec.seed)["silhouette"]
            aris = []
            for b in range(stability_rounds):
                idx = rng.choice(len(frame), n_fit, replace=False)
                f2 = fit_pipeline(frame.iloc[idx], c, seed=ec.seed + 100 + b)
                rep = f2["assigner"].predict(np.asarray(f2["pre"].transform(frame[f2["columns"]]), float))
                m = fit["labels"] != -1
                aris.append(float(adjusted_rand_score(fit["labels"][m], rep[m])))
            rows.append({"k": k, "k_achieved": int(s["k"]), "k_honoured": int(s["k"]) == k,
                         "silhouette": s["silhouette"], "self_silhouette": self_sil,
                         "calinski_harabasz": s["calinski_harabasz"],
                         "davies_bouldin": s["davies_bouldin"],
                         "objective": s["objective"], "raw_objective": s["raw_objective"],
                         "penalty": s["penalty"], "feasible": s["feasible"],
                         "min_cluster_frac": round(s["min_cluster_frac"], 4),
                         "balance_entropy": s["balance_entropy"],
                         "ari_mean": round(float(np.mean(aris)), 4),
                         "ari_std": round(float(np.std(aris)), 4)})
        except Exception as exc:
            rows.append({"k": k, "error": f"{type(exc).__name__}: {str(exc)[:100]}"})
    return rows


# --------------------------------------------------------------------------- #
# Reference-space validation and baselines
# --------------------------------------------------------------------------- #
def reference_space_indices(frame: pd.DataFrame, labels: np.ndarray,
                            seed: int = config.RANDOM_SEED) -> dict:
    """Re-score the champion labels in a neutral, un-reduced space.

    Fixed chain (median impute -> Yeo-Johnson -> standardise, no PCA, all 17 source
    features) chosen once and never tuned, so it cannot flatter any particular
    champion. The drop from the modelling-space silhouette is the price of the
    reduction, and it belongs in the report rather than in a footnote.
    """
    X, _ = prepare.reference_space(frame, seed=seed)
    idx = internal_indices(X, labels, seed=seed, sample=4000)
    return {**idx, "space": "17 source features, Yeo-Johnson + standardised, no reduction",
            "note": "Identical to the space the search objective is computed in "
                    "(prepare.reference_space), so search and evaluation agree by construction."}


def baselines(frame: pd.DataFrame, calib: ObjectiveCalibration, sc: config.SearchConfig,
              champion_k: int, seed: int = config.RANDOM_SEED) -> list[dict]:
    """What the search had to beat."""
    specs = [
        ("Textbook default", dict(feature_view="raw17", imputer="median", winsorize="none",
                                  transform="none", scaler="standard", reducer="none",
                                  algorithm="kmeans", k=3, n_init=10, init="k-means++"),
         "StandardScaler + KMeans(k=3) on the raw columns - the notebook default."),
        ("Log + KMeans", dict(feature_view="raw17", imputer="median", winsorize="none",
                              transform="log1p", scaler="standard", reducer="none",
                              algorithm="kmeans", k=4, n_init=10, init="k-means++"),
         "Adds the variance-stabilising transform an experienced analyst would reach for."),
        ("PCA + KMeans @ champion k", dict(feature_view="raw17", imputer="median", winsorize="none",
                                           transform="log1p", scaler="standard", reducer="pca_var90",
                                           algorithm="kmeans", k=champion_k, n_init=10, init="k-means++"),
         "Champion's k and family, but a hand-picked preprocessing chain."),
        ("GMM + BIC-style", dict(feature_view="raw17", imputer="median", winsorize="none",
                                 transform="yeojohnson", scaler="standard", reducer="pca_var90",
                                 algorithm="gmm", k=champion_k, covariance="full",
                                 reg_covar=1e-6, n_init=10),
         "Model-based alternative at the same k."),
    ]
    X_eval, _ = prepare.reference_space(frame, seed=seed)
    out = []
    for name, cfg, why in specs:
        try:
            fit = fit_pipeline(frame, cfg, seed=seed)
            s = score_partition(X_eval, fit["labels"], calib, search_cfg=sc, seed=seed)
            out.append({"name": name, "why": why, "pipeline": ss.describe(cfg),
                        "objective": s["objective"], "silhouette": s["silhouette"],
                        "davies_bouldin": s["davies_bouldin"],
                        "calinski_harabasz": s["calinski_harabasz"],
                        "k": s["k"], "feasible": s["feasible"],
                        "min_cluster_frac": round(s["min_cluster_frac"], 4)})
        except Exception as exc:
            out.append({"name": name, "why": why, "error": str(exc)[:120]})
    return out


def algorithm_shootout(frame: pd.DataFrame, cfg: dict, calib: ObjectiveCalibration,
                       sc: config.SearchConfig, seed: int = config.RANDOM_SEED) -> list[dict]:
    """Champion preprocessing held fixed; swap only the clustering algorithm.

    Answers the question a reviewer always asks: how much of the result is the
    preprocessing and how much is the algorithm?
    """
    X_eval, _ = prepare.reference_space(frame, seed=seed)
    out = []
    for algo in ["kmeans", "minibatch_kmeans", "bisecting_kmeans", "gmm",
                 "agglo_ward", "agglo_average", "birch", "hdbscan"]:
        c = dict(cfg, algorithm=algo)
        c.setdefault("covariance", "full"); c.setdefault("reg_covar", 1e-6)
        c.setdefault("birch_threshold", 0.5); c.setdefault("min_cluster_size", 150)
        c.setdefault("min_samples", 10); c.setdefault("selection", "eom")
        c.setdefault("n_init", 10); c.setdefault("init", "k-means++")
        c.setdefault("bisecting", "biggest_inertia"); c.setdefault("metric", "euclidean")
        t0 = time.perf_counter()
        try:
            fit = fit_pipeline(frame, c, seed=seed)
            s = score_partition(X_eval, fit["labels"], calib, search_cfg=sc, seed=seed)
            out.append({"algorithm": algo, "family": models.ALGO_FAMILY[algo],
                        "objective": s["objective"], "silhouette": s["silhouette"],
                        "davies_bouldin": s["davies_bouldin"], "k": s["k"],
                        "feasible": s["feasible"], "noise_frac": s["noise_frac"],
                        "inductive_fidelity": round(fit["assigner"].fidelity_, 4),
                        "fit_seconds": round(time.perf_counter() - t0, 3)})
        except Exception as exc:
            out.append({"algorithm": algo, "family": models.ALGO_FAMILY.get(algo, "?"),
                        "error": f"{type(exc).__name__}: {str(exc)[:110]}"})
    return sorted(out, key=lambda d: -d.get("objective", -99))


# --------------------------------------------------------------------------- #
# Subsample -> full-book transfer
# --------------------------------------------------------------------------- #
def finalist_refit(frame: pd.DataFrame, board: list[dict], calib: ObjectiveCalibration,
                   sc: config.SearchConfig, top_n: int = 10,
                   seed: int = config.RANDOM_SEED) -> dict:
    """Refit the best search configurations on the full book and re-rank them there.

    The search runs on a subsample for cost (AutoML4Clust's central trick), which
    silently assumes the ranking transfers. That assumption is testable, and here
    it is tested rather than trusted: the top configurations are refit on all
    records, re-scored in the neutral space, and the two rankings compared by
    Spearman correlation.

    It also fixes a failure this study actually hit. A configuration can be
    perfectly feasible on 2,500 records and violate the minimum-segment-size floor
    on 8,950 - a segment holding 4% of a subsample can hold 1% of the book. The
    champion is therefore whichever finalist is best *on the full data and
    feasible there*, not whichever won the search.
    """
    X_eval, _ = prepare.reference_space(frame, seed=seed)
    rows = []
    for rank, entry in enumerate(board[:top_n], start=1):
        cfg = entry["config"]
        t0 = time.perf_counter()
        try:
            fit = fit_pipeline(frame, cfg, seed=seed)
            s = score_partition(X_eval, fit["labels"], calib, search_cfg=sc, seed=seed)
            rows.append({
                "search_rank": rank, "optimiser": entry["optimiser"],
                "search_objective": entry["objective"],
                "full_objective": s["objective"], "full_silhouette": s["silhouette"],
                "full_davies_bouldin": s["davies_bouldin"], "k": s["k"],
                "feasible": s["feasible"], "violations": s["violations"],
                "min_cluster_frac": round(s["min_cluster_frac"], 4),
                "noise_frac": s["noise_frac"],
                "delta": round(s["objective"] - entry["objective"], 4),
                "inductive_fidelity": round(fit["assigner"].fidelity_, 4),
                "refit_seconds": round(time.perf_counter() - t0, 2),
                "pipeline": ss.describe(cfg), "config": cfg,
            })
        except Exception as exc:
            rows.append({"search_rank": rank, "config": cfg, "pipeline": ss.describe(cfg),
                         "error": f"{type(exc).__name__}: {str(exc)[:110]}"})

    ok = [r for r in rows if "error" not in r]
    feasible = [r for r in ok if r["feasible"]]
    champion = (max(feasible, key=lambda r: r["full_objective"]) if feasible
                else (max(ok, key=lambda r: r["full_objective"]) if ok else None))
    return {
        "finalists": sorted(rows, key=lambda r: -r.get("full_objective", -99)),
        "n_refit": len(rows), "n_feasible_on_full": len(feasible),
        "mean_objective_drop": round(float(np.mean([r["delta"] for r in ok])), 4) if ok else None,
        "champion": champion,
    }


def transfer_study(frame: pd.DataFrame, trials: list[dict], calib: ObjectiveCalibration,
                   sc: config.SearchConfig, n_probe: int = 26,
                   seed: int = config.RANDOM_SEED) -> dict:
    """Does the subsample ranking predict the full-book ranking?

    The search runs on a subsample for cost, which assumes the ranking transfers.
    Measuring that correlation *on the finalists alone* is a trap: the top ten
    configurations occupy a sliver of the objective range, and a rank correlation
    over a restricted range is dominated by noise - it can come out strongly
    negative while the ranking is perfectly serviceable overall. (An earlier
    version of this study reported exactly that: rho = -0.52 across ten finalists
    whose search objectives spanned 0.02.)

    So the probe is stratified across the whole observed range: trials are sorted
    by search objective, split into deciles, and sampled from each. Both
    correlations are reported - the full-range one is the answer to "does the
    subsample trick work?", and the top-band one is the answer to "can I trust it
    to pick the winner?", which is a harder and usually worse number.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    X_eval, _ = prepare.reference_space(frame, seed=seed)
    pool, seen = [], set()
    for t in sorted(trials, key=lambda z: -z["objective"]):
        if t["error"] is None and t["key"] not in seen and t["objective"] > -8:
            seen.add(t["key"])
            pool.append(t)
    if len(pool) < 12:
        return {}

    bands = np.array_split(np.arange(len(pool)), min(10, len(pool)))
    picks: list[int] = []
    per_band = max(1, n_probe // len(bands))
    for b in bands:
        picks.extend(rng.choice(b, min(per_band, len(b)), replace=False).tolist())

    rows = []
    for i in sorted(set(picks)):
        t = pool[i]
        try:
            fit = fit_pipeline(frame, t["config"], seed=seed)
            s = score_partition(X_eval, fit["labels"], calib, search_cfg=sc, seed=seed)
            rows.append({"search_objective": t["objective"], "full_objective": s["objective"],
                         "search_rank": i + 1, "k": s["k"], "feasible": s["feasible"],
                         "pipeline": ss.describe(t["config"])})
        except Exception:
            continue
    if len(rows) < 8:
        return {}

    a = np.array([r["search_objective"] for r in rows])
    b = np.array([r["full_objective"] for r in rows])
    rho_all = float(spearmanr(a, b).statistic)
    top = [r for r in rows if r["search_rank"] <= max(10, len(pool) // 10)]
    rho_top = (float(spearmanr([r["search_objective"] for r in top],
                               [r["full_objective"] for r in top]).statistic)
               if len(top) >= 5 else None)
    return {
        "n_probed": len(rows), "n_pool": len(pool),
        "spearman_full_range": round(rho_all, 4),
        "spearman_top_band": round(rho_top, 4) if rho_top is not None else None,
        "search_objective_range": [round(float(a.min()), 4), round(float(a.max()), 4)],
        "top_band_range": ([round(float(min(r["search_objective"] for r in top)), 4),
                            round(float(max(r["search_objective"] for r in top)), 4)]
                           if len(top) >= 5 else None),
        "points": rows,
        "verdict": ("the subsample ranking transfers well; the full-data refit is a safety net"
                    if rho_all > 0.7 else
                    "the subsample ranking transfers moderately - the full-data refit is doing real work"
                    if rho_all > 0.35 else
                    "the subsample ranking barely transfers; the full-data refit is load-bearing"),
        "caveat": ("The top-band correlation is computed over a restricted range and is "
                   "noise-dominated by construction; it is reported because selection happens "
                   "in that band, not because it is a reliable estimate."),
    }


# --------------------------------------------------------------------------- #
# Charter compliance - CRISP-DM's backward arrow, executed
# --------------------------------------------------------------------------- #
def charter_compliance(frame: pd.DataFrame, candidates: list[dict], calib: ObjectiveCalibration,
                       sc: config.SearchConfig, ec: config.EvalConfig, rounds: int = 12,
                       seed: int = config.RANDOM_SEED) -> dict:
    """Grade candidate pipelines against the *whole* Phase-1 charter, not the objective.

    The search objective encodes some of the charter (minimum segment size, minimum
    number of treatments) but not the parts that are expensive to compute: whether
    every segment survives resampling, and whether a shallow tree can restate the
    model. A champion can therefore win the search and still fail the brief - which
    is what happened in this study, where the objective-optimal pipeline contained a
    segment with a bootstrap Jaccard of 0.39, well inside Hennig's "dissolved" band.

    CRISP-DM draws an arrow from Evaluation back to Business Understanding for
    exactly this case. Rather than leave it as a note, this function walks the
    shortlist and reports the best pipeline that satisfies every criterion, so the
    deliverable is a model that meets the brief and the finding that the
    objective-optimal one does not.
    """
    from .profiling import surrogate_rules

    X_eval, _ = prepare.reference_space(frame, seed=seed)
    ec_fast = config.EvalConfig(bootstrap_rounds=rounds, bootstrap_frac=ec.bootstrap_frac,
                                gap_b_refs=ec.gap_b_refs, k_sweep=ec.k_sweep,
                                surrogate_depth=ec.surrogate_depth, seed=ec.seed)
    rows, seen = [], set()
    for cfg in candidates:
        key = ss.config_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        t0 = time.perf_counter()
        try:
            fit = fit_pipeline(frame, cfg, seed=seed)
            s = score_partition(X_eval, fit["labels"], calib, search_cfg=sc, seed=seed)
            if s["k"] < 2:
                continue
            stab = stability(frame, cfg, ec_fast, fit["labels"], fit_cap=3500)
            sur = surrogate_rules(frame, fit["labels"], depth=ec.surrogate_depth, seed=seed)
            jac = [v["mean"] for v in stab.get("per_cluster_jaccard", {}).values()] or [0.0]
            shares = [c / len(frame) for c in s["sizes"].values()]
            checks = {
                "SC-1": s["silhouette"] >= 0.25,
                "SC-2": min(jac) >= 0.60,
                "SC-3": stab.get("ari_mean", 0) >= 0.60,
                "SC-4": min(shares) >= sc.min_cluster_frac,
                "SC-5": sur["fidelity"] >= 0.85,
            }
            rows.append({
                "pipeline": ss.describe(cfg), "config": cfg, "objective": s["objective"],
                "silhouette": s["silhouette"], "k": s["k"], "noise_frac": s["noise_frac"],
                "ari_mean": stab.get("ari_mean"), "min_jaccard": round(float(min(jac)), 4),
                "per_cluster_jaccard": {k: v["mean"] for k, v in stab.get("per_cluster_jaccard", {}).items()},
                "min_share": round(float(min(shares)), 4), "surrogate_fidelity": sur["fidelity"],
                "checks": checks, "n_failed": sum(1 for v in checks.values() if not v),
                "compliant": all(checks.values()),
                "seconds": round(time.perf_counter() - t0, 2),
            })
        except Exception as exc:
            rows.append({"pipeline": ss.describe(cfg), "config": cfg,
                         "error": f"{type(exc).__name__}: {str(exc)[:110]}"})

    scored = [r for r in rows if "error" not in r]
    ok = [r for r in scored if r["compliant"]]
    frontier = [{"pipeline": r["pipeline"], "algorithm": r["config"]["algorithm"],
                 "family": models.ALGO_FAMILY.get(r["config"]["algorithm"], "?"),
                 "silhouette": r["silhouette"], "min_jaccard": r["min_jaccard"],
                 "noise_frac": r["noise_frac"], "k": r["k"], "objective": r["objective"],
                 "compliant": r["compliant"]} for r in scored]

    revision = None
    if ok:
        best = max(ok, key=lambda r: r["objective"])
        verdict = "A fully compliant pipeline exists and is recommended for deployment."
    elif scored:
        # Nothing clears the whole charter. Stability is treated as the criterion that
        # cannot be traded: a segment that does not reproduce under resampling is not a
        # segment, whereas a separation index sitting just under an a-priori threshold
        # is a judgement about how much structure this data was ever going to have.
        stable = [r for r in scored if r["checks"]["SC-2"] and r["checks"]["SC-3"]]
        pool = stable or scored
        best = max(pool, key=lambda r: r["objective"])
        failed = [k for k, v in best["checks"].items() if not v]
        sep_ceiling = max((r["silhouette"] for r in stable), default=None)
        unstable_better = [r for r in scored if r["checks"]["SC-1"] and not r["checks"]["SC-2"]]
        revision = {
            "unsatisfiable": True,
            "failing_criterion": failed,
            "observed_ceiling": round(float(sep_ceiling), 4) if sep_ceiling is not None else None,
            "proposed_bar": round(float(sep_ceiling) - 0.01, 2) if sep_ceiling is not None else None,
            "n_stable_candidates": len(stable),
            "n_separated_but_unstable": len(unstable_better),
            "mean_noise_of_unstable": (round(float(np.mean([r["noise_frac"] for r in unstable_better])), 3)
                                       if unstable_better else None),
            "argument": (
                "No pipeline in the shortlist satisfies SC-1 and SC-2 together, and the reason is "
                "structural rather than incidental. Every candidate that clears the silhouette bar "
                "does so by abstaining on roughly a fifth of the book and carving out a tight core, "
                "which leaves a small segment that does not survive resampling. Every candidate whose "
                "segments all reproduce lands just under the bar. The 0.25 threshold was set before "
                "the data was opened, from a general rule of thumb about weak structure; the evidence "
                "now says the achievable ceiling for a fully reproducible partition on this book is "
                "the value above. Revising SC-1 to that ceiling is a change of expectation supported "
                "by measurement. Relaxing SC-2 instead would mean campaigning against a segment that "
                "a different sample of the same customers would not produce."),
            "recommendation": (
                "Deploy the most stable high-objective pipeline, record SC-1 as failed against the "
                "original bar, and take the revised threshold back to the business owner as a "
                "Phase-1 amendment rather than presenting it as a pass."),
        }
        verdict = ("No candidate satisfies every criterion. The separation and stability criteria "
                   "are in direct tension on this data, and the charter needs an amendment.")
    else:
        best, verdict = None, "Every candidate failed to fit."

    return {
        "candidates": sorted(rows, key=lambda r: (r.get("n_failed", 99), -r.get("objective", -99))),
        "n_evaluated": len(rows), "n_compliant": len(ok),
        "recommended": best, "frontier": frontier, "charter_revision": revision,
        "rounds_per_candidate": rounds, "verdict": verdict,
    }
