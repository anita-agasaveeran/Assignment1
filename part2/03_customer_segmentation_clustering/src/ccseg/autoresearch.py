"""AutoResearch - hill-climbing CASH optimisation for clustering pipelines.

CRISP-DM Phase 4 (Modeling) is normally where a practitioner tries three
algorithms by hand and keeps the one with the best silhouette. This module
replaces that with a documented search procedure and a control arm, so the
Modeling phase produces *evidence* rather than a preference.

Method
------
Iterated local search over the conditional CASH space of ``search_space.py``:

  1. **Restart**  - draw a random configuration (GRASP-style diversification).
  2. **Local step** - sample ``neighbours_per_step`` one-dimension mutations.
       * ``stochastic``  accepts the first strict improvement found  (first-improvement)
       * ``steepest``    evaluates all sampled neighbours, takes the best (steepest-ascent)
  3. **Sideways moves** - a non-worsening move is accepted up to ``sideways_patience``
     times, which lets the climber traverse the large plateaus that categorical
     spaces produce (Selman, Levesque & Mitchell 1992, on plateau moves in GSAT).
  4. **Tabu memory** - the last ``tabu_size`` configuration hashes are refused, so a
     plateau walk cannot cycle (Glover 1986).
  5. **Adaptive neighbourhood** - dimensions that have produced improvements are
     sampled more often on later steps, in the spirit of variable-neighbourhood
     search (Mladenovic & Hansen 1997).
  6. Global best across restarts is returned.

Two cost controls, both taken from the AutoML-for-clustering literature:

  * **Subsample-then-refit.** AutoML4Clust (Tschechlov, Fritz & Schwarz, EDBT 2021)
    shows that optimising the CVI on a subsample and transferring the winning
    configuration preserves solution quality at a fraction of the cost. The search
    runs on ``search_subsample`` records; the champion is refit on the full book.
  * **Two-level memoisation.** The objective is cached on the canonical config hash,
    and the *preprocessed matrix* is cached separately on the preprocessing
    sub-config - most local moves change only a model dimension, so the expensive
    imputation/transform/PCA chain is reused.

The control arm is random search over the same space with the same evaluation
budget and the same evaluator. Reporting a search without a control tells you
nothing about whether the search worked.
"""
from __future__ import annotations

import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, models, prepare
from . import search_space as ss
from .metrics import ObjectiveCalibration, internal_indices, score_partition

FAILED = -9.99  # objective assigned to a configuration that could not be fitted


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
@dataclass
class Evaluator:
    """Turns a configuration into a score, with caching, timing and failure capture."""
    frame: pd.DataFrame
    calib: ObjectiveCalibration
    search_cfg: config.SearchConfig
    objective: str = "composite"
    seed: int = config.RANDOM_SEED

    cache: dict = field(default_factory=dict)
    pre_cache: dict = field(default_factory=dict)
    X_eval: np.ndarray | None = None      # the neutral scoring space, built once
    n_calls: int = 0
    n_cache_hits: int = 0
    n_pre_hits: int = 0
    n_failures: int = 0
    total_fit_s: float = 0.0

    def __post_init__(self):
        # Built once, from the same rows the search sees, and never touched again.
        self.X_eval, _ = prepare.reference_space(self.frame, seed=self.seed)

    def preprocess(self, cfg: dict) -> np.ndarray:
        pkey = tuple(cfg[d] for d in ss.PREPROCESSING_DIMS)
        if pkey in self.pre_cache:
            self.n_pre_hits += 1
            return self.pre_cache[pkey]
        cols = prepare.feature_view(cfg["feature_view"])
        pipe = prepare.build_preprocessor(cfg, seed=self.seed)
        X = np.asarray(pipe.fit_transform(self.frame[cols]), dtype=float)
        self.pre_cache[pkey] = X
        return X

    def __call__(self, cfg: dict) -> dict:
        key = ss.config_key(cfg)
        if key in self.cache:
            self.n_cache_hits += 1
            return {**self.cache[key], "cached": True}

        t0 = time.perf_counter()
        rec: dict
        try:
            X = self.preprocess(cfg)
            if cfg["algorithm"] in models.MAX_N and len(X) > models.MAX_N[cfg["algorithm"]]:
                raise RuntimeError(f"{cfg['algorithm']} guarded above n={models.MAX_N[cfg['algorithm']]}")
            est = models.build_clusterer(cfg, seed=self.seed)
            labels = models.fit_predict(est, X)
            # The objective is measured in the neutral space (see prepare.reference_space);
            # the candidate's own space is scored too, but only as a diagnostic.
            sc = score_partition(self.X_eval, labels, self.calib, objective=self.objective,
                                 search_cfg=self.search_cfg, seed=self.seed)
            self_idx = internal_indices(X, labels, seed=self.seed,
                                        sample=self.search_cfg.silhouette_sample)
            sc["self_silhouette"] = self_idx["silhouette"]
            sc["self_davies_bouldin"] = self_idx["davies_bouldin"]
            sc["optimism"] = (round(self_idx["silhouette"] - sc["silhouette"], 5)
                              if np.isfinite(self_idx["silhouette"]) and np.isfinite(sc["silhouette"])
                              else None)
            sc["model_space_dims"] = int(X.shape[1])
            rec = {"objective": sc["objective"], "metrics": sc, "error": None}
        except Exception as exc:  # a config that cannot be fitted is a data point, not a crash
            self.n_failures += 1
            rec = {"objective": FAILED, "metrics": {},
                   "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                   "traceback_head": traceback.format_exc().strip().splitlines()[-1][:200]}
        rec["elapsed_s"] = round(time.perf_counter() - t0, 4)
        rec["key"] = key
        self.total_fit_s += rec["elapsed_s"]
        self.n_calls += 1
        self.cache[key] = rec
        return {**rec, "cached": False}

    def stats(self) -> dict:
        tot = self.n_calls + self.n_cache_hits
        return {
            "unique_evaluations": self.n_calls,
            "cache_lookups": tot,
            "objective_cache_hits": self.n_cache_hits,
            "objective_cache_hit_rate": round(self.n_cache_hits / max(tot, 1), 4),
            "preprocessing_cache_hits": self.n_pre_hits,
            "preprocessing_cache_hit_rate": round(self.n_pre_hits / max(self.n_calls, 1), 4),
            "distinct_preprocessing_chains": len(self.pre_cache),
            "failures": self.n_failures,
            "failure_rate": round(self.n_failures / max(self.n_calls, 1), 4),
            "total_fit_seconds": round(self.total_fit_s, 2),
            "mean_seconds_per_eval": round(self.total_fit_s / max(self.n_calls, 1), 4),
        }


# --------------------------------------------------------------------------- #
# Calibration draw
# --------------------------------------------------------------------------- #
def calibrate(frame: pd.DataFrame, sc: config.SearchConfig, n: int = 18) -> tuple[ObjectiveCalibration, list[dict]]:
    """Random probe used only to freeze the CH scale constant before the search.

    These evaluations are charged against nothing - they exist so that the
    objective is stationary. Their raw indices are still reported, because they
    double as a naive 'what does a random pipeline look like?' reference.
    """
    rng = np.random.default_rng(sc.seed - 1)
    probe = Evaluator(frame, ObjectiveCalibration(500.0), sc, objective="composite", seed=sc.seed)
    chs, recs = [], []
    for _ in range(n):
        cfg = ss.random_config(rng, sc.k_range)
        r = probe(cfg)
        if r["error"] is None and np.isfinite(r["metrics"].get("calinski_harabasz", np.nan)):
            chs.append(r["metrics"]["calinski_harabasz"])
        recs.append({"config": ss.canonical(cfg), "objective": r["objective"], "error": r["error"]})
    return ObjectiveCalibration.from_samples(chs), recs


# --------------------------------------------------------------------------- #
# Hill climbing
# --------------------------------------------------------------------------- #
def hill_climb(evaluator: Evaluator, sc: config.SearchConfig) -> dict:
    rng = np.random.default_rng(sc.seed)
    trials: list[dict] = []
    tabu: deque[str] = deque(maxlen=sc.tabu_size)
    dim_credit: dict[str, float] = {d: 1.0 for d in ss.SPACE}
    restart_traces: list[dict] = []

    best_cfg, best_obj = None, -np.inf
    t_start = time.perf_counter()

    def log(cfg, res, *, move, restart, accepted, kind):
        nonlocal best_cfg, best_obj
        if res["objective"] > best_obj:
            best_obj, best_cfg = res["objective"], dict(cfg)
        trials.append({
            "i": len(trials),
            "t": round(time.perf_counter() - t_start, 3),
            "optimiser": "hill_climb",
            "restart": restart,
            "move": move,
            "step_kind": kind,          # restart | probe | accept_improve | accept_sideways | reject
            "accepted": accepted,
            "config": ss.canonical(cfg),
            "key": res["key"],
            "objective": res["objective"],
            "incumbent": round(float(current_obj_holder[0]), 5),
            "best_so_far": round(float(best_obj), 5),
            "elapsed_s": res["elapsed_s"],
            "cached": res.get("cached", False),
            "error": res["error"],
            "metrics": {k: res["metrics"].get(k) for k in
                        ("silhouette", "calinski_harabasz", "davies_bouldin", "k",
                         "min_cluster_frac", "noise_frac", "balance_entropy",
                         "penalty", "feasible", "self_silhouette", "optimism",
                         "model_space_dims")} if res["metrics"] else {},
        })

    current_obj_holder = [-np.inf]  # mutable so `log` can read the incumbent

    for restart in range(sc.restarts):
        if _out_of_budget(evaluator, sc, t_start):
            break
        cur = ss.random_config(rng, sc.k_range)
        current_obj_holder[0] = -np.inf
        res = evaluator(cur)
        current_obj_holder[0] = res["objective"]
        cur_obj = res["objective"]
        tabu.append(res["key"])
        log(cur, res, move="restart", restart=restart, accepted=True, kind="restart")

        steps, sideways_used, improved_steps = 0, 0, 0
        trace = [round(float(cur_obj), 5)]

        while not _out_of_budget(evaluator, sc, t_start):
            cands = ss.neighbours(cur, rng, sc.neighbours_per_step, sc.k_range, dim_credit)
            cands = [(c, m) for c, m in cands if ss.config_key(c) not in tabu]
            if not cands:
                break

            moved = False
            scored: list[tuple[float, dict, str, dict]] = []
            for cand, move in cands:
                if _out_of_budget(evaluator, sc, t_start):
                    break
                r = evaluator(cand)
                dim = move.split(":")[0]
                if sc.strategy == "stochastic":
                    if r["objective"] > cur_obj + 1e-9:            # first improvement -> take it
                        log(cand, r, move=move, restart=restart, accepted=True, kind="accept_improve")
                        dim_credit[dim] = dim_credit.get(dim, 1.0) + 1.0
                        cur, cur_obj = cand, r["objective"]
                        current_obj_holder[0] = cur_obj
                        tabu.append(r["key"])
                        moved, improved_steps = True, improved_steps + 1
                        break
                    log(cand, r, move=move, restart=restart, accepted=False, kind="reject")
                    dim_credit[dim] = max(dim_credit.get(dim, 1.0) * 0.98, 0.2)
                    scored.append((r["objective"], cand, move, r))
                else:                                              # steepest ascent
                    log(cand, r, move=move, restart=restart, accepted=False, kind="probe")
                    scored.append((r["objective"], cand, move, r))

            if not moved and scored:
                obj, cand, move, r = max(scored, key=lambda z: z[0])
                if obj > cur_obj + 1e-9:                           # steepest-ascent acceptance
                    log(cand, r, move=move, restart=restart, accepted=True, kind="accept_improve")
                    dim_credit[move.split(":")[0]] = dim_credit.get(move.split(":")[0], 1.0) + 1.0
                    cur, cur_obj = cand, obj
                    current_obj_holder[0] = cur_obj
                    tabu.append(r["key"])
                    moved, improved_steps = True, improved_steps + 1
                elif abs(obj - cur_obj) <= 1e-3 and sideways_used < sc.sideways_patience and obj > FAILED:
                    log(cand, r, move=move, restart=restart, accepted=True, kind="accept_sideways")
                    cur, cur_obj = cand, obj                       # plateau traversal
                    current_obj_holder[0] = cur_obj
                    tabu.append(r["key"])
                    moved, sideways_used = True, sideways_used + 1

            steps += 1
            trace.append(round(float(cur_obj), 5))
            if not moved:
                break                                              # local optimum reached

        restart_traces.append({
            "restart": restart, "steps": steps, "improving_steps": improved_steps,
            "sideways_moves": sideways_used, "start": trace[0], "end": trace[-1],
            "gain": round(trace[-1] - trace[0], 5), "trace": trace,
            "terminated": "local_optimum" if steps and not _out_of_budget(evaluator, sc, t_start) else "budget",
        })

    return {
        "optimiser": "hill_climb",
        "strategy": sc.strategy,
        "best_config": best_cfg,
        "best_objective": round(float(best_obj), 5),
        "trials": trials,
        "restarts": restart_traces,
        "dim_credit": {k: round(v, 2) for k, v in sorted(dim_credit.items(), key=lambda z: -z[1])},
        "wall_seconds": round(time.perf_counter() - t_start, 2),
    }


def random_search(evaluator: Evaluator, sc: config.SearchConfig, budget: int) -> dict:
    """Control arm: uniform random sampling over the identical space and budget."""
    rng = np.random.default_rng(sc.seed + 777)
    trials, best_cfg, best_obj = [], None, -np.inf
    t0 = time.perf_counter()
    for i in range(budget):
        if time.perf_counter() - t0 > sc.time_budget_s:
            break
        cfg = ss.random_config(rng, sc.k_range)
        r = evaluator(cfg)
        if r["objective"] > best_obj:
            best_obj, best_cfg = r["objective"], dict(cfg)
        trials.append({
            "i": i, "t": round(time.perf_counter() - t0, 3), "optimiser": "random_search",
            "restart": -1, "move": "sample", "step_kind": "sample", "accepted": False,
            "config": ss.canonical(cfg), "key": r["key"], "objective": r["objective"],
            "incumbent": round(float(best_obj), 5), "best_so_far": round(float(best_obj), 5),
            "elapsed_s": r["elapsed_s"], "cached": r.get("cached", False), "error": r["error"],
            "metrics": {k: r["metrics"].get(k) for k in
                        ("silhouette", "calinski_harabasz", "davies_bouldin", "k",
                         "min_cluster_frac", "noise_frac", "balance_entropy",
                         "penalty", "feasible", "self_silhouette", "optimism",
                         "model_space_dims")} if r["metrics"] else {},
        })
    return {"optimiser": "random_search", "best_config": best_cfg,
            "best_objective": round(float(best_obj), 5), "trials": trials,
            "wall_seconds": round(time.perf_counter() - t0, 2)}


def _out_of_budget(ev: Evaluator, sc: config.SearchConfig, t0: float) -> bool:
    return ev.n_calls >= sc.max_evals or (time.perf_counter() - t0) > sc.time_budget_s


# --------------------------------------------------------------------------- #
# Post-hoc analysis of the search itself
# --------------------------------------------------------------------------- #
def anytime_curve(trials: list[dict], budget: int) -> list[float]:
    """Best-objective-so-far as a function of evaluation count.

    This is the anytime performance curve used throughout the AutoML literature to
    compare optimisers; comparing only final scores hides an optimiser that needed
    ten times the budget to get there.
    """
    best, out = -np.inf, []
    for t in trials:
        best = max(best, t["objective"])
        out.append(round(float(best), 5))
    while len(out) < budget:
        out.append(out[-1] if out else 0.0)
    return out[:budget]


def dimension_importance(trials: list[dict]) -> list[dict]:
    """Marginal importance of each search dimension.

    For every dimension, group the observed objectives by the value that dimension
    took and compute the between-group share of variance (an eta-squared). This is
    the cheap, model-free cousin of the fANOVA analysis of Hutter, Hoos &
    Leyton-Brown (ICML 2014).

    Caveat, stated because it matters: hill-climbing samples the space
    non-uniformly, so these are *observational* effects over the trajectory, not
    causal effects over the whole space. The random-search arm is included in the
    pool precisely to dilute that bias.
    """
    rows = [t for t in trials if t["error"] is None and t["objective"] > FAILED]
    if len(rows) < 12:
        return []
    y = np.array([t["objective"] for t in rows], dtype=float)
    total_var = float(y.var())
    out = []
    for dim in ss.SPACE:
        groups: dict[str, list[float]] = {}
        for t, v in zip(rows, y):
            if dim in t["config"]:
                groups.setdefault(str(t["config"][dim]), []).append(float(v))
        groups = {g: vs for g, vs in groups.items() if len(vs) >= 3}
        if len(groups) < 2 or total_var <= 0:
            continue
        n = sum(len(v) for v in groups.values())
        grand = float(np.mean([x for v in groups.values() for x in v]))
        between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups.values()) / n
        best_val = max(groups.items(), key=lambda z: np.mean(z[1]))
        worst_val = min(groups.items(), key=lambda z: np.mean(z[1]))
        out.append({
            "dimension": dim,
            "eta_squared": round(float(between / total_var), 4),
            "n_observed": n,
            "n_values": len(groups),
            "best_value": best_val[0], "best_mean": round(float(np.mean(best_val[1])), 4),
            "worst_value": worst_val[0], "worst_mean": round(float(np.mean(worst_val[1])), 4),
            "per_value": {g: {"mean": round(float(np.mean(v)), 4), "n": len(v)}
                          for g, v in sorted(groups.items(), key=lambda z: -np.mean(z[1]))},
        })
    return sorted(out, key=lambda d: -d["eta_squared"])


def leaderboard(trials: list[dict], top: int = 15) -> list[dict]:
    seen, rows = set(), []
    for t in sorted(trials, key=lambda z: -z["objective"]):
        if t["error"] is not None or t["key"] in seen:
            continue
        seen.add(t["key"])
        rows.append({
            "rank": len(rows) + 1, "objective": t["objective"],
            "optimiser": t["optimiser"], "config": t["config"],
            "pipeline": ss.describe(t["config"]), "metrics": t["metrics"],
            "elapsed_s": t["elapsed_s"],
        })
        if len(rows) >= top:
            break
    return rows


def optimism_analysis(trials: list[dict]) -> dict:
    """How much would scoring each candidate in its own space have inflated it?

    ``optimism`` is (silhouette in the candidate's own modelling space) minus
    (silhouette of the same labels in the neutral space). It is the size of the
    prize the optimiser would have been chasing had the two spaces not been
    separated, and it grows sharply as the reduction gets more aggressive - which
    is exactly why an unguarded search collapses onto two principal components.
    """
    rows = [t for t in trials if t.get("metrics", {}).get("optimism") is not None]
    if not rows:
        return {}
    by_dim: dict[int, list[float]] = {}
    for t in rows:
        d = t["metrics"].get("model_space_dims")
        if d:
            by_dim.setdefault(int(d), []).append(float(t["metrics"]["optimism"]))
    opt = np.array([t["metrics"]["optimism"] for t in rows], dtype=float)
    worst = max(rows, key=lambda t: t["metrics"]["optimism"])
    return {
        "n_trials": len(rows),
        "mean_optimism": round(float(opt.mean()), 4),
        "max_optimism": round(float(opt.max()), 4),
        "pct_inflated": round(100 * float((opt > 0.05).mean()), 2),
        "by_model_space_dims": {d: {"mean_optimism": round(float(np.mean(v)), 4), "n": len(v)}
                                for d, v in sorted(by_dim.items())},
        "worst_case": {"pipeline": ss.describe(worst["config"]),
                       "self_silhouette": worst["metrics"].get("self_silhouette"),
                       "neutral_silhouette": worst["metrics"].get("silhouette"),
                       "optimism": worst["metrics"].get("optimism")},
    }
