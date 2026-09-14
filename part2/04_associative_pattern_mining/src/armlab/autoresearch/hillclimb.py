"""Stochastic hill climbing with adaptive steps, tabu memory and random restarts.

Choice of optimiser
-------------------
The mining objective is cheap-ish but *non-differentiable, noisy and riddled
with plateaus*: nudging min_support by 0.001 usually changes nothing at all,
then crosses a support threshold and changes everything.  Gradient methods are
inapplicable; Bayesian optimisation is overkill at ~60 evaluations on a 5-D
space and its surrogate would be badly fit by those plateaus.

Stochastic local search is the standard answer for exactly this shape of
landscape (Hoos & Stutzle, *Stochastic Local Search*, 2004).  Three well-known
augmentations turn plain hill climbing into something that actually escapes
local optima:

  * random restarts -- the classic fix for local optima (Selman, Levesque &
    Mitchell, AAAI 1992, on GSAT).
  * an adaptive step that contracts on stall, so the search anneals from coarse
    exploration to fine exploitation.
  * a tabu set over discretised coordinates (Glover, ORSA J. Computing 1989)
    so plateau moves are not re-evaluated.

Everything is logged, so the dashboard can show *why* a configuration won, not
merely that it did.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from armlab.autoresearch import space
from armlab.autoresearch.ledger import Ledger, Trial
from armlab.autoresearch.objective import Score
from armlab.config import AutoResearchConfig

Evaluator = Callable[[dict[str, Any]], tuple[Score, dict[str, Any]]]


@dataclass
class SearchResult:
    best_params: dict[str, Any]
    best_score: float
    best_trial: Trial
    ledger: Ledger
    n_evaluations: int
    n_restarts_used: int
    seconds: float
    improvement_over_seed: float
    seed_score: float
    convergence: list[dict[str, Any]]


def _perturb(u: np.ndarray, step: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian move on a random subset of coordinates, reflected at the bounds."""
    d = u.size
    mask = rng.random(d) < 0.6
    if not mask.any():
        mask[rng.integers(0, d)] = True
    v = u.copy()
    v[mask] = u[mask] + rng.normal(0.0, step, size=int(mask.sum()))
    # Reflect rather than clip: clipping piles probability mass on the bounds
    # and the search sticks to the edges of the space.
    v = np.where(v < 0, -v, v)
    v = np.where(v > 1, 2 - v, v)
    return np.clip(v, 0.0, 1.0)


def search(evaluate: Evaluator, cfg: AutoResearchConfig,
           seed_params: dict[str, Any],
           ledger_path: Path | str | None = None,
           progress: Callable[[str], None] | None = None) -> SearchResult:
    t_start = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    ledger = Ledger(ledger_path)
    tabu: dict[tuple, float] = {}
    budget = cfg.max_evaluations
    n_evals = 0
    convergence: list[dict[str, Any]] = []

    best_trial: Trial | None = None
    seed_score = 0.0

    def run(u: np.ndarray, kind: str, restart: int, iteration: int,
            step: float, parent: int | None) -> tuple[Trial, float]:
        nonlocal n_evals, best_trial
        params = space.decode(u)
        k = space.key(u)
        cache_hit = k in tabu
        t0 = time.perf_counter()
        sc, extra = evaluate(params)
        elapsed = time.perf_counter() - t0
        n_evals += 1
        tabu[k] = sc.value
        if len(tabu) > cfg.tabu_size * 8:
            for old in list(tabu)[: len(tabu) - cfg.tabu_size * 4]:
                tabu.pop(old, None)
        is_best = best_trial is None or sc.value > best_trial.score
        tr = ledger.record(Trial(
            index=n_evals - 1, restart=restart, iteration=iteration, kind=kind,
            params=params, coords=[float(x) for x in u], score=sc.value,
            terms=sc.terms, weighted=sc.weighted,
            diagnostics=sc.diagnostics | extra, accepted=False,
            is_best_so_far=is_best, step_size=step, seconds=elapsed,
            parent=parent, cache_hit=cache_hit))
        if is_best:
            best_trial = tr
        convergence.append({"evaluation": n_evals, "score": sc.value,
                            "best": best_trial.score, "restart": restart,
                            "step": step, "kind": kind})
        if progress:
            progress(f"[{n_evals:>3}/{budget}] r{restart} {kind:<9} "
                     f"score={sc.value:.4f} best={best_trial.score:.4f} "
                     f"sup={params['min_support']:.4f} "
                     f"conf={params['min_confidence']:.2f} "
                     f"lift={params['min_lift']:.2f} "
                     f"val={sc.diagnostics.get('n_validated', 0)}")
        return tr, sc.value

    restart = 0
    while n_evals < budget and restart <= cfg.restarts:
        if restart == 0:
            u = space.encode(seed_params)
            kind = "seed"
        else:
            u = rng.random(len(space.SPACE))
            kind = "restart"
        cur_trial, cur_score = run(u, kind, restart, 0, cfg.initial_step, None)
        cur_trial.accepted = True
        if restart == 0:
            seed_score = cur_score

        step = cfg.initial_step
        stall = 0
        iteration = 0
        while n_evals < budget:
            iteration += 1
            best_n_trial: Trial | None = None
            best_n_score = -np.inf
            best_n_u = None
            for _ in range(cfg.neighbours_per_step):
                if n_evals >= budget:
                    break
                v = _perturb(u, step, rng)
                if space.key(v) in tabu:
                    v = _perturb(u, max(step, cfg.initial_step), rng)
                tr, s = run(v, "neighbour", restart, iteration, step, cur_trial.index)
                if s > best_n_score:
                    best_n_trial, best_n_score, best_n_u = tr, s, v

            if best_n_trial is not None and best_n_score > cur_score + 1e-9:
                best_n_trial.accepted = True
                u, cur_trial, cur_score = best_n_u, best_n_trial, best_n_score
                stall = 0
                step = min(cfg.initial_step, step / cfg.step_decay)
            else:
                stall += 1
                step = max(cfg.min_step, step * cfg.step_decay)
                if stall >= cfg.patience:
                    break
        restart += 1

    assert best_trial is not None
    return SearchResult(
        best_params=best_trial.params, best_score=best_trial.score,
        best_trial=best_trial, ledger=ledger, n_evaluations=n_evals,
        n_restarts_used=restart - 1, seconds=time.perf_counter() - t_start,
        improvement_over_seed=best_trial.score - seed_score,
        seed_score=seed_score, convergence=convergence)
