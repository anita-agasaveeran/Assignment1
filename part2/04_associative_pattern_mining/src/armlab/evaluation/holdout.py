"""CRISP-DM Phase 5 -- Evaluation.

Association rules have no accuracy score, which is exactly why so many
market-basket projects ship rules that do not replicate.  We treat the rule set
as a *predictive artefact* and score it on transactions the miner never saw:

  * replication  -- does the rule still clear its thresholds out of sample?
  * stability    -- how far do confidence and lift drift, and does the *ranking*
                    survive (Spearman rho on lift)?
  * coverage     -- what share of holdout baskets does the rule set even fire on?
                    A perfect but inapplicable rule set is worthless.
  * precision    -- when a rule fires, how often is the consequent really there?
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from armlab.config import Config
from armlab.data.prepare import BasketDataset
from armlab.metrics.interestingness import Cont, compute_all
from armlab.stats import significance as sig


def incidence_matrix(ds: BasketDataset, n_items: int) -> np.ndarray:
    """Dense boolean transactions x items matrix.

    ~6k x 1.8k booleans is 11 MB -- cheap, and it turns every itemset count into
    a vectorised column-AND instead of a Python loop over transactions.
    """
    mat = np.zeros((ds.n_transactions, n_items), dtype=bool)
    for r, t in enumerate(ds.transactions):
        if t:
            mat[r, t] = True
    return mat


def count_itemsets(mat: np.ndarray, itemsets: list[frozenset[int]]) -> np.ndarray:
    out = np.empty(len(itemsets), dtype=np.int64)
    for i, s in enumerate(itemsets):
        cols = list(s)
        out[i] = int(mat[:, cols].all(axis=1).sum()) if cols else mat.shape[0]
    return out


def evaluate(rules: pd.DataFrame, holdout: BasketDataset, cfg: Config,
             n_items: int, mat: np.ndarray | None = None) -> dict[str, Any]:
    if rules is None or len(rules) == 0:
        return _empty_report()

    if mat is None:
        mat = incidence_matrix(holdout, n_items)
    N = holdout.n_transactions
    ant = list(rules["antecedent"])
    con = list(rules["consequent"])
    uni = [a | b for a, b in zip(ant, con)]

    ca = count_itemsets(mat, ant)
    cb = count_itemsets(mat, con)
    cab = count_itemsets(mat, uni)

    c = Cont.of(cab, ca, cb, np.full(len(ca), N))
    hm = compute_all(c)
    p_hold = sig.fisher_exact_right(c)
    rej, q = sig.correct(p_hold, cfg.stats.correction, cfg.stats.alpha, len(p_hold))

    enough = cab >= cfg.stats.min_holdout_count
    m = cfg.mining
    replicated = (enough
                  & (hm["confidence"] >= m.min_confidence)
                  & (hm["lift"] >= m.min_lift))
    validated = replicated & rej

    conf_tr = rules["confidence"].to_numpy()
    lift_tr = rules["lift"].to_numpy()
    conf_ho = hm["confidence"]
    lift_ho = hm["lift"]

    finite = np.isfinite(lift_tr) & np.isfinite(lift_ho)
    if finite.sum() > 2 and np.ptp(lift_tr[finite]) > 0 and np.ptp(lift_ho[finite]) > 0:
        rho = float(spearmanr(lift_tr[finite], lift_ho[finite]).statistic)
    else:
        rho = 0.0

    # Coverage & firing precision over holdout baskets.
    fired = np.zeros(N, dtype=bool)
    hits = 0
    fires = 0
    for a, b in zip(ant, con):
        a_mask = mat[:, list(a)].all(axis=1)
        fired |= a_mask
        nf = int(a_mask.sum())
        if nf:
            fires += nf
            hits += int((a_mask & mat[:, list(b)].all(axis=1)).sum())

    return {
        "n_rules": int(len(rules)),
        "n_holdout_transactions": int(N),
        "n_with_min_count": int(enough.sum()),
        "n_replicated": int(replicated.sum()),
        "replication_rate": float(replicated.mean()),
        "n_validated": int(validated.sum()),
        "validated_rate": float(validated.mean()),
        "coverage": float(fired.mean()),
        "firing_precision": float(hits / fires) if fires else 0.0,
        "lift_rank_spearman": rho,
        "confidence_drift_mean": float(np.mean(conf_ho - conf_tr)),
        "confidence_drift_mae": float(np.mean(np.abs(conf_ho - conf_tr))),
        "lift_drift_mean": float(np.mean(np.clip(lift_ho, 0, 1e6) - np.clip(lift_tr, 0, 1e6))),
        "median_holdout_p": float(np.median(p_hold)),
        "per_rule": {
            "holdout_support": hm["support"].tolist(),
            "holdout_confidence": conf_ho.tolist(),
            "holdout_lift": np.clip(lift_ho, 0, 1e6).tolist(),
            "holdout_count": cab.tolist(),
            "holdout_p": p_hold.tolist(),
            "holdout_q": q.tolist(),
            "replicated": replicated.tolist(),
            "validated": validated.tolist(),
        },
    }


def _empty_report() -> dict[str, Any]:
    return {"n_rules": 0, "n_holdout_transactions": 0, "n_with_min_count": 0,
            "n_replicated": 0, "replication_rate": 0.0, "n_validated": 0,
            "validated_rate": 0.0, "coverage": 0.0, "firing_precision": 0.0,
            "lift_rank_spearman": 0.0, "confidence_drift_mean": 0.0,
            "confidence_drift_mae": 0.0, "lift_drift_mean": 0.0,
            "median_holdout_p": 1.0, "per_rule": {}}


def attach(rules: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    per = report.get("per_rule") or {}
    if not per or len(rules) == 0:
        return rules
    out = rules.copy()
    for k, v in per.items():
        out[k] = v
    return out
