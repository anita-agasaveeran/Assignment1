"""Rule generation: frequent itemsets -> scored, statistically tested rules."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from armlab.config import Config
from armlab.data.prepare import BasketDataset
from armlab.metrics.interestingness import Cont, MEASURES, compute_all
from armlab.mining.apriori import ALGORITHMS
from armlab.mining.fpgrowth import MiningStats, fpgrowth
from armlab.stats import significance as sig


@dataclass
class MiningResult:
    rules: pd.DataFrame
    itemsets: dict[frozenset[int], int]
    stats: MiningStats
    n_transactions: int
    n_candidate_rules: int = 0
    seconds_rules: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def mine_itemsets(ds: BasketDataset, cfg: Config) -> tuple[dict[frozenset[int], int], MiningStats]:
    min_count = max(1, int(np.ceil(cfg.mining.min_support * ds.n_transactions)))
    algo = cfg.mining.algorithm
    if algo == "fpgrowth":
        return fpgrowth(ds.transactions, min_count, cfg.mining.max_itemset_len)
    fn = ALGORITHMS[algo]
    return fn(ds.transactions, min_count, cfg.mining.max_itemset_len)


def generate_rules(ds: BasketDataset, cfg: Config,
                   itemsets: dict[frozenset[int], int] | None = None,
                   stats: MiningStats | None = None,
                   run_significance: bool = True) -> MiningResult:
    t_mine = time.perf_counter()
    if itemsets is None:
        itemsets, stats = mine_itemsets(ds, cfg)
    assert stats is not None
    t0 = time.perf_counter()

    N = ds.n_transactions
    m = cfg.mining
    ante: list[frozenset[int]] = []
    cons: list[frozenset[int]] = []
    n11: list[int] = []
    n1_: list[int] = []
    n_1: list[int] = []
    n_candidates = 0

    for z, cz in itemsets.items():
        k = len(z)
        if k < 2:
            continue
        zs = sorted(z)
        for a_len in range(1, min(m.max_antecedent_len, k - 1) + 1):
            for a in combinations(zs, a_len):
                n_candidates += 1
                fa = frozenset(a)
                fb = z - fa
                ca = itemsets.get(fa)
                cb = itemsets.get(fb)
                if not ca or not cb:
                    continue
                ante.append(fa); cons.append(fb)
                n11.append(cz); n1_.append(ca); n_1.append(cb)

    if not ante:
        empty = pd.DataFrame(columns=_columns())
        return MiningResult(empty, itemsets, stats, N, n_candidates,
                            time.perf_counter() - t0,
                            {"reason": "no rules cleared the itemset stage"})

    c = Cont.of(np.array(n11), np.array(n1_), np.array(n_1), np.full(len(n11), N))
    metrics = compute_all(c)

    keep = ((metrics["confidence"] >= m.min_confidence)
            & (metrics["lift"] >= m.min_lift)
            & (metrics["leverage"] >= m.min_leverage))
    idx = np.flatnonzero(keep)

    df = pd.DataFrame({k: v[idx] for k, v in metrics.items()})
    df["antecedent"] = [ante[i] for i in idx]
    df["consequent"] = [cons[i] for i in idx]
    df["antecedent_len"] = [len(ante[i]) for i in idx]
    df["consequent_len"] = [len(cons[i]) for i in idx]
    df["n11"] = np.asarray(n11)[idx]
    df["n1_"] = np.asarray(n1_)[idx]
    df["n_1"] = np.asarray(n_1)[idx]
    df["N"] = N

    diagnostics: dict[str, Any] = {
        "n_candidate_rules": n_candidates,
        "n_rules_after_thresholds": int(len(df)),
        "min_count": stats.min_count,
    }

    if run_significance and len(df):
        c2 = Cont.of(df["n11"].to_numpy(), df["n1_"].to_numpy(),
                     df["n_1"].to_numpy(), df["N"].to_numpy())
        p = sig.fisher_exact_right(c2)
        m_space = sig.search_space_size(ds.n_items, m.max_itemset_len,
                                        m.max_antecedent_len)
        rej_rep, q_rep = sig.correct(p, cfg.stats.correction, cfg.stats.alpha, None)
        rej_ss, q_ss = sig.correct(p, cfg.stats.correction, cfg.stats.alpha,
                                   max(m_space, len(p)))
        df["p_fisher"] = p
        df["q_value_reported"] = q_rep
        df["q_value_searchspace"] = q_ss
        df["significant"] = rej_rep
        df["significant_searchspace"] = rej_ss
        df["productive"] = sig.productive(list(df["antecedent"]), list(df["consequent"]),
                                          itemsets, N, cfg.stats.alpha)
        diagnostics["significance"] = sig.summarise(
            p, q_ss, rej_ss, cfg.stats.alpha, cfg.stats.correction, m_space)
        diagnostics["significance"]["n_productive"] = int(df["productive"].sum())
        diagnostics["significance"]["n_significant_and_productive"] = int(
            (df["productive"] & df["significant_searchspace"]).sum())
    elif len(df):
        for col, val in (("p_fisher", 1.0), ("q_value_reported", 1.0),
                         ("q_value_searchspace", 1.0)):
            df[col] = val
        df["significant"] = False
        df["significant_searchspace"] = False
        df["productive"] = True

    df = df.sort_values(["leverage", "lift"], ascending=False).reset_index(drop=True)
    if m.top_k and len(df) > m.top_k:
        df = df.head(m.top_k).copy()

    df["rule_id"] = [f"R{i:05d}" for i in range(len(df))]
    diagnostics["n_rules_returned"] = int(len(df))
    return MiningResult(df, itemsets, stats, N, n_candidates,
                        time.perf_counter() - t0, diagnostics)


def _columns() -> list[str]:
    return (list(MEASURES.keys()) + [
        "antecedent", "consequent", "antecedent_len", "consequent_len",
        "n11", "n1_", "n_1", "N", "p_fisher", "q_value_reported",
        "q_value_searchspace", "significant", "significant_searchspace",
        "productive", "rule_id"])


def humanise(df: pd.DataFrame, ds: BasketDataset, limit: int | None = None) -> list[dict]:
    """Rows for the dashboard: item ids replaced by product names."""
    out = []
    src = df if limit is None else df.head(limit)
    for r in src.itertuples(index=False):
        d = r._asdict()
        a = sorted(d.pop("antecedent")); b = sorted(d.pop("consequent"))
        rec = {
            "rule_id": d.get("rule_id"),
            "antecedent_items": [ds.label(i) for i in a],
            "consequent_items": [ds.label(i) for i in b],
            "antecedent_codes": [ds.item_codes.get(i, "") for i in a],
            "consequent_codes": [ds.item_codes.get(i, "") for i in b],
            "rule": f"{{{', '.join(ds.label(i) for i in a)}}} => "
                    f"{{{', '.join(ds.label(i) for i in b)}}}",
        }
        for k, v in d.items():
            if isinstance(v, (np.bool_, bool)):
                rec[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                fv = float(v)
                rec[k] = round(fv, 6) if np.isfinite(fv) else None
            else:
                rec[k] = v
        out.append(rec)
    return out
