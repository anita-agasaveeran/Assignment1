"""Statistical validation of discovered rules.

Association rule mining is a massive multiple-comparisons exercise: mining at
2% support over 1,800 items evaluates millions of hypotheses, so at alpha=0.05
tens of thousands of *pure noise* rules clear the bar.  Reporting rules without
this section is the single most common flaw in market-basket write-ups.

We apply three independent guards, in increasing order of strictness:

1. Fisher's exact test on the 2x2 table (Fisher 1922).  Exact, not asymptotic --
   chi-square is unreliable for the small cell counts typical of retail baskets.
2. Multiplicity correction: Benjamini-Hochberg FDR (Benjamini & Hochberg, JRSS-B
   1995) or Bonferroni.  Webb (Mach. Learn. 68:1-33, 2007) argues the correction
   must count the *searched* hypothesis space, not the reported rules -- we
   expose both and default to the honest (search-space) variant.
3. Productivity (Webb 2007): a rule survives only if it beats every one of its
   own generalisations.  This kills the classic artefact where {a,b} => c is
   reported purely because a => c is strong.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import hypergeom

from armlab.metrics.interestingness import Cont


def fisher_exact_right(c: Cont) -> np.ndarray:
    """One-sided (right-tail) Fisher exact p-value, vectorised.

    H0: A and B independent given the margins.  Under H0 the count n11 follows
    Hypergeometric(N, n1., n.1); the p-value is P(X >= n11).
    """
    n11 = np.asarray(c.n11, dtype=np.int64)
    N = np.asarray(c.N, dtype=np.int64)
    n1_ = np.asarray(c.n1_, dtype=np.int64)
    n_1 = np.asarray(c.n_1, dtype=np.int64)
    p = hypergeom.sf(n11 - 1, N, n1_, n_1)
    return np.clip(np.nan_to_num(p, nan=1.0), 0.0, 1.0)


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05,
                       n_tests: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """BH step-up procedure. Returns (reject, q_values).

    `n_tests` lets us correct against the full searched space rather than the
    subset we chose to report (Webb 2007).
    """
    p = np.asarray(p, dtype=float)
    m = int(n_tests) if n_tests else p.size
    if p.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, p.size + 1)
    q_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q <= alpha, q


def bonferroni(p: np.ndarray, alpha: float = 0.05,
               n_tests: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    m = int(n_tests) if n_tests else max(p.size, 1)
    adj = np.clip(p * m, 0.0, 1.0)
    return adj <= alpha, adj


def correct(p: np.ndarray, method: str, alpha: float,
            n_tests: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if method == "bh":
        return benjamini_hochberg(p, alpha, n_tests)
    if method == "bonferroni":
        return bonferroni(p, alpha, n_tests)
    p = np.asarray(p, dtype=float)
    return p <= alpha, p


def productive(antecedents: list[frozenset[int]], consequents: list[frozenset[int]],
               counts: dict[frozenset[int], int], n_tx: int,
               alpha: float = 0.05) -> np.ndarray:
    """Webb's (2007) productivity filter.

    A => B is productive iff for every proper non-empty A' subset of A, the
    confidence of A => B is significantly greater than that of A' => B.  We test
    each generalisation with a one-sided Fisher exact test on the sub-population
    covered by A' (does restricting to A really raise the rate of B?).
    """
    out = np.ones(len(antecedents), dtype=bool)
    for i, (a, b) in enumerate(zip(antecedents, consequents)):
        if len(a) < 2:
            continue
        n_ab = counts.get(a | b)
        n_a = counts.get(a)
        if not n_ab or not n_a:
            out[i] = False
            continue
        for r in range(1, len(a)):
            for sub in combinations(sorted(a), r):
                sa = frozenset(sub)
                n_sa = counts.get(sa)
                n_sab = counts.get(sa | b)
                if not n_sa or not n_sab:
                    continue
                # Within the A'-covered population, is B enriched inside A?
                p = hypergeom.sf(n_ab - 1, n_sa, n_a, n_sab)
                if not np.isfinite(p) or p > alpha:
                    out[i] = False
                    break
            if not out[i]:
                break
    return out


def search_space_size(n_items: int, max_itemset_len: int,
                      max_antecedent_len: int) -> int:
    """Number of rules the search *could* have examined -- the honest m for
    multiplicity correction (Webb 2007, Sec. 4)."""
    from math import comb
    total = 0
    for k in range(2, max_itemset_len + 1):
        if k > n_items:
            break
        n_sets = comb(n_items, k)
        n_splits = sum(comb(k, a) for a in range(1, min(max_antecedent_len, k - 1) + 1))
        total += n_sets * n_splits
    return int(total)


def swap_randomisation_null(transactions: list[list[int]], n_items: int,
                            seed: int = 0, n_swaps_factor: int = 10) -> list[list[int]]:
    """Margin-preserving randomisation (Gionis et al., ACM TKDD 2007).

    Repeatedly swaps two items between two transactions, keeping both the
    transaction sizes and the item frequencies fixed.  Rules that survive at the
    same thresholds on this surrogate are pure structure-free artefacts, which
    gives an *empirical* false-discovery baseline to sit alongside the analytic
    FDR.
    """
    rng = np.random.default_rng(seed)
    tx = [set(t) for t in transactions]
    n = len(tx)
    if n < 2:
        return [sorted(t) for t in tx]
    target = n_swaps_factor * sum(len(t) for t in tx)
    done = 0
    attempts = 0
    max_attempts = target * 20
    while done < target and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        ti, tj = tx[i], tx[j]
        only_i = ti - tj
        only_j = tj - ti
        if not only_i or not only_j:
            continue
        a = list(only_i)[rng.integers(0, len(only_i))]
        b = list(only_j)[rng.integers(0, len(only_j))]
        ti.remove(a); ti.add(b)
        tj.remove(b); tj.add(a)
        done += 1
    swap_randomisation_null.last_swaps = done          # type: ignore[attr-defined]
    swap_randomisation_null.last_attempts = attempts   # type: ignore[attr-defined]
    return [sorted(t) for t in tx]


def summarise(p: np.ndarray, q: np.ndarray, reject: np.ndarray,
              alpha: float, method: str, n_tests: int) -> dict[str, Any]:
    p = np.asarray(p, dtype=float)
    return {
        "n_rules_tested": int(p.size),
        "n_hypotheses_corrected_for": int(n_tests),
        "correction": method,
        "alpha": alpha,
        "n_significant_uncorrected": int((p <= alpha).sum()),
        "n_significant_corrected": int(reject.sum()),
        "expected_false_positives_uncorrected": float(round(alpha * p.size, 2)),
        "median_p": float(np.median(p)) if p.size else 1.0,
        "min_p": float(p.min()) if p.size else 1.0,
    }
