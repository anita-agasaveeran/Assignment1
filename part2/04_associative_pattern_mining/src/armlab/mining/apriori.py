"""Apriori (Agrawal & Srikant, VLDB 1994) and Eclat (Zaki, IEEE TKDE 2000).

These exist for two reasons: they are the historical baselines the report
compares against, and they are an independent oracle -- `tests/` asserts all
three algorithms return byte-identical itemset->count maps, which is the only
honest way to trust a from-scratch FP-Growth.
"""
from __future__ import annotations

import time
from collections import defaultdict
from itertools import combinations
from typing import Iterable

from armlab.mining.fpgrowth import MiningStats


def apriori(transactions: Iterable[Iterable[int]], min_count: int,
            max_len: int = 4) -> tuple[dict[frozenset[int], int], MiningStats]:
    t0 = time.perf_counter()
    txs = [frozenset(t) for t in transactions]
    stats = MiningStats(algorithm="apriori", n_transactions=len(txs),
                        min_count=min_count)

    counts: dict[int, int] = defaultdict(int)
    for t in txs:
        for it in t:
            counts[it] += 1
    stats.candidates_generated += len(counts)
    stats.passes = 1

    results: dict[frozenset[int], int] = {}
    level = {frozenset([i]): c for i, c in counts.items() if c >= min_count}
    results.update(level)

    k = 2
    while level and k <= max_len:
        prev = sorted(level, key=lambda s: sorted(s))
        candidates = _join(prev, k, set(level))
        stats.candidates_generated += len(candidates)
        stats.passes += 1
        if not candidates:
            break
        cnt: dict[frozenset[int], int] = defaultdict(int)
        for t in txs:
            for c in candidates:
                if c <= t:
                    cnt[c] += 1
        level = {c: n for c, n in cnt.items() if n >= min_count}
        results.update(level)
        k += 1

    stats.itemsets_found = len(results)
    stats.seconds = time.perf_counter() - t0
    for s in results:
        stats.per_level[len(s)] = stats.per_level.get(len(s), 0) + 1
    return results, stats


def _join(prev: list[frozenset[int]], k: int,
          prev_set: set[frozenset[int]]) -> list[frozenset[int]]:
    """F_{k-1} x F_{k-1} join with the downward-closure prune."""
    out: list[frozenset[int]] = []
    sorted_prev = [sorted(s) for s in prev]
    for i in range(len(sorted_prev)):
        a = sorted_prev[i]
        for j in range(i + 1, len(sorted_prev)):
            b = sorted_prev[j]
            if a[:-1] != b[:-1]:
                break
            cand = frozenset(a) | frozenset(b)
            if len(cand) != k:
                continue
            # Apriori prune: every (k-1)-subset must be frequent.
            if all(frozenset(s) in prev_set for s in combinations(sorted(cand), k - 1)):
                out.append(cand)
    return out


def eclat(transactions: Iterable[Iterable[int]], min_count: int,
          max_len: int = 4) -> tuple[dict[frozenset[int], int], MiningStats]:
    """Vertical tid-set intersection with depth-first search."""
    t0 = time.perf_counter()
    txs = [list(t) for t in transactions]
    stats = MiningStats(algorithm="eclat", n_transactions=len(txs),
                        min_count=min_count)

    tid: dict[int, set[int]] = defaultdict(set)
    for i, t in enumerate(txs):
        for it in t:
            tid[it].add(i)

    results: dict[frozenset[int], int] = {}
    frequent = sorted(((i, s) for i, s in tid.items() if len(s) >= min_count),
                      key=lambda kv: (-len(kv[1]), kv[0]))
    _eclat_dfs([], frequent, min_count, max_len, results, stats)

    stats.itemsets_found = len(results)
    stats.seconds = time.perf_counter() - t0
    for s in results:
        stats.per_level[len(s)] = stats.per_level.get(len(s), 0) + 1
    return results, stats


def _eclat_dfs(prefix: list[int], items: list[tuple[int, set[int]]], min_count: int,
               max_len: int, results: dict[frozenset[int], int],
               stats: MiningStats) -> None:
    for idx, (item, tids) in enumerate(items):
        iset = prefix + [item]
        stats.candidates_generated += 1
        results[frozenset(iset)] = len(tids)
        if len(iset) >= max_len:
            continue
        suffix = []
        for other, otids in items[idx + 1:]:
            inter = tids & otids
            if len(inter) >= min_count:
                suffix.append((other, inter))
        if suffix:
            stats.max_depth = max(stats.max_depth, len(iset) + 1)
            _eclat_dfs(iset, suffix, min_count, max_len, results, stats)


ALGORITHMS = {"apriori": apriori, "eclat": eclat}
