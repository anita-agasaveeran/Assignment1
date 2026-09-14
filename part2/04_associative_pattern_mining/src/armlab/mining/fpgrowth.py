"""FP-Growth (Han, Pei & Yin, SIGMOD 2000) -- implemented from scratch.

Why not `mlxtend`?  Because the AutoResearch loop evaluates ~60 configurations
and needs (a) exact support *counts* rather than float supports, (b) itemset
lattice reuse, and (c) instrumentation (nodes built, recursion depth) that the
library does not expose.  The dashboard charts that instrumentation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


class FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item: int | None, parent: "FPNode | None"):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[int, FPNode] = {}
        self.link: FPNode | None = None


@dataclass
class MiningStats:
    """Instrumentation surfaced in the dashboard's algorithm panel."""
    algorithm: str = "fpgrowth"
    n_transactions: int = 0
    min_count: int = 0
    nodes_created: int = 0
    conditional_trees: int = 0
    max_depth: int = 0
    candidates_generated: int = 0
    itemsets_found: int = 0
    passes: int = 0
    seconds: float = 0.0
    per_level: dict[int, int] = field(default_factory=dict)


class FPTree:
    def __init__(self) -> None:
        self.root = FPNode(None, None)
        self.headers: dict[int, FPNode] = {}
        self.tails: dict[int, FPNode] = {}
        self.n_nodes = 0

    def insert(self, items: list[int], count: int) -> None:
        node = self.root
        for it in items:
            child = node.children.get(it)
            if child is None:
                child = FPNode(it, node)
                node.children[it] = child
                self.n_nodes += 1
                if it in self.headers:
                    self.tails[it].link = child
                    self.tails[it] = child
                else:
                    self.headers[it] = child
                    self.tails[it] = child
            child.count += count
            node = child

    def single_path(self) -> list[FPNode] | None:
        """If the tree is a single chain we can enumerate its power set directly
        (Han et al. 2000, Lemma 2) instead of recursing."""
        path, node = [], self.root
        while node.children:
            if len(node.children) > 1:
                return None
            node = next(iter(node.children.values()))
            path.append(node)
        return path


def _order(counts: dict[int, int], min_count: int) -> list[int]:
    """Descending support, item id as deterministic tie-break."""
    return sorted((i for i, c in counts.items() if c >= min_count),
                  key=lambda i: (-counts[i], i))


def fpgrowth(transactions: Iterable[Iterable[int]], min_count: int,
             max_len: int = 4) -> tuple[dict[frozenset[int], int], MiningStats]:
    """Return every itemset with support count >= min_count, up to max_len."""
    import time
    t0 = time.perf_counter()
    txs = [list(t) for t in transactions]
    stats = MiningStats(algorithm="fpgrowth", n_transactions=len(txs),
                        min_count=min_count)

    counts: dict[int, int] = defaultdict(int)
    for t in txs:
        for it in t:
            counts[it] += 1

    order = _order(counts, min_count)
    rank = {it: r for r, it in enumerate(order)}

    tree = FPTree()
    for t in txs:
        filtered = sorted((i for i in t if i in rank), key=rank.__getitem__)
        if filtered:
            tree.insert(filtered, 1)
    stats.nodes_created += tree.n_nodes

    results: dict[frozenset[int], int] = {}
    _grow(tree, frozenset(), min_count, max_len, results, stats, depth=1)

    stats.itemsets_found = len(results)
    stats.seconds = time.perf_counter() - t0
    for s in results:
        stats.per_level[len(s)] = stats.per_level.get(len(s), 0) + 1
    return results, stats


def _grow(tree: FPTree, suffix: frozenset[int], min_count: int, max_len: int,
          results: dict[frozenset[int], int], stats: MiningStats, depth: int) -> None:
    stats.max_depth = max(stats.max_depth, depth)
    if len(suffix) >= max_len:
        return

    path = tree.single_path()
    if path is not None and len(path) > 0:
        # Single-prefix-path optimisation: enumerate combinations directly.
        from itertools import combinations
        budget = max_len - len(suffix)
        for r in range(1, min(len(path), budget) + 1):
            for combo in combinations(path, r):
                stats.candidates_generated += 1
                iset = suffix | {n.item for n in combo}
                cnt = min(n.count for n in combo)
                if cnt >= min_count:
                    results[frozenset(iset)] = max(results.get(frozenset(iset), 0), cnt)
        return

    # Least-frequent item first, so the conditional bases stay small.
    for item in sorted(tree.headers, key=lambda i: _header_count(tree, i)):
        iset = suffix | {item}
        cnt = _header_count(tree, item)
        if cnt < min_count:
            continue
        stats.candidates_generated += 1
        results[frozenset(iset)] = cnt
        if len(iset) >= max_len:
            continue

        # Conditional pattern base.
        base: list[tuple[list[int], int]] = []
        node = tree.headers[item]
        while node is not None:
            prefix, p = [], node.parent
            while p is not None and p.item is not None:
                prefix.append(p.item)
                p = p.parent
            if prefix:
                base.append((prefix[::-1], node.count))
            node = node.link
        if not base:
            continue

        cond_counts: dict[int, int] = defaultdict(int)
        for prefix, c in base:
            for it in prefix:
                cond_counts[it] += c
        cond_order = _order(cond_counts, min_count)
        if not cond_order:
            continue
        cond_rank = {it: r for r, it in enumerate(cond_order)}

        cond_tree = FPTree()
        for prefix, c in base:
            filt = sorted((i for i in prefix if i in cond_rank), key=cond_rank.__getitem__)
            if filt:
                cond_tree.insert(filt, c)
        if not cond_tree.headers:
            continue
        stats.conditional_trees += 1
        stats.nodes_created += cond_tree.n_nodes
        _grow(cond_tree, frozenset(iset), min_count, max_len, results, stats, depth + 1)


def _header_count(tree: FPTree, item: int) -> int:
    total, node = 0, tree.headers.get(item)
    while node is not None:
        total += node.count
        node = node.link
    return total
