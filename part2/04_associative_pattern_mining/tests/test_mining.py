"""The three miners are exact algorithms, so they must agree exactly."""
import random

import pytest

from armlab.mining.apriori import apriori, eclat
from armlab.mining.fpgrowth import fpgrowth

# The textbook example from Han, Pei & Yin (SIGMOD 2000, Table 1), item letters
# mapped to ints: f=0 c=1 a=2 b=3 m=4 p=5 l=6 o=7 n=8 e=9 d=10 g=11 h=12 i=13 j=14 k=15
HAN = [
    [0, 2, 1, 10, 11, 13, 4, 5],
    [2, 3, 1, 0, 6, 4, 7],
    [3, 0, 12, 14, 7],
    [3, 1, 15, 8, 5],
    [0, 2, 1, 9, 6, 5, 4, 8],
]


def _rand_db(n=400, items=25, seed=0):
    rng = random.Random(seed)
    db = []
    for _ in range(n):
        k = rng.randint(2, 8)
        db.append(sorted(rng.sample(range(items), k)))
    return db


def _brute(db, min_count, max_len):
    """Reference implementation: enumerate every subset. Slow but obviously right."""
    from collections import defaultdict
    from itertools import combinations
    c = defaultdict(int)
    for t in db:
        for r in range(1, min(max_len, len(t)) + 1):
            for s in combinations(sorted(t), r):
                c[frozenset(s)] += 1
    return {k: v for k, v in c.items() if v >= min_count}


@pytest.mark.parametrize("min_count", [1, 2, 3])
def test_han_example_matches_brute_force(min_count):
    ref = _brute(HAN, min_count, 4)
    for fn in (fpgrowth, apriori, eclat):
        got, _ = fn(HAN, min_count, 4)
        assert got == ref, f"{fn.__name__} disagrees at min_count={min_count}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_algorithms_agree_on_random_databases(seed):
    db = _rand_db(seed=seed)
    a, _ = fpgrowth(db, 12, 3)
    b, _ = apriori(db, 12, 3)
    c, _ = eclat(db, 12, 3)
    assert a == b == c
    assert a == _brute(db, 12, 3)


def test_max_len_is_respected():
    db = _rand_db(seed=5)
    for fn in (fpgrowth, apriori, eclat):
        sets, _ = fn(db, 5, 2)
        assert max(len(s) for s in sets) <= 2


def test_downward_closure_holds():
    """Every subset of a frequent itemset must itself be frequent, with a
    support count at least as large."""
    from itertools import combinations
    db = _rand_db(seed=9)
    sets, _ = fpgrowth(db, 10, 3)
    for s, count in sets.items():
        for r in range(1, len(s)):
            for sub in combinations(sorted(s), r):
                fs = frozenset(sub)
                assert fs in sets
                assert sets[fs] >= count


def test_empty_and_degenerate_inputs():
    for fn in (fpgrowth, apriori, eclat):
        sets, stats = fn([], 1, 3)
        assert sets == {}
        sets, _ = fn([[1], [1], [1]], 5, 3)
        assert sets == {}
