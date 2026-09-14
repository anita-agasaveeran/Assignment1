"""Statistical machinery: exact tests, multiplicity control, randomisation."""
import numpy as np
import pytest
from scipy.stats import fisher_exact

from armlab.metrics.interestingness import Cont
from armlab.stats import significance as sig


def test_fisher_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n11 = int(rng.integers(1, 50)); n10 = int(rng.integers(1, 50))
        n01 = int(rng.integers(1, 50)); n00 = int(rng.integers(1, 200))
        c = Cont.of([n11], [n11 + n10], [n11 + n01], [n11 + n10 + n01 + n00])
        ours = sig.fisher_exact_right(c)[0]
        ref = fisher_exact([[n11, n10], [n01, n00]], alternative="greater")[1]
        assert ours == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_bh_controls_more_than_no_correction():
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.2, 0.9])
    rej_bh, q = sig.benjamini_hochberg(p, 0.05)
    assert rej_bh.sum() <= (p <= 0.05).sum()
    assert np.all(q >= p - 1e-12)          # q-values never below raw p
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)   # monotone after step-up


def test_bh_with_larger_hypothesis_space_is_stricter():
    p = np.array([1e-6, 1e-4, 1e-3, 0.01])
    few = sig.benjamini_hochberg(p, 0.05)[0].sum()
    many = sig.benjamini_hochberg(p, 0.05, n_tests=10_000_000)[0].sum()
    assert many <= few


def test_bonferroni():
    p = np.array([0.001, 0.02])
    rej, adj = sig.bonferroni(p, 0.05, n_tests=100)
    assert adj[0] == pytest.approx(0.1)
    assert not rej.any()


def test_search_space_size_grows_with_items():
    a = sig.search_space_size(50, 3, 2)
    b = sig.search_space_size(100, 3, 2)
    assert b > a > 0


def test_swap_randomisation_preserves_margins():
    rng = np.random.default_rng(3)
    tx = [sorted(rng.choice(30, size=int(rng.integers(2, 8)), replace=False).tolist())
          for _ in range(200)]
    out = sig.swap_randomisation_null(tx, 30, seed=1, n_swaps_factor=5)

    assert [len(t) for t in out] == [len(t) for t in tx]     # basket sizes fixed
    def freq(db):
        f = np.zeros(30, dtype=int)
        for t in db:
            for i in t:
                f[i] += 1
        return f
    assert np.array_equal(freq(out), freq(tx))               # item frequencies fixed
    assert all(len(set(t)) == len(t) for t in out)           # still sets
    assert out != tx                                          # something actually moved


def test_productive_filter_rejects_redundant_specialisation():
    """{a,b} => c must be rejected when a => c already explains it."""
    counts = {
        frozenset([0]): 500, frozenset([1]): 500, frozenset([2]): 400,
        frozenset([0, 2]): 400,          # a => c is already perfect
        frozenset([0, 1]): 250, frozenset([0, 1, 2]): 200,
    }
    out = sig.productive([frozenset([0, 1])], [frozenset([2])], counts, 1000)
    assert not out[0]
