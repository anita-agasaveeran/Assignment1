"""Interestingness measures, checked against hand-computed values."""
import numpy as np
import pytest

from armlab.metrics.interestingness import MEASURES, Cont, compute_all


@pytest.fixture
def c():
    # N=1000, n11=100, n1_=200, n_1=300  ->  P(A)=.2 P(B)=.3 P(AB)=.1
    return Cont.of([100.0], [200.0], [300.0], [1000.0])


def test_core_measures(c):
    r = compute_all(c)
    assert r["support"][0] == pytest.approx(0.10)
    assert r["confidence"][0] == pytest.approx(0.50)
    assert r["lift"][0] == pytest.approx(0.1 / (0.2 * 0.3))
    assert r["leverage"][0] == pytest.approx(0.1 - 0.06)
    assert r["conviction"][0] == pytest.approx((1 - 0.3) / (1 - 0.5))
    assert r["cosine"][0] == pytest.approx(0.1 / np.sqrt(0.06))
    assert r["jaccard"][0] == pytest.approx(0.1 / (0.2 + 0.3 - 0.1))
    assert r["all_confidence"][0] == pytest.approx(0.1 / 0.3)
    assert r["added_value"][0] == pytest.approx(0.5 - 0.3)
    assert r["laplace"][0] == pytest.approx(101 / 202)
    assert r["odds_ratio"][0] == pytest.approx((100 * 600) / (100 * 200))


def test_chi_square_equals_phi_squared_times_n(c):
    r = compute_all(c)
    assert r["chi_square"][0] == pytest.approx(r["phi"][0] ** 2 * 1000, rel=1e-9)


def test_independence_gives_neutral_values():
    # n11 = n1_ * n_1 / N exactly
    c = Cont.of([60.0], [200.0], [300.0], [1000.0])
    r = compute_all(c)
    assert r["lift"][0] == pytest.approx(1.0)
    assert r["leverage"][0] == pytest.approx(0.0, abs=1e-12)
    assert r["phi"][0] == pytest.approx(0.0, abs=1e-12)
    assert r["added_value"][0] == pytest.approx(0.0, abs=1e-12)
    assert r["chi_square"][0] == pytest.approx(0.0, abs=1e-9)
    assert r["zhang"][0] == pytest.approx(0.0, abs=1e-12)


def test_yule_relations():
    c = Cont.of([100.0], [200.0], [300.0], [1000.0])
    r = compute_all(c)
    a = r["odds_ratio"][0]
    assert r["yule_q"][0] == pytest.approx((a - 1) / (a + 1))
    assert r["yule_y"][0] == pytest.approx((np.sqrt(a) - 1) / (np.sqrt(a) + 1))


def test_all_measures_finite_on_edge_cases():
    """Zero cells must not produce nan/inf leaking into the rule table."""
    c = Cont.of([0.0, 50.0, 10.0], [10.0, 50.0, 10.0], [10.0, 50.0, 1000.0],
                [1000.0, 50.0, 1000.0])
    r = compute_all(c)
    for k, v in r.items():
        assert np.all(np.isfinite(v)), f"{k} produced a non-finite value"


def test_glossary_covers_every_measure():
    from armlab.metrics.interestingness import glossary
    g = {x["key"] for x in glossary()}
    assert g == set(MEASURES)
    for m in MEASURES.values():
        assert m.citation and m.formula and m.range
