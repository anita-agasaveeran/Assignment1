"""The derived property matrix must reproduce the published literature.

These assertions are the regression test on the numerical derivation: if a
measure's implementation drifts, its properties change and this fails.
"""
from armlab.metrics.properties import property_matrix


def _m(matrix):
    return {r["key"]: r for r in matrix["measures"]}


def test_only_odds_ratio_family_is_scaling_invariant():
    """Tan, Kumar & Srivastava (KDD 2002): O2 is satisfied only by the odds
    ratio and its monotone transforms, Yule's Q and Y."""
    m = _m(property_matrix())
    o2 = {k for k, r in m.items() if r["O2"]}
    assert o2 == {"odds_ratio", "yule_q", "yule_y"}


def test_null_invariant_family():
    """The null-invariant measures are exactly those that never read n00."""
    m = _m(property_matrix())
    o4 = {k for k, r in m.items() if r["O4"]}
    for k in ("cosine", "jaccard", "all_confidence", "max_confidence", "kulczynski"):
        assert k in o4, f"{k} should be null-invariant"
    for k in ("support", "lift", "leverage", "phi", "odds_ratio", "kappa"):
        assert k not in o4, f"{k} must NOT be null-invariant"


def test_piatetsky_shapiro_p1():
    """P1: the measure is zero when A and B are independent."""
    m = _m(property_matrix())
    for k in ("leverage", "phi", "kappa", "added_value", "certainty_factor",
              "chi_square", "j_measure", "zhang"):
        assert m[k]["P1"], f"{k} should vanish at independence"
    for k in ("support", "confidence", "lift", "cosine", "jaccard"):
        assert not m[k]["P1"], f"{k} is not zero at independence"


def test_symmetric_measures():
    """O1: symmetry under swapping antecedent and consequent."""
    m = _m(property_matrix())
    for k in ("lift", "leverage", "cosine", "jaccard", "kulczynski", "phi",
              "odds_ratio", "kappa", "chi_square"):
        assert m[k]["O1"], f"{k} should be symmetric"
    for k in ("confidence", "conviction", "certainty_factor", "added_value", "laplace"):
        assert not m[k]["O1"], f"{k} is directional and must not be symmetric"


def test_matrix_is_deterministic():
    a, b = property_matrix(seed=7), property_matrix(seed=7)
    assert a["measures"] == b["measures"]
