"""Interestingness measures for association rules, with provenance.

Every measure is defined over the 2x2 contingency table of a rule A => B:

                 B        not B      total
     A         n11        n10         n1_
     not A     n01        n00         n0_
     total     n_1        n_0         N

`MEASURES` is the single source of truth: the dashboard's metric glossary, the
rule table columns, the AutoResearch objective terms and the property matrix
are all generated from it, so a measure can never be documented one way and
computed another.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Cont:
    """Vectorised contingency tables. All fields are float arrays of equal shape."""
    n11: np.ndarray
    n1_: np.ndarray
    n_1: np.ndarray
    N: np.ndarray

    @property
    def n10(self) -> np.ndarray: return self.n1_ - self.n11
    @property
    def n01(self) -> np.ndarray: return self.n_1 - self.n11
    @property
    def n00(self) -> np.ndarray: return self.N - self.n1_ - self.n_1 + self.n11
    @property
    def n0_(self) -> np.ndarray: return self.N - self.n1_
    @property
    def n_0(self) -> np.ndarray: return self.N - self.n_1
    # probabilities
    @property
    def pAB(self) -> np.ndarray: return self.n11 / self.N
    @property
    def pA(self) -> np.ndarray: return self.n1_ / self.N
    @property
    def pB(self) -> np.ndarray: return self.n_1 / self.N

    @staticmethod
    def of(n11, n1_, n_1, N) -> "Cont":
        f = lambda x: np.asarray(x, dtype=float)
        return Cont(f(n11), f(n1_), f(n_1), f(N))


def _safe(num, den):
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(np.asarray(num, dtype=float)),
                     where=np.abs(den) > EPS)


# --------------------------------------------------------------------------
# Measure implementations
# --------------------------------------------------------------------------
def m_support(c: Cont): return c.pAB
def m_antecedent_support(c: Cont): return c.pA
def m_consequent_support(c: Cont): return c.pB
def m_confidence(c: Cont): return _safe(c.n11, c.n1_)
def m_lift(c: Cont): return _safe(c.pAB, c.pA * c.pB)
def m_leverage(c: Cont): return c.pAB - c.pA * c.pB


def m_conviction(c: Cont):
    """(1-P(B))/(1-conf); +inf for an exceptionless rule -> capped for sorting."""
    conf = m_confidence(c)
    den = 1.0 - conf
    out = np.where(np.abs(den) > EPS, _safe(1.0 - c.pB, den), np.inf)
    return np.where(np.isfinite(out), out, 1e9)


def m_cosine(c: Cont): return _safe(c.pAB, np.sqrt(c.pA * c.pB))
def m_jaccard(c: Cont): return _safe(c.pAB, c.pA + c.pB - c.pAB)
def m_all_confidence(c: Cont): return _safe(c.pAB, np.maximum(c.pA, c.pB))
def m_max_confidence(c: Cont):
    return np.maximum(_safe(c.n11, c.n1_), _safe(c.n11, c.n_1))
def m_kulczynski(c: Cont):
    return 0.5 * (_safe(c.n11, c.n1_) + _safe(c.n11, c.n_1))
def m_imbalance_ratio(c: Cont):
    return _safe(np.abs(c.pA - c.pB), c.pA + c.pB - c.pAB)
def m_added_value(c: Cont): return m_confidence(c) - c.pB
def m_certainty_factor(c: Cont):
    return _safe(m_confidence(c) - c.pB, 1.0 - c.pB)
def m_laplace(c: Cont): return _safe(c.n11 + 1.0, c.n1_ + 2.0)
def m_klosgen(c: Cont): return np.sqrt(np.clip(c.pAB, 0, None)) * m_added_value(c)
def m_zhang(c: Cont):
    """Zhang (2000): +1 perfect association, -1 perfect dissociation."""
    num = c.pAB - c.pA * c.pB
    den = np.maximum(c.pAB * (1 - c.pB), c.pB * (c.pA - c.pAB))
    return _safe(num, den)


def m_phi(c: Cont):
    num = c.n11 * c.n00 - c.n10 * c.n01
    den = np.sqrt(c.n1_ * c.n_1 * c.n0_ * c.n_0)
    return _safe(num, den)


def m_odds_ratio(c: Cont):
    num, den = c.n11 * c.n00, c.n10 * c.n01
    out = np.where(np.abs(den) > EPS, _safe(num, den), 1e9)
    return np.clip(out, 0.0, 1e9)


def m_yule_q(c: Cont):
    a = c.n11 * c.n00; b = c.n10 * c.n01
    return _safe(a - b, a + b)


def m_yule_y(c: Cont):
    a = np.sqrt(c.n11 * c.n00); b = np.sqrt(c.n10 * c.n01)
    return _safe(a - b, a + b)


def m_kappa(c: Cont):
    po = (c.n11 + c.n00) / c.N
    pe = c.pA * c.pB + (1 - c.pA) * (1 - c.pB)
    return _safe(po - pe, 1.0 - pe)


def m_chi_square(c: Cont):
    """Pearson chi-square on the 2x2 table (Brin et al., SIGMOD 1997)."""
    tot = np.zeros_like(c.n11, dtype=float)
    for obs, r, col in ((c.n11, c.n1_, c.n_1), (c.n10, c.n1_, c.n_0),
                        (c.n01, c.n0_, c.n_1), (c.n00, c.n0_, c.n_0)):
        exp = _safe(r * col, c.N)
        tot = tot + _safe((obs - exp) ** 2, exp)
    return tot


def _xlogx_terms(c: Cont):
    """Joint cells and their independence expectations, as probabilities."""
    p = [c.n11 / c.N, c.n10 / c.N, c.n01 / c.N, c.n00 / c.N]
    q = [c.pA * c.pB, c.pA * (1 - c.pB), (1 - c.pA) * c.pB, (1 - c.pA) * (1 - c.pB)]
    return p, q


def m_mutual_information(c: Cont):
    """Normalised mutual information I(A;B)/min(H(A),H(B))."""
    p, q = _xlogx_terms(c)
    mi = np.zeros_like(c.n11, dtype=float)
    for pi, qi in zip(p, q):
        r = _safe(pi, qi)
        mi = mi + np.where(pi > EPS, pi * np.log(np.maximum(r, EPS)), 0.0)
    def H(x):
        x = np.clip(x, EPS, 1 - EPS)
        return -(x * np.log(x) + (1 - x) * np.log(1 - x))
    return _safe(mi, np.minimum(H(c.pA), H(c.pB)))


def m_j_measure(c: Cont):
    """Smyth & Goodman (1992) information-theoretic rule value."""
    conf = m_confidence(c)
    t1 = np.where(conf > EPS, conf * np.log(np.maximum(_safe(conf, c.pB), EPS)), 0.0)
    t2 = np.where(1 - conf > EPS,
                  (1 - conf) * np.log(np.maximum(_safe(1 - conf, 1 - c.pB), EPS)), 0.0)
    return c.pA * (t1 + t2)


def m_gini(c: Cont):
    cA = _safe(c.n11, c.n1_); cnA = _safe(c.n01, c.n0_)
    g = (c.pA * (cA ** 2 + (1 - cA) ** 2)
         + (1 - c.pA) * (cnA ** 2 + (1 - cnA) ** 2)
         - c.pB ** 2 - (1 - c.pB) ** 2)
    return g


def m_collective_strength(c: Cont):
    agree = (c.n11 + c.n00) / c.N
    exp_agree = c.pA * c.pB + (1 - c.pA) * (1 - c.pB)
    return _safe(agree * (1 - exp_agree), exp_agree * (1 - agree))


def m_goodman_kruskal(c: Cont):
    """Goodman-Kruskal lambda: proportional reduction in prediction error."""
    num = (np.maximum(c.n11, c.n10) + np.maximum(c.n01, c.n00)
           + np.maximum(c.n11, c.n01) + np.maximum(c.n10, c.n00)
           - np.maximum(c.n_1, c.n_0) - np.maximum(c.n1_, c.n0_))
    den = 2 * c.N - np.maximum(c.n_1, c.n_0) - np.maximum(c.n1_, c.n0_)
    return _safe(num, den)


def m_two_way_support(c: Cont):
    return c.pAB * np.log(np.maximum(_safe(c.pAB, c.pA * c.pB), EPS))


def m_sebag_schoenauer(c: Cont): return _safe(c.n11, c.n10)
def m_lever_count(c: Cont): return c.n11


@dataclass(frozen=True)
class Measure:
    key: str
    label: str
    fn: Callable[[Cont], np.ndarray]
    citation: str
    formula: str
    range: str
    higher_is_better: bool = True
    family: str = "association"
    note: str = ""


MEASURES: dict[str, Measure] = {m.key: m for m in [
    Measure("support", "Support", m_support,
            "Agrawal, Imielinski & Swami, SIGMOD 1993",
            "P(A,B)", "[0,1]", family="frequency",
            note="Business reach of the rule. The only measure with the "
                 "downward-closure property that makes mining tractable."),
    Measure("antecedent_support", "Antecedent support", m_antecedent_support,
            "Agrawal et al., SIGMOD 1993", "P(A)", "[0,1]", family="frequency"),
    Measure("consequent_support", "Consequent support", m_consequent_support,
            "Agrawal et al., SIGMOD 1993", "P(B)", "[0,1]", family="frequency"),
    Measure("confidence", "Confidence", m_confidence,
            "Agrawal, Imielinski & Swami, SIGMOD 1993",
            "P(B|A)", "[0,1]", family="association",
            note="Not a measure of association: a rule whose consequent is "
                 "universally popular scores high while carrying no information."),
    Measure("lift", "Lift (Interest)", m_lift,
            "Brin, Motwani, Ullman & Tsur, SIGMOD 1997",
            "P(A,B) / (P(A)P(B))", "[0,inf)", family="association",
            note="1.0 means independence. Sensitive to rare items -- the "
                 "'rare item problem'."),
    Measure("leverage", "Leverage", m_leverage,
            "Piatetsky-Shapiro, KDD 1991",
            "P(A,B) - P(A)P(B)", "[-0.25,0.25]", family="association",
            note="Absolute excess co-occurrence; prefers frequent itemsets, so "
                 "it counterbalances lift."),
    Measure("conviction", "Conviction", m_conviction,
            "Brin, Motwani, Ullman & Tsur, SIGMOD 1997",
            "(1 - P(B)) / (1 - conf)", "[0,inf)", family="association",
            note="Directional: measures how often the rule would be wrong if A "
                 "and B were independent."),
    Measure("zhang", "Zhang's metric", m_zhang,
            "Zhang, 2000", "(P(A,B)-P(A)P(B)) / max(P(A,B)(1-P(B)), P(B)(P(A)-P(A,B)))",
            "[-1,1]", family="association",
            note="Bounded association strength; negative values are dissociations."),
    Measure("cosine", "Cosine (IS)", m_cosine,
            "Tan, Kumar & Srivastava, KDD 2002",
            "P(A,B) / sqrt(P(A)P(B))", "[0,1]", family="null-invariant"),
    Measure("jaccard", "Jaccard", m_jaccard,
            "Tan, Kumar & Srivastava, KDD 2002",
            "P(A,B) / (P(A)+P(B)-P(A,B))", "[0,1]", family="null-invariant"),
    Measure("all_confidence", "All-confidence", m_all_confidence,
            "Omiecinski, IEEE TKDE 2003",
            "P(A,B) / max(P(A), P(B))", "[0,1]", family="null-invariant"),
    Measure("max_confidence", "Max-confidence", m_max_confidence,
            "Wu, Chen & Han, ACM TKDD 2010",
            "max(P(B|A), P(A|B))", "[0,1]", family="null-invariant"),
    Measure("kulczynski", "Kulczynski", m_kulczynski,
            "Wu, Chen & Han, ACM TKDD 2010",
            "(P(B|A) + P(A|B)) / 2", "[0,1]", family="null-invariant",
            note="Recommended with Imbalance Ratio: Kulczynski says how strong, "
                 "IR says how lopsided."),
    Measure("imbalance_ratio", "Imbalance ratio", m_imbalance_ratio,
            "Wu, Chen & Han, ACM TKDD 2010",
            "|P(A)-P(B)| / (P(A)+P(B)-P(A,B))", "[0,1]", higher_is_better=False,
            family="null-invariant"),
    Measure("phi", "Phi coefficient", m_phi,
            "Tan, Kumar & Srivastava, KDD 2002",
            "(n11*n00 - n10*n01) / sqrt(n1. n.1 n0. n.0)", "[-1,1]",
            family="correlation", note="Pearson correlation for binary variables."),
    Measure("odds_ratio", "Odds ratio", m_odds_ratio,
            "Mosteller, 1968", "(n11*n00) / (n10*n01)", "[0,inf)",
            family="correlation"),
    Measure("yule_q", "Yule's Q", m_yule_q,
            "Yule, 1900", "(alpha-1)/(alpha+1)", "[-1,1]", family="correlation"),
    Measure("yule_y", "Yule's Y", m_yule_y,
            "Yule, 1912", "(sqrt(alpha)-1)/(sqrt(alpha)+1)", "[-1,1]",
            family="correlation"),
    Measure("kappa", "Cohen's kappa", m_kappa,
            "Cohen, 1960", "(Po - Pe)/(1 - Pe)", "[-1,1]", family="correlation"),
    Measure("chi_square", "Chi-square", m_chi_square,
            "Brin, Motwani, Ullman & Tsur, SIGMOD 1997",
            "sum (obs-exp)^2/exp over the 2x2 table", "[0,inf)",
            family="significance"),
    Measure("mutual_information", "Normalised mutual information", m_mutual_information,
            "Tan, Kumar & Srivastava, KDD 2002",
            "I(A;B) / min(H(A), H(B))", "[0,1]", family="information"),
    Measure("j_measure", "J-measure", m_j_measure,
            "Smyth & Goodman, IEEE TKDE 1992",
            "P(A) * KL(P(B|A) || P(B))", "[0,1]", family="information",
            note="Average information content the rule delivers per transaction."),
    Measure("gini", "Gini index", m_gini,
            "Tan, Kumar & Srivastava, KDD 2002",
            "impurity reduction of B given A", "[0,1]", family="information"),
    Measure("two_way_support", "Two-way support", m_two_way_support,
            "Yao & Zhong, PAKDD 1999", "P(A,B) log(P(A,B)/(P(A)P(B)))",
            "(-inf,inf)", family="information"),
    Measure("collective_strength", "Collective strength", m_collective_strength,
            "Aggarwal & Yu, PODS 1998",
            "(agreement/expected) * ((1-expected)/(1-agreement))", "[0,inf)",
            family="correlation"),
    Measure("goodman_kruskal", "Goodman-Kruskal lambda", m_goodman_kruskal,
            "Goodman & Kruskal, JASA 1954", "proportional reduction in error",
            "[0,1]", family="correlation"),
    Measure("certainty_factor", "Certainty factor", m_certainty_factor,
            "Shortliffe & Buchanan, 1975",
            "(conf - P(B)) / (1 - P(B))", "[-1,1]", family="association"),
    Measure("added_value", "Added value", m_added_value,
            "Sahar & Mansour, 1999", "conf - P(B)", "[-1,1]", family="association",
            note="How much A actually moves the needle over the base rate."),
    Measure("laplace", "Laplace correction", m_laplace,
            "Clark & Boswell, EWSL 1991", "(n11+1)/(n1.+2)", "[0,1]",
            family="association",
            note="Shrinks confidence toward 0.5 for low-count antecedents."),
    Measure("klosgen", "Klosgen", m_klosgen,
            "Klosgen, 1996", "sqrt(P(A,B)) * (conf - P(B))", "[-0.38,0.38]",
            family="association"),
    Measure("sebag_schoenauer", "Sebag-Schoenauer", m_sebag_schoenauer,
            "Sebag & Schoenauer, EKAW 1988", "n11 / n10", "[0,inf)",
            family="association"),
    Measure("count", "Support count", m_lever_count,
            "Agrawal et al., SIGMOD 1993", "n11", "[0,N]", family="frequency"),
]}

RANKING_MEASURES = [k for k, m in MEASURES.items() if m.family != "frequency"]


def compute_all(c: Cont) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return {k: np.nan_to_num(np.asarray(m.fn(c), dtype=float),
                                 nan=0.0, posinf=1e9, neginf=-1e9)
                for k, m in MEASURES.items()}


def glossary() -> list[dict]:
    return [{"key": m.key, "label": m.label, "citation": m.citation,
             "formula": m.formula, "range": m.range, "family": m.family,
             "higher_is_better": m.higher_is_better, "note": m.note}
            for m in MEASURES.values()]
