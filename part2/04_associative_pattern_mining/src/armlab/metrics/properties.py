"""Empirical verification of the measure properties of Tan, Kumar & Srivastava
(KDD 2002), plus the Piatetsky-Shapiro (1991) axioms.

Rather than transcribing the paper's Table 2 and hoping we copied it correctly,
we *derive* it: each property is a numerical experiment over randomly generated
2x2 contingency tables, and a measure "satisfies" a property only if it passes
on every probe.  The dashboard renders the derived matrix, so the claim it
makes about any measure is reproducible by re-running this module.

Properties
----------
P1  M = 0 when A and B are statistically independent.       [Piatetsky-Shapiro 1991]
P2  M increases monotonically with n11 when n1., n.1 fixed.  [Piatetsky-Shapiro 1991]
P3  M decreases monotonically with n1. when n11, n.1 fixed.  [Piatetsky-Shapiro 1991]
O1  Symmetry under variable permutation: M(A=>B) == M(B=>A). [Tan et al. 2002]
O2  Row/column scaling invariance (grade invariance).        [Tan et al. 2002]
O3  Inversion invariance: swap n11<->n00 and n10<->n01.      [Tan et al. 2002]
O4  Null invariance: adding transactions containing neither
    A nor B leaves M unchanged.                              [Tan et al. 2002]
"""
from __future__ import annotations

from typing import Any

import numpy as np

from armlab.metrics.interestingness import MEASURES, Cont

TOL = 1e-6


def cells(n11, n10, n01, n00) -> Cont:
    n11 = np.asarray(n11, dtype=float); n10 = np.asarray(n10, dtype=float)
    n01 = np.asarray(n01, dtype=float); n00 = np.asarray(n00, dtype=float)
    return Cont.of(n11, n11 + n10, n11 + n01, n11 + n10 + n01 + n00)


def _eval(key: str, c: Cont) -> np.ndarray:
    with np.errstate(all="ignore"):
        v = np.asarray(MEASURES[key].fn(c), dtype=float)
    return np.nan_to_num(v, nan=0.0, posinf=1e9, neginf=-1e9)


def _rel_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return bool(np.all(np.abs(a - b) / scale < tol))


def _random_tables(rng: np.random.Generator, n: int = 400) -> Cont:
    """Well-conditioned tables: no zero margins, no zero cells."""
    m = rng.integers(5, 300, size=(n, 4)).astype(float)
    return cells(m[:, 0], m[:, 1], m[:, 2], m[:, 3])


# ---------------------------------------------------------------- P1
def check_p1(key: str, rng) -> bool:
    N = rng.integers(500, 5000, size=300).astype(float)
    pA = rng.uniform(0.1, 0.8, size=300)
    pB = rng.uniform(0.1, 0.8, size=300)
    n1_, n_1 = N * pA, N * pB
    n11 = n1_ * n_1 / N                      # exact independence
    c = Cont.of(n11, n1_, n_1, N)
    return bool(np.all(np.abs(_eval(key, c)) < 1e-6))


# ---------------------------------------------------------------- P2 / P3
def _monotone(vals: np.ndarray, direction: str) -> bool:
    d = np.diff(vals)
    if np.max(np.abs(vals)) - np.min(np.abs(vals)) < TOL and np.ptp(vals) < TOL:
        return False                          # constant: not monotone in the
                                              # strict sense the axioms require
    scale = max(1.0, float(np.max(np.abs(vals))))
    if direction == "up":
        return bool(np.all(d > -TOL * scale) and np.ptp(vals) > TOL)
    return bool(np.all(d < TOL * scale) and np.ptp(vals) > TOL)


def check_p2(key: str, rng, trials: int = 40) -> bool:
    """Increase n11 holding n1., n.1, N fixed."""
    for _ in range(trials):
        N = float(rng.integers(1000, 4000))
        n1_ = float(rng.integers(int(0.2 * N), int(0.6 * N)))
        n_1 = float(rng.integers(int(0.2 * N), int(0.6 * N)))
        lo = max(1.0, n1_ + n_1 - N + 1)
        hi = min(n1_, n_1) - 1
        if hi - lo < 10:
            continue
        grid = np.linspace(lo, hi, 25)
        c = Cont.of(grid, np.full_like(grid, n1_), np.full_like(grid, n_1),
                    np.full_like(grid, N))
        if not _monotone(_eval(key, c), "up"):
            return False
    return True


def check_p3(key: str, rng, trials: int = 40) -> bool:
    """Increase n1. holding n11, n.1, N fixed -- M must decrease."""
    for _ in range(trials):
        N = float(rng.integers(1000, 4000))
        n_1 = float(rng.integers(int(0.2 * N), int(0.5 * N)))
        n11 = float(rng.integers(int(0.05 * N), int(0.15 * N)))
        lo = n11 + 1
        hi = min(N - n_1 + n11 - 1, 0.85 * N)
        if hi - lo < 10:
            continue
        grid = np.linspace(lo, hi, 25)
        c = Cont.of(np.full_like(grid, n11), grid, np.full_like(grid, n_1),
                    np.full_like(grid, N))
        if not _monotone(_eval(key, c), "down"):
            return False
    return True


# ---------------------------------------------------------------- O1
def check_o1(key: str, rng) -> bool:
    m = rng.integers(5, 300, size=(400, 4)).astype(float)
    a = cells(m[:, 0], m[:, 1], m[:, 2], m[:, 3])
    b = cells(m[:, 0], m[:, 2], m[:, 1], m[:, 3])   # transpose the table
    return _rel_equal(_eval(key, a), _eval(key, b))


# ---------------------------------------------------------------- O2
def check_o2(key: str, rng) -> bool:
    m = rng.integers(20, 200, size=(300, 4)).astype(float)
    k = rng.uniform(1.5, 4.0, size=(300, 4))         # k1 row1, k2 row2, k3 col1, k4 col2
    a = cells(m[:, 0], m[:, 1], m[:, 2], m[:, 3])
    b = cells(m[:, 0] * k[:, 0] * k[:, 2], m[:, 1] * k[:, 0] * k[:, 3],
              m[:, 2] * k[:, 1] * k[:, 2], m[:, 3] * k[:, 1] * k[:, 3])
    return _rel_equal(_eval(key, a), _eval(key, b), tol=1e-5)


# ---------------------------------------------------------------- O3
def check_o3(key: str, rng) -> bool:
    m = rng.integers(5, 300, size=(400, 4)).astype(float)
    a = cells(m[:, 0], m[:, 1], m[:, 2], m[:, 3])
    b = cells(m[:, 3], m[:, 2], m[:, 1], m[:, 0])   # n11<->n00, n10<->n01
    return _rel_equal(_eval(key, a), _eval(key, b))


# ---------------------------------------------------------------- O4
def check_o4(key: str, rng) -> bool:
    m = rng.integers(20, 200, size=(300, 4)).astype(float)
    add = rng.integers(50, 2000, size=300).astype(float)
    a = cells(m[:, 0], m[:, 1], m[:, 2], m[:, 3])
    b = cells(m[:, 0], m[:, 1], m[:, 2], m[:, 3] + add)
    return _rel_equal(_eval(key, a), _eval(key, b), tol=1e-5)


PROPERTIES = {
    "P1": ("M = 0 at independence", "Piatetsky-Shapiro 1991", check_p1),
    "P2": ("increases with P(A,B)", "Piatetsky-Shapiro 1991", check_p2),
    "P3": ("decreases with P(A)", "Piatetsky-Shapiro 1991", check_p3),
    "O1": ("symmetry A=>B vs B=>A", "Tan, Kumar & Srivastava 2002", check_o1),
    "O2": ("row/column scaling invariance", "Tan, Kumar & Srivastava 2002", check_o2),
    "O3": ("inversion invariance", "Tan, Kumar & Srivastava 2002", check_o3),
    "O4": ("null invariance", "Tan, Kumar & Srivastava 2002", check_o4),
}


def property_matrix(seed: int = 20260901) -> dict[str, Any]:
    rows = []
    for key, meas in MEASURES.items():
        rng = np.random.default_rng(seed)
        row = {"key": key, "label": meas.label, "family": meas.family,
               "citation": meas.citation, "formula": meas.formula}
        for pkey, (_, _, fn) in PROPERTIES.items():
            try:
                row[pkey] = bool(fn(key, rng))
            except Exception:
                row[pkey] = False
        rows.append(row)
    return {
        "properties": [{"key": k, "description": d, "citation": c}
                       for k, (d, c, _) in PROPERTIES.items()],
        "measures": rows,
        "method": "Each cell is derived numerically from randomly generated 2x2 "
                  "contingency tables (seed fixed), not transcribed from the "
                  "paper. Re-running this module reproduces the matrix exactly.",
        "seed": seed,
    }
