"""CRISP-DM Phase 2 - Data Understanding.

Acquire the Kaggle dataset, then produce a machine-readable profile and a data
quality audit. Every issue raised here has to be answered by Phase 3 (prepare.py)
or explicitly accepted in the model card - nothing is allowed to be silently fixed.
"""
from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config

# --------------------------------------------------------------------------- #
# Domain semantics - the feature dictionary a stakeholder actually reads
# --------------------------------------------------------------------------- #
FEATURE_DICTIONARY: dict[str, dict] = {
    "BALANCE":                          dict(kind="monetary",  unit="USD",   desc="Revolving balance left in the account to make purchases."),
    "BALANCE_FREQUENCY":                dict(kind="frequency", unit="0-1",   desc="How often the balance is updated (1 = frequently updated)."),
    "PURCHASES":                        dict(kind="monetary",  unit="USD",   desc="Total value of purchases made from the account."),
    "ONEOFF_PURCHASES":                 dict(kind="monetary",  unit="USD",   desc="Largest-ticket purchases done in one go."),
    "INSTALLMENTS_PURCHASES":           dict(kind="monetary",  unit="USD",   desc="Purchases done in instalments."),
    "CASH_ADVANCE":                     dict(kind="monetary",  unit="USD",   desc="Cash taken in advance against the card."),
    "PURCHASES_FREQUENCY":              dict(kind="frequency", unit="0-1",   desc="How frequently purchases are being made."),
    "ONEOFF_PURCHASES_FREQUENCY":       dict(kind="frequency", unit="0-1",   desc="How frequently one-off purchases occur."),
    "PURCHASES_INSTALLMENTS_FREQUENCY": dict(kind="frequency", unit="0-1",   desc="How frequently instalment purchases occur."),
    "CASH_ADVANCE_FREQUENCY":           dict(kind="frequency", unit="0-1",   desc="How frequently cash in advance is taken."),
    "CASH_ADVANCE_TRX":                 dict(kind="count",     unit="count", desc="Number of cash-advance transactions."),
    "PURCHASES_TRX":                    dict(kind="count",     unit="count", desc="Number of purchase transactions."),
    "CREDIT_LIMIT":                     dict(kind="monetary",  unit="USD",   desc="Credit limit granted on the card."),
    "PAYMENTS":                         dict(kind="monetary",  unit="USD",   desc="Amount of payment made by the user."),
    "MINIMUM_PAYMENTS":                 dict(kind="monetary",  unit="USD",   desc="Minimum amount of payment made by the user."),
    "PRC_FULL_PAYMENT":                 dict(kind="ratio",     unit="0-1",   desc="Percent of full payment paid by the user."),
    "TENURE":                           dict(kind="count",     unit="months", desc="Tenure of credit-card service for the user."),
}
MONETARY = [c for c, m in FEATURE_DICTIONARY.items() if m["kind"] == "monetary"]
BOUNDED_01 = [c for c, m in FEATURE_DICTIONARY.items() if m["unit"] == "0-1"]
FEATURES = list(FEATURE_DICTIONARY)


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def load_raw(force_download: bool = False) -> tuple[pd.DataFrame, dict]:
    """Fetch CC GENERAL.csv from Kaggle (anonymous, cached) into data/raw/."""
    local = config.DATA_RAW / "cc_general.csv"
    provenance = {
        "dataset": config.KAGGLE_DATASET,
        "url": config.KAGGLE_URL,
        "file": config.KAGGLE_FILE,
        "source": "cache",
    }
    if force_download or not local.exists():
        import kagglehub  # imported lazily so offline reruns work off the cached copy
        from pathlib import Path
        cached = Path(kagglehub.dataset_download(config.KAGGLE_DATASET)) / config.KAGGLE_FILE
        shutil.copyfile(cached, local)
        provenance["source"] = "kagglehub"
    df = pd.read_csv(local)
    provenance["sha256"] = _sha256(local)
    provenance["rows"], provenance["cols"] = df.shape
    return df, provenance


def _sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def profile(df: pd.DataFrame) -> dict:
    """Univariate profile + correlation structure + intrinsic-dimension estimate."""
    X = df[FEATURES]
    rows = []
    for col in FEATURES:
        s = X[col].astype(float)
        v = s.dropna().to_numpy()
        rows.append({
            "feature": col,
            "kind": FEATURE_DICTIONARY[col]["kind"],
            "unit": FEATURE_DICTIONARY[col]["unit"],
            "description": FEATURE_DICTIONARY[col]["desc"],
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * float(s.isna().mean()), 2),
            "zeros_pct": round(100 * float((v == 0).mean()), 2),
            "unique": int(s.nunique()),
            "mean": _f(v.mean()), "std": _f(v.std(ddof=1)),
            "min": _f(v.min()), "p25": _f(np.percentile(v, 25)),
            "median": _f(np.median(v)), "p75": _f(np.percentile(v, 75)),
            "p95": _f(np.percentile(v, 95)), "p99": _f(np.percentile(v, 99)),
            "max": _f(v.max()),
            "skew": _f(stats.skew(v)),
            "kurtosis": _f(stats.kurtosis(v)),
            # tail mass: share of the total held by the top 1% of records
            "top1pct_share": _f(100 * v[v >= np.percentile(v, 99)].sum() / v.sum()) if v.sum() > 0 else 0.0,
            "histogram": _hist(v),
        })

    corr = X.corr(method="spearman").fillna(0.0)
    eig = np.linalg.eigvalsh(np.corrcoef(_zfill(X.to_numpy(dtype=float)).T))[::-1]
    eig = np.clip(eig, 0, None)
    explained = eig / eig.sum()
    return {
        "n_rows": int(len(df)),
        "n_features": len(FEATURES),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "features": rows,
        "correlation": {
            "labels": FEATURES,
            "matrix": [[round(float(corr.iat[i, j]), 3) for j in range(len(FEATURES))]
                       for i in range(len(FEATURES))],
            "top_pairs": _top_pairs(corr),
        },
        "pca_scree": {
            "explained": [round(float(x), 4) for x in explained],
            "cumulative": [round(float(x), 4) for x in np.cumsum(explained)],
            # Kaiser criterion on the correlation matrix + 90% variance rule
            "n_kaiser": int((eig > 1.0).sum()),
            "n_90pct": int(np.searchsorted(np.cumsum(explained), 0.90) + 1),
        },
    }


def _zfill(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    col_med = np.nanmedian(a, axis=0)
    idx = np.where(np.isnan(a))
    a[idx] = np.take(col_med, idx[1])
    return a


def _f(x) -> float:
    x = float(x)
    return round(x, 4) if abs(x) < 1000 else round(x, 1)


def _hist(v: np.ndarray, bins: int = 28) -> dict:
    hi = float(np.percentile(v, 99.5))
    lo = float(v.min())
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(np.clip(v, lo, hi), bins=bins, range=(lo, hi))
    return {"counts": [int(c) for c in counts],
            "edges": [round(float(e), 3) for e in edges],
            "clipped_at_p995": True}


def _top_pairs(corr: pd.DataFrame, n: int = 10) -> list[dict]:
    out = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            out.append({"a": cols[i], "b": cols[j], "rho": round(float(corr.iat[i, j]), 3)})
    return sorted(out, key=lambda d: -abs(d["rho"]))[:n]


# --------------------------------------------------------------------------- #
# Data quality audit
# --------------------------------------------------------------------------- #
@dataclass
class Issue:
    id: str
    severity: str          # blocker | high | medium | low
    feature: str
    finding: str
    evidence: str
    resolution: str        # what Phase 3 does about it

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def audit(df: pd.DataFrame) -> list[dict]:
    """Rule-driven audit. Each finding names the Phase-3 action that answers it."""
    issues: list[Issue] = []
    n = len(df)

    # 1. Missingness
    for col in FEATURES:
        miss = int(df[col].isna().sum())
        if miss:
            pct = 100 * miss / n
            sev = "high" if pct > 2 else "medium"
            # Is missingness informative? Compare a proxy behaviour across the mask.
            mask = df[col].isna()
            ref = "PURCHASES" if col != "PURCHASES" else "BALANCE"
            d = _cohens_d(df.loc[~mask, ref].to_numpy(float), df.loc[mask, ref].to_numpy(float))
            mech = ("not MCAR: missing rows differ on %s (Cohen's d=%.2f)" % (ref, d)
                    if abs(d) > 0.2 else "consistent with MCAR on observed behaviour")
            issues.append(Issue(
                f"DQ-{len(issues)+1:02d}", sev, col,
                f"{miss} missing values ({pct:.2f}%)",
                mech,
                "Imputation strategy is a searched hyperparameter (median / KNN / iterative); "
                "a missing-indicator flag is added so the model can use the absence itself.",
            ))

    # 2. Range violations against the declared domain
    for col in BOUNDED_01:
        over = int((df[col] > 1.0).sum())
        if over:
            issues.append(Issue(
                f"DQ-{len(issues)+1:02d}", "high", col,
                f"{over} records exceed the documented [0,1] domain (max={df[col].max():.2f})",
                "Source dictionary defines this as a frequency in [0,1]; values up to 1.5 are present.",
                "Clipped to [0,1] in the fixed cleaning step and counted, rather than dropped - "
                "the affected accounts are otherwise valid.",
            ))

    # 3. Accounting identities that should hold
    lhs = df["ONEOFF_PURCHASES"] + df["INSTALLMENTS_PURCHASES"]
    broken = int((np.abs(lhs - df["PURCHASES"]) > 1.0).sum())
    if broken:
        issues.append(Issue(
            f"DQ-{len(issues)+1:02d}", "medium", "PURCHASES",
            f"{broken} records ({100*broken/n:.1f}%) violate PURCHASES = ONEOFF + INSTALLMENTS",
            f"median absolute discrepancy = {float(np.abs(lhs - df['PURCHASES']).median()):.2f} USD",
            "Retained. The decomposition is treated as three correlated views, not an identity; "
            "the discrepancy itself is engineered as a feature only if it survives the search.",
        ))

    # 4. Degenerate / near-constant features
    for col in FEATURES:
        top = df[col].value_counts(normalize=True, dropna=True)
        if len(top) and float(top.iloc[0]) > 0.80:
            issues.append(Issue(
                f"DQ-{len(issues)+1:02d}", "medium", col,
                f"near-constant: {100*float(top.iloc[0]):.1f}% of records share the value {top.index[0]}",
                f"{int(df[col].nunique())} distinct values over {n} records",
                "Low-variance feature contributes almost nothing to a distance metric; a variance "
                "filter is one of the searched feature-selection options.",
            ))

    # 5. Heavy tails - the dominant modelling risk for distance-based clustering
    for col in MONETARY:
        sk = float(stats.skew(df[col].dropna()))
        if sk > 3:
            issues.append(Issue(
                f"DQ-{len(issues)+1:02d}", "high", col,
                f"severe right skew (skewness={sk:.1f})",
                f"top 1% of accounts hold {100*df[col].nlargest(max(1,n//100)).sum()/df[col].sum():.0f}% "
                f"of all {col}",
                "Euclidean distance would be dominated by a handful of whales; a variance-stabilising "
                "transform (log1p / Yeo-Johnson / quantile) is a searched hyperparameter.",
            ))

    # 6. Structural zeros (a real behaviour, not a defect - flag so it is not imputed away)
    for col in ["CASH_ADVANCE", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES"]:
        z = 100 * float((df[col] == 0).mean())
        if z > 40:
            issues.append(Issue(
                f"DQ-{len(issues)+1:02d}", "low", col,
                f"{z:.0f}% exact zeros",
                "zero-inflation is genuine inactivity, not a coding error",
                "Preserved. log1p handles the zero point without a shift constant; zeros are never "
                "treated as missing.",
            ))

    # 7. Identifier hygiene
    dup = int(df[config.ID_COLUMN].duplicated().sum())
    issues.append(Issue(
        f"DQ-{len(issues)+1:02d}", "low" if dup == 0 else "blocker", config.ID_COLUMN,
        f"{dup} duplicate customer identifiers",
        f"{df[config.ID_COLUMN].nunique()} unique IDs over {n} rows",
        "ID is excluded from every feature matrix; it exists only to join scores back to the book.",
    ))

    return [i.to_dict() for i in issues]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def clustering_tendency(X: np.ndarray, seed: int = config.RANDOM_SEED, m_frac: float = 0.05) -> dict:
    """Hopkins statistic - is there structure at all, or are we clustering noise?

    Lawson & Jurs (1990). H ~ 0.5 => uniformly random (clustering is meaningless);
    H -> 1.0 => strongly clusterable. Computed on standardised data.
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m = max(20, int(m_frac * n))
    idx = rng.choice(n, m, replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    # u: distances from uniform random points in the data's bounding box
    lo, hi = X.min(axis=0), X.max(axis=0)
    U = rng.uniform(lo, hi, size=(m, d))
    u = nn.kneighbors(U, n_neighbors=1, return_distance=True)[0][:, 0]
    # w: distances from real sampled points to their nearest *other* real point
    w = nn.kneighbors(X[idx], n_neighbors=2, return_distance=True)[0][:, 1]
    H = float(u.sum() / (u.sum() + w.sum()))
    return {"hopkins": round(H, 4), "n_sampled": m,
            "interpretation": ("strong clustering tendency" if H > 0.75 else
                               "moderate clustering tendency" if H > 0.6 else
                               "weak - close to uniform random")}
