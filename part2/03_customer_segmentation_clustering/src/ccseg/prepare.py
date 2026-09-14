"""CRISP-DM Phase 3 - Data Preparation.

Two layers, deliberately separated:

  * ``fixed_clean``  - decisions that are *not* negotiable and are applied before any
    search: domain-range repair, id removal, feature engineering. Doing these inside
    the search would let the optimiser "tune away" a data-quality defect.
  * ``build_preprocessor`` - decisions that are genuinely uncertain (imputation,
    variance-stabilising transform, scaling, dimensionality reduction) and are
    therefore exposed to the AutoResearch optimiser as hyperparameters.

Everything is expressed as a scikit-learn Pipeline so the exact transform chain that
produced the champion is serialisable and replayable at scoring time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, MinMaxScaler, PowerTransformer, QuantileTransformer,
    RobustScaler, StandardScaler,
)

from . import config
from .data import BOUNDED_01, FEATURES, MONETARY

# --------------------------------------------------------------------------- #
# Fixed cleaning + feature engineering
# --------------------------------------------------------------------------- #
ENGINEERED = [
    "AVG_PURCHASE_TRX",      # basket size
    "AVG_CASH_ADVANCE_TRX",  # cash-advance ticket size
    "CREDIT_UTILISATION",    # balance / limit  - the classic risk ratio
    "PAYMENT_TO_MIN_RATIO",  # payments / minimum payments - repayment discipline
    "PURCHASE_TO_LIMIT",     # spend intensity relative to granted line
    "MONTHLY_SPEND",         # purchases / tenure - normalises the 6..12 month window
    "INSTALMENT_SHARE",      # instalments / purchases - product-mix preference
    "CASH_DEPENDENCE",       # cash advance / (cash advance + purchases)
    "MINPAY_IMPUTED",        # missing-indicator: absence is itself informative
]


def fixed_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the non-negotiable repairs and derive ratio features.

    Returns the cleaned frame and a ledger of exactly what was changed, so the
    dashboard can show the diff rather than asserting the data is "clean".
    """
    out = df.copy()
    ledger: dict = {"actions": []}

    # DQ-03: repair values outside the documented [0,1] frequency domain.
    for col in BOUNDED_01:
        over = int((out[col] > 1.0).sum())
        if over:
            out[col] = out[col].clip(0.0, 1.0)
            ledger["actions"].append(
                {"action": "clip_to_domain", "feature": col, "affected": over,
                 "detail": "values > 1.0 clipped to 1.0 (documented domain is [0,1])"})

    # DQ-02/01: record where the value was absent *before* imputation runs.
    out["MINPAY_IMPUTED"] = out["MINIMUM_PAYMENTS"].isna().astype(float)
    ledger["actions"].append(
        {"action": "missing_indicator", "feature": "MINIMUM_PAYMENTS",
         "affected": int(out["MINPAY_IMPUTED"].sum()),
         "detail": "MINPAY_IMPUTED added; imputation itself is left to the searched strategy"})

    eps = 1e-9
    out["AVG_PURCHASE_TRX"] = out["PURCHASES"] / (out["PURCHASES_TRX"] + eps)
    out["AVG_CASH_ADVANCE_TRX"] = out["CASH_ADVANCE"] / (out["CASH_ADVANCE_TRX"] + eps)
    out["CREDIT_UTILISATION"] = out["BALANCE"] / (out["CREDIT_LIMIT"] + eps)
    out["PAYMENT_TO_MIN_RATIO"] = out["PAYMENTS"] / (out["MINIMUM_PAYMENTS"] + eps)
    out["PURCHASE_TO_LIMIT"] = out["PURCHASES"] / (out["CREDIT_LIMIT"] + eps)
    out["MONTHLY_SPEND"] = out["PURCHASES"] / out["TENURE"].clip(lower=1)
    out["INSTALMENT_SHARE"] = out["INSTALLMENTS_PURCHASES"] / (out["PURCHASES"] + eps)
    out["CASH_DEPENDENCE"] = out["CASH_ADVANCE"] / (out["CASH_ADVANCE"] + out["PURCHASES"] + eps)

    # Ratios with a near-zero denominator explode; cap at a high empirical quantile
    # instead of dropping the account.
    for col in ["AVG_PURCHASE_TRX", "AVG_CASH_ADVANCE_TRX", "CREDIT_UTILISATION",
                "PAYMENT_TO_MIN_RATIO", "PURCHASE_TO_LIMIT", "MONTHLY_SPEND"]:
        cap = float(out[col].quantile(0.995))
        n_capped = int((out[col] > cap).sum())
        out[col] = out[col].clip(upper=cap)
        ledger["actions"].append(
            {"action": "cap_ratio_p995", "feature": col, "affected": n_capped,
             "detail": f"ratio capped at its 99.5th percentile ({cap:.2f}) to bound division blow-up"})

    ledger["engineered_features"] = ENGINEERED
    ledger["n_features_before"] = len(FEATURES)
    ledger["n_features_after"] = len(FEATURES) + len(ENGINEERED)
    return out, ledger


# --------------------------------------------------------------------------- #
# Feature views - a searched choice, because "use everything" is a hypothesis
# --------------------------------------------------------------------------- #
CORE_VIEW = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
    "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT",
]
BEHAVIOUR_VIEW = [c for c in FEATURES if c not in MONETARY] + [
    "CREDIT_UTILISATION", "INSTALMENT_SHARE", "CASH_DEPENDENCE",
    "PAYMENT_TO_MIN_RATIO", "MONTHLY_SPEND",
]


def feature_view(name: str) -> list[str]:
    if name == "raw17":
        return list(FEATURES)
    if name == "engineered":
        return list(FEATURES) + ENGINEERED
    if name == "core8":
        return list(CORE_VIEW)
    if name == "behaviour":
        return list(dict.fromkeys(BEHAVIOUR_VIEW))
    raise ValueError(f"unknown feature view {name!r}")


# --------------------------------------------------------------------------- #
# Winsoriser (fit on train quantiles, applied at scoring time)
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column at fitted quantiles. Bounds the influence of extreme
    observations on the centroid without deleting the customer from the book."""

    def __init__(self, upper: float = 0.99):
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        self.lower_ = np.nanquantile(X, 1 - self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def _imputer(kind: str):
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "knn5":
        return KNNImputer(n_neighbors=5)
    if kind == "iterative":
        return IterativeImputer(max_iter=8, random_state=config.RANDOM_SEED, sample_posterior=False)
    raise ValueError(kind)


def _transform(kind: str):
    if kind == "none":
        return "passthrough"
    if kind == "log1p":
        # Monetary columns are non-negative and zero-inflated; log1p is exact at 0.
        return FunctionTransformer(lambda X: np.log1p(np.clip(X, 0, None)),
                                   feature_names_out="one-to-one")
    if kind == "yeojohnson":
        return PowerTransformer(method="yeo-johnson", standardize=False)
    if kind == "quantile_normal":
        return QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                   subsample=100_000, random_state=config.RANDOM_SEED)
    raise ValueError(kind)


def _scaler(kind: str):
    return {"standard": StandardScaler(), "robust": RobustScaler(),
            "minmax": MinMaxScaler()}[kind]


class SafePCA(PCA):
    """PCA whose requested rank is clamped to what the matrix can actually supply.

    The searched ``pca_n10`` is legal against the 26-column engineered view and
    illegal against the 8-column core view. Clamping (rather than letting the
    optimiser trip over a ValueError) keeps the two decisions - feature view and
    reduction depth - independent, which is what the neighbourhood structure assumes.
    """

    def fit(self, X, y=None):
        if isinstance(self.n_components, int):
            self.n_components = min(self.n_components, min(np.shape(X)))
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        if isinstance(self.n_components, int):
            self.n_components = min(self.n_components, min(np.shape(X)))
        return super().fit_transform(X, y)


def _reducer(kind: str, seed: int):
    if kind == "none":
        return "passthrough"
    if kind.startswith("pca_var"):
        return SafePCA(n_components=float(kind.split("_var")[1]) / 100.0, random_state=seed)
    if kind.startswith("pca_n"):
        return SafePCA(n_components=int(kind.split("_n")[1]), random_state=seed)
    raise ValueError(kind)


def build_preprocessor(cfg: dict, seed: int = config.RANDOM_SEED) -> Pipeline:
    """Assemble the searched preprocessing chain for one candidate configuration."""
    steps: list[tuple[str, object]] = [("impute", _imputer(cfg["imputer"]))]
    if cfg["winsorize"] != "none":
        steps.append(("winsorize", Winsorizer(upper=float(cfg["winsorize"]))))
    steps.append(("transform", _transform(cfg["transform"])))
    steps.append(("scale", _scaler(cfg["scaler"])))
    steps.append(("reduce", _reducer(cfg["reducer"], seed)))
    return Pipeline(steps)


def transform_effect(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Skewness before/after each candidate transform - the evidence behind the
    Phase-3 choice, rendered in the dashboard rather than asserted in prose."""
    from scipy import stats
    rows = []
    X = df[cols].to_numpy(dtype=float)
    X = SimpleImputer(strategy="median").fit_transform(X)
    for name in ["none", "log1p", "yeojohnson", "quantile_normal"]:
        t = _transform(name)
        Xt = X if t == "passthrough" else t.fit_transform(X)
        sk = np.abs(stats.skew(Xt, axis=0))
        rows.append({"transform": name,
                     "mean_abs_skew": round(float(np.nanmean(sk)), 3),
                     "max_abs_skew": round(float(np.nanmax(sk)), 3),
                     "per_feature": {c: round(float(s), 2) for c, s in zip(cols, sk)}})
    return rows


# --------------------------------------------------------------------------- #
# The neutral evaluation space
# --------------------------------------------------------------------------- #
REFERENCE_SPEC = dict(imputer="median", winsorize="none", transform="yeojohnson",
                      scaler="standard", reducer="none")
REFERENCE_VIEW = "raw17"


def reference_space(frame, seed: int = config.RANDOM_SEED):
    """The fixed space in which *every* candidate partition is scored.

    Rationale, and it is the single most consequential decision in this project.
    If a candidate is scored in the space it chose for itself, the search stops
    optimising cluster structure and starts optimising the *scoring space*:
    projecting to two principal components discards exactly the variance that
    would have made the clusters overlap, so the silhouette climbs while the
    partition gets no better. In an early run of this study the optimiser did
    precisely that - it selected ``pca_n2`` and reported a silhouette of 0.68 that
    was 0.195 when the same labels were scored against the real features.

    So the two roles are separated. A candidate may cluster in whatever space it
    likes; it is judged on the geometry its labels induce in a common space that
    no candidate can influence: all 17 source features, median-imputed,
    Yeo-Johnson transformed, standardised, no reduction. Fixed before the search,
    never tuned, identical for every configuration and for the final evaluation.
    """
    pre = build_preprocessor(REFERENCE_SPEC, seed=seed)
    X = np.asarray(pre.fit_transform(frame[feature_view(REFERENCE_VIEW)]), dtype=float)
    return X, pre
