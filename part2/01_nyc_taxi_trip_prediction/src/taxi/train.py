"""CRISP-DM Phase 4/5: fit the models and score them honestly.

    python -m taxi.train

Produces, in ``models/``:
  bundle.joblib   fitted FeatureBuilder + point model + two quantile models
  metrics.json    baselines vs. the model, on a held-out *future* period
  feature_importance.json
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from .config import DATA_PROCESSED, MODELS, QUANTILES, SEED, TEST_SIZE
from .features import FeatureBuilder
from .model import Bundle

TARGET = "trip_duration"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rmsle(y_true, y_pred) -> float:
    """The Kaggle competition metric: RMSE on log1p of the duration.

    Log space is the right call for a target this skewed — being 5 minutes off
    on a 90-minute airport run is a very different error from being 5 minutes
    off on a 6-minute crosstown hop, and RMSLE prices them accordingly.
    """
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1.0)
    y_true = np.maximum(np.asarray(y_true, dtype=np.float64), 1.0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def score(name: str, y_true, y_pred) -> dict:
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1.0)
    err = y_pred - np.asarray(y_true, dtype=np.float64)
    return {
        "model": name,
        "rmsle": rmsle(y_true, y_pred),
        "mae_seconds": float(mean_absolute_error(y_true, y_pred)),
        "median_abs_error_seconds": float(np.median(np.abs(err))),
        "rmse_seconds": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "within_2min_pct": float(100.0 * np.mean(np.abs(err) <= 120)),
        "within_5min_pct": float(100.0 * np.mean(np.abs(err) <= 300)),
    }


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def temporal_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Hold out the most recent slice, not a random one.

    A random split leaks: trips from the same rush hour land on both sides of
    the split, and the model gets credit for memorising a specific evening.
    Training on the past and scoring on the future is what deployment actually
    looks like, so that is what we measure.
    """
    df = df.sort_values("pickup_datetime")
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _hgb(**kw) -> HistGradientBoostingRegressor:
    params = dict(
        max_iter=500,
        learning_rate=0.08,
        max_depth=None,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=SEED,
    )
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def main() -> None:
    src = DATA_PROCESSED / "trips.parquet"
    if not src.exists():
        raise SystemExit(f"{src} missing — run `python -m taxi.ingest` first.")
    df = pd.read_parquet(src)
    print(f"Loaded {len(df):,} clean trips "
          f"({df['pickup_datetime'].min():%Y-%m-%d} .. {df['pickup_datetime'].max():%Y-%m-%d})")

    train_df, test_df = temporal_split(df)
    print(f"Temporal split: train={len(train_df):,} (through "
          f"{train_df['pickup_datetime'].max():%Y-%m-%d}), test={len(test_df):,}")

    t0 = time.time()
    builder = FeatureBuilder()
    X_train = builder.fit(train_df).transform(train_df)
    X_test = builder.transform(test_df)
    y_train = train_df[TARGET].to_numpy(dtype=np.float64)
    y_test = test_df[TARGET].to_numpy(dtype=np.float64)
    print(f"Features: {X_train.shape[1]} columns in {time.time() - t0:.1f}s")

    log_y = np.log1p(y_train)
    results = []

    # -- Baseline 1: the number a dispatcher would guess ---------------------
    results.append(score("baseline_global_median", y_test,
                         np.full(len(y_test), float(np.median(y_train)))))

    # -- Baseline 2: distance / typical speed --------------------------------
    # The obvious non-ML rule, and a genuinely strong one. Anything we build
    # has to beat it or it is not worth deploying.
    kmh = np.median(X_train["haversine_km"] / (y_train / 3600.0))
    results.append(score(f"baseline_constant_speed_{kmh:.1f}kmh", y_test,
                         X_test["haversine_km"] / kmh * 3600.0))

    # -- Baseline 3: linear model in log space -------------------------------
    ridge_cols = ["haversine_km", "grid_km", "hour", "day_of_week", "is_weekend",
                  "is_am_rush", "is_pm_rush", "is_night"]
    ridge = Ridge(alpha=1.0).fit(X_train[ridge_cols], log_y)
    results.append(score("ridge_log", y_test, np.expm1(ridge.predict(X_test[ridge_cols]))))

    # -- The model -----------------------------------------------------------
    print("Fitting gradient-boosted trees...")
    t0 = time.time()
    point = _hgb().fit(X_train, log_y)
    print(f"  point model: {point.n_iter_} iterations, {time.time() - t0:.1f}s")
    results.append(score("hist_gradient_boosting", y_test, np.expm1(point.predict(X_test))))

    # -- Prediction interval -------------------------------------------------
    qmodels = {}
    for q in QUANTILES:
        t0 = time.time()
        qmodels[q] = _hgb(loss="quantile", quantile=q, max_iter=300).fit(X_train, log_y)
        print(f"  q{q:.2f} model: {time.time() - t0:.1f}s")

    lo = np.expm1(qmodels[QUANTILES[0]].predict(X_test))
    hi = np.expm1(qmodels[QUANTILES[1]].predict(X_test))
    coverage = float(100.0 * np.mean((y_test >= lo) & (y_test <= hi)))
    nominal = 100.0 * (QUANTILES[1] - QUANTILES[0])
    print(f"  interval coverage: {coverage:.1f}% (nominal {nominal:.0f}%)")

    # -- What the model leans on --------------------------------------------
    # Permutation importance on a subsample: honest about correlated features
    # in a way that split-count importance is not.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_test), min(20_000, len(X_test)), replace=False)
    perm = permutation_importance(
        point, X_test.iloc[idx], np.log1p(y_test[idx]),
        n_repeats=3, random_state=SEED, n_jobs=-1,
    )
    importance = sorted(
        ({"feature": f, "importance": float(m)}
         for f, m in zip(X_train.columns, perm.importances_mean)),
        key=lambda d: -d["importance"],
    )

    bundle = Bundle(
        builder=builder, point=point,
        lo=qmodels[QUANTILES[0]], hi=qmodels[QUANTILES[1]],
        quantiles=QUANTILES, feature_names=list(X_train.columns),
        trained_at=pd.Timestamp.now("UTC").isoformat(), n_train=len(train_df),
    )
    joblib.dump(bundle, MODELS / "bundle.joblib", compress=3)

    metrics = {
        "trained_at": bundle.trained_at,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_period": [str(train_df["pickup_datetime"].min()), str(train_df["pickup_datetime"].max())],
        "test_period": [str(test_df["pickup_datetime"].min()), str(test_df["pickup_datetime"].max())],
        "split": "temporal (most recent 20% held out)",
        "primary_metric": "rmsle",
        "results": results,
        "interval": {"quantiles": list(QUANTILES), "nominal_coverage_pct": nominal,
                     "empirical_coverage_pct": coverage},
        "n_features": X_train.shape[1],
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS / "feature_importance.json").write_text(json.dumps(importance, indent=2))

    print("\n%-34s %8s %10s %8s" % ("model", "RMSLE", "MAE(s)", "R2"))
    for r in results:
        print("%-34s %8.4f %10.1f %8.3f" % (r["model"], r["rmsle"], r["mae_seconds"], r["r2"]))
    print(f"\nSaved bundle + metrics to {MODELS}")


if __name__ == "__main__":
    main()
