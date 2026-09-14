"""Does the grid rotation earn its place?

Permutation importance ranks `haversine_km` first and leaves `grid_km` out of
the top ten, which invites the wrong conclusion — that the rotated-grid feature
was wasted work.  Permutation importance describes what the *fitted* model
happens to lean on, not what the feature set makes possible: with two features
correlated at r = 0.99, the trees commit to one and the other looks idle.

The only way to settle it is to refit without each one.

    python scripts/ablation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxi.config import DATA_PROCESSED  # noqa: E402
from taxi.features import FeatureBuilder  # noqa: E402
from taxi.train import _hgb, rmsle, temporal_split  # noqa: E402

VARIANTS = {
    "all features": [],
    "drop grid_km": ["grid_km", "grid_over_haversine"],
    "drop haversine_km": ["haversine_km", "grid_over_haversine"],
    "drop both distances": ["haversine_km", "grid_km", "grid_over_haversine"],
}


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "trips.parquet")
    train_df, test_df = temporal_split(df)
    builder = FeatureBuilder().fit(train_df)
    X_train, X_test = builder.transform(train_df), builder.transform(test_df)
    y_train = np.log1p(train_df["trip_duration"].to_numpy())
    y_test = test_df["trip_duration"].to_numpy()

    r = float(np.corrcoef(X_train["haversine_km"], X_train["grid_km"])[0, 1])
    print(f"corr(haversine_km, grid_km) = {r:.4f}\n")
    print(f"{'variant':24s} {'RMSLE':>8s} {'delta':>8s}")

    base = None
    for name, drop in VARIANTS.items():
        cols = [c for c in X_train.columns if c not in drop]
        t0 = time.time()
        model = _hgb(max_iter=300).fit(X_train[cols], y_train)
        score = rmsle(y_test, np.expm1(model.predict(X_test[cols])))
        base = score if base is None else base
        print(f"{name:24s} {score:8.4f} {score - base:+8.4f}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
