"""The serialisable model bundle.

This lives apart from ``train.py`` on purpose.  Training runs as
``python -m taxi.train``, which makes that module ``__main__``; anything
pickled from there records its class as ``__main__.Bundle`` and cannot be
unpickled by the API process.  Defining ``Bundle`` here gives it one stable
import path — ``taxi.model.Bundle`` — in every process that touches it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FeatureBuilder


@dataclass
class Bundle:
    """Everything the API needs to serve a prediction, in one artefact."""

    builder: FeatureBuilder
    point: object          # HistGradientBoostingRegressor on log1p(duration)
    lo: object             # ...same, fitted to the lower quantile
    hi: object
    quantiles: tuple[float, float]
    feature_names: list[str]
    trained_at: str
    n_train: int

    def predict(self, df: pd.DataFrame) -> dict:
        """Point estimate plus a prediction interval, in seconds.

        All three models are fitted on log1p(duration), so predictions come
        back through expm1.  Keeping the quantile models in log space is what
        makes the interval multiplicative — roughly 20% either side of a
        40-minute run, rather than a flat 8 minutes bolted onto every trip.
        """
        X = self.builder.transform(df)
        centre = np.maximum(np.expm1(self.point.predict(X)), 30.0)
        lo = np.expm1(self.lo.predict(X))
        hi = np.expm1(self.hi.predict(X))
        # Quantile models are fitted independently and can cross on odd inputs;
        # clamping here keeps the API from ever returning low > estimate.
        return {
            "seconds": centre,
            "low_seconds": np.minimum(np.maximum(lo, 20.0), centre),
            "high_seconds": np.maximum(hi, centre),
        }
