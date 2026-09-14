"""Feature engineering shared by training and serving.

Every transform lives here so the API cannot drift from the notebook: the
training script and the FastAPI process import the *same* functions and the
same fitted ``FeatureBuilder``.  ``build_frame`` is the single entry point and
it works on one row or on ten million.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from .config import (
    AIRPORT_RADIUS_KM,
    LANDMARKS,
    N_CLUSTERS,
    NYC_GRID_ROTATION_DEG,
    SEED,
)

EARTH_R_KM = 6371.0088


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; vectorised over arrays or scalars."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64))
                              for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing from origin to destination, in degrees."""
    lat1r, lat2r = np.radians(np.asarray(lat1, dtype=np.float64)), np.radians(np.asarray(lat2, dtype=np.float64))
    dlon = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def grid_distance_km(lat1, lon1, lat2, lon2, rotation_deg=NYC_GRID_ROTATION_DEG):
    """L1 distance measured along NYC's rotated avenue/street grid.

    A cab cannot fly the great circle: it drives up an avenue and across a
    street.  Rotating into the grid frame before taking the L1 norm tracks the
    driven path much better than either raw haversine or an unrotated L1.
    """
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    theta = np.radians(rotation_deg)
    lat_mid = np.radians((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * (np.pi / 180.0) * EARTH_R_KM
    dx = (lon2 - lon1) * (np.pi / 180.0) * EARTH_R_KM * np.cos(lat_mid)
    along = np.abs(dx * np.cos(theta) + dy * np.sin(theta))
    across = np.abs(-dx * np.sin(theta) + dy * np.cos(theta))
    return along + across


# --------------------------------------------------------------------------- #
# fitted state
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBuilder:
    """Holds the part of feature engineering that must be *learned* from the
    training split — the spatial clusters — so it can be replayed at serve
    time.  Pickled next to the model and reloaded by the API."""

    n_clusters: int = N_CLUSTERS
    kmeans: MiniBatchKMeans | None = None
    columns: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        pts = np.vstack(
            [
                df[["pickup_latitude", "pickup_longitude"]].to_numpy(dtype=np.float64),
                df[["dropoff_latitude", "dropoff_longitude"]].to_numpy(dtype=np.float64),
            ]
        )
        if len(pts) > 500_000:                    # clustering needs breadth, not bulk
            rng = np.random.default_rng(SEED)
            pts = pts[rng.choice(len(pts), 500_000, replace=False)]
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=SEED,
            n_init=10,
            batch_size=4096,
        ).fit(pts)
        self.columns = list(build_frame(df.head(64), self.kmeans).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = build_frame(df, self.kmeans)
        if self.columns:                          # lock column order for the model
            out = out.reindex(columns=self.columns)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def cluster_centers(self) -> list[dict]:
        """Centroids as lat/lon dicts — the front end draws these as zones."""
        if self.kmeans is None:
            return []
        return [
            {"cluster": i, "lat": float(c[0]), "lon": float(c[1])}
            for i, c in enumerate(self.kmeans.cluster_centers_)
        ]


# --------------------------------------------------------------------------- #
# the transform itself
# --------------------------------------------------------------------------- #
def build_frame(df: pd.DataFrame, kmeans: MiniBatchKMeans | None = None) -> pd.DataFrame:
    """Turn raw trip rows into the model matrix.

    Expects ``pickup_datetime``, the four coordinate columns, ``passenger_count``
    and ``vendor_id``.  Returns numeric columns only — the estimator is a
    gradient-boosted tree, so nothing needs scaling or one-hot encoding.
    """
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64)
    plon = df["pickup_longitude"].to_numpy(dtype=np.float64)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(df["pickup_datetime"])

    f = pd.DataFrame(index=df.index)

    # --- where -------------------------------------------------------------
    f["pickup_latitude"] = plat
    f["pickup_longitude"] = plon
    f["dropoff_latitude"] = dlat
    f["dropoff_longitude"] = dlon
    f["haversine_km"] = haversine_km(plat, plon, dlat, dlon)
    f["grid_km"] = grid_distance_km(plat, plon, dlat, dlon)
    f["bearing"] = bearing_deg(plat, plon, dlat, dlon)
    f["center_latitude"] = (plat + dlat) / 2.0
    f["center_longitude"] = (plon + dlon) / 2.0
    # Detour ratio: how much further the grid path runs than the crow flight.
    f["grid_over_haversine"] = f["grid_km"] / np.maximum(f["haversine_km"], 0.05)

    # --- when --------------------------------------------------------------
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    minute_of_day = hour * 60 + ts.dt.minute.to_numpy()
    f["hour"] = hour
    f["day_of_week"] = dow
    f["month"] = ts.dt.month.to_numpy()
    f["day_of_year"] = ts.dt.dayofyear.to_numpy()
    f["minute_of_day"] = minute_of_day
    f["is_weekend"] = (dow >= 5).astype(np.int8)
    # Cyclical encodings so 23:59 and 00:01 sit next to each other.
    f["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    f["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    # Rush-hour flags, weekday-only — the evening peak is the punishing one.
    weekday = dow < 5
    f["is_am_rush"] = (weekday & (hour >= 7) & (hour < 10)).astype(np.int8)
    f["is_pm_rush"] = (weekday & (hour >= 16) & (hour < 20)).astype(np.int8)
    f["is_night"] = ((hour >= 22) | (hour < 6)).astype(np.int8)

    # --- who ---------------------------------------------------------------
    f["passenger_count"] = (
        pd.to_numeric(df["passenger_count"], errors="coerce").fillna(1).clip(0, 6).to_numpy()
    )
    vendor = df["vendor_id"] if "vendor_id" in df.columns else 1
    f["vendor_id"] = pd.to_numeric(pd.Series(vendor, index=df.index), errors="coerce").fillna(1).to_numpy()

    # --- landmarks ---------------------------------------------------------
    # Airport runs behave unlike anything else: flat fares, highway routing,
    # terminal queues.  Distance-to-landmark lets the trees isolate them.
    for name, (lat, lon) in LANDMARKS.items():
        dp = haversine_km(plat, plon, lat, lon)
        dd = haversine_km(dlat, dlon, lat, lon)
        f[f"pickup_dist_{name}"] = dp
        f[f"dropoff_dist_{name}"] = dd
        if name in ("jfk", "lga", "ewr"):
            f[f"is_{name}_trip"] = (
                (dp < AIRPORT_RADIUS_KM) | (dd < AIRPORT_RADIUS_KM)
            ).astype(np.int8)

    # --- learned zones -----------------------------------------------------
    if kmeans is not None:
        f["pickup_cluster"] = kmeans.predict(np.column_stack([plat, plon]))
        f["dropoff_cluster"] = kmeans.predict(np.column_stack([dlat, dlon]))
        f["same_cluster"] = (f["pickup_cluster"] == f["dropoff_cluster"]).astype(np.int8)

    return f.astype(np.float64)
