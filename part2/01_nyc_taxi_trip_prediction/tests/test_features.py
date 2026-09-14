"""Feature-engineering tests: the geometry has to be right, and the transform
has to behave identically on one row and on a million."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxi.features import (  # noqa: E402
    FeatureBuilder,
    bearing_deg,
    build_frame,
    grid_distance_km,
    haversine_km,
)

TIMES_SQ = (40.7580, -73.9855)
JFK = (40.6413, -73.7781)


def _trip(**over):
    row = {
        "pickup_latitude": TIMES_SQ[0], "pickup_longitude": TIMES_SQ[1],
        "dropoff_latitude": JFK[0], "dropoff_longitude": JFK[1],
        "pickup_datetime": pd.Timestamp("2016-06-15 18:00:00"),
        "passenger_count": 1, "vendor_id": 1,
    }
    row.update(over)
    return pd.DataFrame([row])


# --------------------------------------------------------------------- geometry
def test_haversine_matches_known_distance():
    # Times Square to JFK is ~21 km as the crow flies.
    d = haversine_km(*TIMES_SQ, *JFK)
    assert 20.0 < d < 22.5


def test_haversine_is_zero_for_identical_points():
    assert haversine_km(40.75, -73.98, 40.75, -73.98) == pytest.approx(0.0, abs=1e-9)


def test_haversine_is_symmetric():
    assert haversine_km(*TIMES_SQ, *JFK) == pytest.approx(haversine_km(*JFK, *TIMES_SQ))


def test_grid_distance_is_never_shorter_than_great_circle():
    """An L-shaped path along the grid cannot beat a straight line."""
    rng = np.random.default_rng(0)
    lat1, lat2 = rng.uniform(40.6, 40.85, 500), rng.uniform(40.6, 40.85, 500)
    lon1, lon2 = rng.uniform(-74.02, -73.8, 500), rng.uniform(-74.02, -73.8, 500)
    grid = grid_distance_km(lat1, lon1, lat2, lon2)
    hav = haversine_km(lat1, lon1, lat2, lon2)
    assert np.all(grid >= hav - 1e-6)


def test_grid_distance_collapses_to_straight_line_along_the_grid():
    """A trip straight up an avenue has no cross-street leg, so grid == haversine."""
    theta = np.radians(29.0)
    lat0, lon0 = 40.75, -73.98
    dkm = 3.0
    dlat = dkm * np.sin(theta) / 111.32
    dlon = dkm * np.cos(theta) / (111.32 * np.cos(np.radians(lat0)))
    grid = grid_distance_km(lat0, lon0, lat0 + dlat, lon0 + dlon)
    hav = haversine_km(lat0, lon0, lat0 + dlat, lon0 + dlon)
    assert grid == pytest.approx(hav, rel=0.02)


def test_bearing_cardinal_directions():
    assert bearing_deg(40.70, -74.00, 40.80, -74.00) == pytest.approx(0.0, abs=0.5)     # north
    assert bearing_deg(40.75, -74.00, 40.75, -73.90) == pytest.approx(90.0, abs=0.5)    # east
    assert bearing_deg(40.80, -74.00, 40.70, -74.00) == pytest.approx(180.0, abs=0.5)   # south


# --------------------------------------------------------------------- transform
def test_build_frame_is_all_numeric_and_finite():
    f = build_frame(_trip())
    assert f.notna().all().all()
    assert np.isfinite(f.to_numpy()).all()
    assert (f.dtypes == np.float64).all()


def test_airport_flag_fires_for_a_jfk_trip():
    f = build_frame(_trip())
    assert f.loc[0, "is_jfk_trip"] == 1
    assert f.loc[0, "is_lga_trip"] == 0


def test_airport_flag_stays_off_for_a_crosstown_hop():
    f = build_frame(_trip(dropoff_latitude=40.7527, dropoff_longitude=-73.9772))
    assert f.loc[0, "is_jfk_trip"] == 0


def test_rush_hour_flags_respect_the_weekend():
    weekday = build_frame(_trip(pickup_datetime=pd.Timestamp("2016-06-15 18:00")))  # Wednesday
    weekend = build_frame(_trip(pickup_datetime=pd.Timestamp("2016-06-18 18:00")))  # Saturday
    assert weekday.loc[0, "is_pm_rush"] == 1
    assert weekend.loc[0, "is_pm_rush"] == 0
    assert weekend.loc[0, "is_weekend"] == 1


def test_cyclical_hour_encoding_wraps_around_midnight():
    a = build_frame(_trip(pickup_datetime=pd.Timestamp("2016-06-15 23:59")))
    b = build_frame(_trip(pickup_datetime=pd.Timestamp("2016-06-16 00:01")))
    gap = np.hypot(a.loc[0, "hour_sin"] - b.loc[0, "hour_sin"],
                   a.loc[0, "hour_cos"] - b.loc[0, "hour_cos"])
    assert gap < 0.02      # two minutes apart on the clock, and in feature space too


def test_passenger_count_is_clipped_and_defaulted():
    assert build_frame(_trip(passenger_count=99)).loc[0, "passenger_count"] == 6
    assert build_frame(_trip(passenger_count=None)).loc[0, "passenger_count"] == 1


# --------------------------------------------------------------------- builder
def _corpus(n=400):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "pickup_latitude": rng.uniform(40.65, 40.85, n),
        "pickup_longitude": rng.uniform(-74.02, -73.78, n),
        "dropoff_latitude": rng.uniform(40.65, 40.85, n),
        "dropoff_longitude": rng.uniform(-74.02, -73.78, n),
        "pickup_datetime": pd.date_range("2016-01-01", periods=n, freq="17min"),
        "passenger_count": rng.integers(1, 5, n),
        "vendor_id": rng.integers(1, 3, n),
    })


def test_builder_locks_column_order():
    b = FeatureBuilder(n_clusters=5).fit(_corpus())
    assert list(b.transform(_corpus(50)).columns) == b.columns


def test_builder_adds_cluster_features():
    b = FeatureBuilder(n_clusters=5).fit(_corpus())
    f = b.transform(_corpus(20))
    assert {"pickup_cluster", "dropoff_cluster", "same_cluster"} <= set(f.columns)
    assert f["pickup_cluster"].between(0, 4).all()
    assert len(b.cluster_centers()) == 5


def test_single_row_matches_the_batch_row():
    """Train/serve skew guard: serving one row must give the same numbers as
    scoring that row inside a batch."""
    b = FeatureBuilder(n_clusters=5).fit(_corpus())
    batch = _corpus(30)
    full = b.transform(batch)
    one = b.transform(batch.iloc[[11]])
    np.testing.assert_allclose(one.to_numpy()[0], full.to_numpy()[11], rtol=1e-12)
