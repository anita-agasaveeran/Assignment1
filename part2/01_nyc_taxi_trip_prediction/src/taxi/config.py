"""Single source of truth for paths, constants and modelling knobs."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "docs" / "figures"
WEB = ROOT / "web"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
# The Kaggle "New York City Taxi Trip Duration" competition ships a 1.46M-row
# sample of NYC TLC yellow-taxi records from 2016-01..2016-06, with raw pickup
# and dropoff coordinates.  Two notes on provenance:
#
#   * Kaggle itself needs credentials, so this project rebuilds an equivalent
#     dataset from the primary public source instead.
#   * The TLC's current parquet mirror re-published 2016 with zone IDs and
#     *dropped the coordinates*, which would sink a map-driven product.  NYC
#     Open Data still serves the original coordinate-level 2016 records.
#
# So: same underlying trips, same columns as the Kaggle CSV, no credentials.
SOCRATA_DATASET = "uacg-pexx"          # "2016 Yellow Taxi Trip Data"
SOCRATA_URL = f"https://data.cityofnewyork.us/resource/{SOCRATA_DATASET}.csv"
MONTHS = ("2016-01", "2016-02", "2016-03", "2016-04", "2016-05", "2016-06")
PAGE_SIZE = 50_000
PAGES_PER_MONTH = int(os.environ.get("TAXI_PAGES_PER_MONTH", 4))   # 200k rows/month

# --------------------------------------------------------------------------- #
# Cleaning thresholds (rationale in docs/crisp-dm.md, Phase 3)
# --------------------------------------------------------------------------- #
MIN_DURATION_S = 60           # sub-minute trips are meter noise, not journeys
MAX_DURATION_S = 3 * 3600     # 3h is beyond p99.99; above it is meter-left-running
MAX_DISTANCE_KM = 100.0
NYC_BBOX = (-74.30, -73.65, 40.49, 40.93)   # lon_min, lon_max, lat_min, lat_max

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
SEED = 255
TEST_SIZE = 0.2
N_CLUSTERS = 40               # KMeans zones learned from pickup+dropoff points
QUANTILES = (0.1, 0.9)        # prediction interval surfaced in the UI

# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
# NYC's street grid runs ~29 degrees off true north; rotating into that frame
# before taking an L1 norm makes "Manhattan distance" follow actual avenues.
NYC_GRID_ROTATION_DEG = 29.0

LANDMARKS = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
    "midtown": (40.7549, -73.9840),
    "downtown": (40.7075, -74.0113),
}
AIRPORT_RADIUS_KM = 2.0
