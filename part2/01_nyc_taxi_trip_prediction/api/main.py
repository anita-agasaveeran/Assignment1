"""CRISP-DM Phase 6: the model, deployed.

    uvicorn api.main:app --reload

Serves the prediction API and the single-page front end from one process.  The
model bundle is loaded once at startup and shared; feature construction goes
through the exact ``FeatureBuilder`` that was fitted during training, so there
is no second implementation to drift.
"""

from __future__ import annotations

import os

# Must be set before numpy/scikit-learn initialise their OpenMP runtime.
#
# Serving is the opposite workload to training: one row at a time, not a
# million.  On a single row the cost of fanning 500 trees across every core is
# far larger than the work itself — measured at 76 ms/prediction on this
# machine, against 7.5 ms pinned to one thread.  Training still uses all cores;
# only the API process is pinned.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxi.config import LANDMARKS, MODELS, NYC_BBOX, WEB   # noqa: E402
from taxi.features import grid_distance_km, haversine_km   # noqa: E402
import taxi.model  # noqa: E402,F401  — makes taxi.model.Bundle importable for joblib

LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = NYC_BBOX

QUICK_PLACES = [
    {"name": "Times Square", "lat": 40.7580, "lon": -73.9855},
    {"name": "Grand Central", "lat": 40.7527, "lon": -73.9772},
    {"name": "Penn Station", "lat": 40.7506, "lon": -73.9935},
    {"name": "JFK Airport", "lat": 40.6413, "lon": -73.7781},
    {"name": "LaGuardia Airport", "lat": 40.7769, "lon": -73.8740},
    {"name": "Newark Airport", "lat": 40.6895, "lon": -74.1745},
    {"name": "Wall Street", "lat": 40.7061, "lon": -74.0090},
    {"name": "Brooklyn Bridge", "lat": 40.7061, "lon": -73.9969},
    {"name": "Central Park (S)", "lat": 40.7660, "lon": -73.9773},
    {"name": "Columbia University", "lat": 40.8075, "lon": -73.9626},
    {"name": "Williamsburg", "lat": 40.7081, "lon": -73.9571},
    {"name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262},
]


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
class TripRequest(BaseModel):
    pickup_latitude: float = Field(..., description="Pickup latitude, WGS84")
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    pickup_datetime: datetime | None = Field(
        None, description="Local NYC pickup time; defaults to now"
    )
    passenger_count: int = Field(1, ge=1, le=6)
    vendor_id: int = Field(1, ge=1, le=2)

    @model_validator(mode="after")
    def _inside_nyc(self):
        for label, lat, lon in (
            ("pickup", self.pickup_latitude, self.pickup_longitude),
            ("dropoff", self.dropoff_latitude, self.dropoff_longitude),
        ):
            if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                raise ValueError(
                    f"{label} ({lat:.4f}, {lon:.4f}) is outside the NYC service area — "
                    "the model was only ever shown trips inside it"
                )
        return self

    def to_frame(self) -> pd.DataFrame:
        ts = self.pickup_datetime or datetime.now()
        return pd.DataFrame([{
            "pickup_latitude": self.pickup_latitude,
            "pickup_longitude": self.pickup_longitude,
            "dropoff_latitude": self.dropoff_latitude,
            "dropoff_longitude": self.dropoff_longitude,
            "pickup_datetime": pd.Timestamp(ts.replace(tzinfo=None)),
            "passenger_count": self.passenger_count,
            "vendor_id": self.vendor_id,
        }])


# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #
class Registry:
    """Lazy singleton around the pickled bundle."""

    def __init__(self) -> None:
        self.bundle = None
        self.metrics: dict = {}
        self.importance: list = []
        self.error: str | None = None

    def load(self) -> None:
        try:
            self.bundle = joblib.load(MODELS / "bundle.joblib")
            self.metrics = json.loads((MODELS / "metrics.json").read_text())
            self.importance = json.loads((MODELS / "feature_importance.json").read_text())
            self.error = None
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def require(self):
        if self.bundle is None:
            raise HTTPException(
                503,
                detail=f"Model not loaded ({self.error or 'no bundle on disk'}). "
                       "Run `python -m taxi.train` first.",
            )
        return self.bundle


sklearn.set_config(assume_finite=True)   # features are built here, not user-supplied

registry = Registry()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the bundle once, at startup, rather than on the first request."""
    registry.load()
    print("Model loaded" if registry.bundle else f"Model NOT loaded: {registry.error}")
    yield


app = FastAPI(
    title="NYC Taxi Trip Duration",
    version="1.0.0",
    description="Trip-duration estimates from a gradient-boosted model trained on "
                "2016 NYC yellow-taxi records.",
    lifespan=lifespan,
)


def get_bundle():
    return registry.require()


Bundle = Annotated[object, Depends(get_bundle)]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _explain(row: pd.Series, ts: pd.Timestamp, seconds: float, grid_km: float) -> list[str]:
    """Plain-language context for the estimate.

    Not a SHAP explanation — the honest framing is "here is what about this
    trip pushes it away from a typical one", which is what a rider wants.
    """
    notes = []
    hour, dow = ts.hour, ts.dayofweek
    if dow < 5 and 16 <= hour < 20:
        notes.append("Weekday evening rush — the slowest window of the week.")
    elif dow < 5 and 7 <= hour < 10:
        notes.append("Weekday morning rush; expect stop-and-go through Midtown.")
    elif hour >= 22 or hour < 6:
        notes.append("Overnight — streets are clear and speeds run roughly double the PM peak.")

    for code, label in (("jfk", "JFK"), ("lga", "LaGuardia"), ("ewr", "Newark")):
        if row.get(f"is_{code}_trip", 0):
            notes.append(f"Airport trip ({label}) — highway routing, and terminal access adds time.")
            break

    kmh = grid_km / (seconds / 3600.0) if seconds > 0 else 0
    notes.append(f"Implied average speed {kmh:.1f} km/h over {grid_km:.1f} km of street grid.")
    if grid_km < 1.5:
        notes.append("Very short hop — at this range, signals and traffic dominate distance.")
    return notes


def _payload(bundle, req: TripRequest) -> dict:
    df = req.to_frame()
    ts = pd.Timestamp(df.loc[0, "pickup_datetime"])
    pred = bundle.predict(df)
    X = bundle.builder.transform(df)

    seconds = float(pred["seconds"][0])
    low = float(pred["low_seconds"][0])
    high = float(pred["high_seconds"][0])
    straight = float(haversine_km(req.pickup_latitude, req.pickup_longitude,
                                  req.dropoff_latitude, req.dropoff_longitude))
    grid = float(grid_distance_km(req.pickup_latitude, req.pickup_longitude,
                                  req.dropoff_latitude, req.dropoff_longitude))

    return {
        "duration_seconds": round(seconds, 1),
        "duration_minutes": round(seconds / 60.0, 1),
        "low_seconds": round(low, 1),
        "high_seconds": round(high, 1),
        "interval_pct": int(100 * (bundle.quantiles[1] - bundle.quantiles[0])),
        "eta": (ts + pd.to_timedelta(seconds, unit="s")).strftime("%H:%M"),
        "pickup_datetime": ts.isoformat(),
        "straight_line_km": round(straight, 3),
        "grid_km": round(grid, 3),
        "avg_speed_kmh": round(grid / (seconds / 3600.0), 1) if seconds > 0 else None,
        "notes": _explain(X.iloc[0], ts, seconds, grid),
    }


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if registry.bundle else "degraded",
        "model_loaded": registry.bundle is not None,
        "error": registry.error,
        "trained_at": getattr(registry.bundle, "trained_at", None),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/predict")
def predict(req: TripRequest, bundle: Bundle) -> dict:
    """Estimate one trip's duration, with a prediction interval."""
    return _payload(bundle, req)


@app.post("/api/predict/by-hour")
def predict_by_hour(req: TripRequest, bundle: Bundle) -> dict:
    """The same trip departing at each hour of that day.

    This is what turns a point estimate into a decision: riders do not want to
    know how long the trip takes, they want to know when to leave.
    """
    base = (req.pickup_datetime or datetime.now()).replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    )
    rows = []
    for h in range(24):
        r = req.model_copy(update={"pickup_datetime": base.replace(hour=h)})
        rows.append(r.to_frame())
    df = pd.concat(rows, ignore_index=True)
    pred = bundle.predict(df)

    series = [
        {"hour": h,
         "duration_seconds": round(float(pred["seconds"][h]), 1),
         "duration_minutes": round(float(pred["seconds"][h]) / 60.0, 1),
         "low_seconds": round(float(pred["low_seconds"][h]), 1),
         "high_seconds": round(float(pred["high_seconds"][h]), 1)}
        for h in range(24)
    ]
    best = min(series, key=lambda s: s["duration_seconds"])
    worst = max(series, key=lambda s: s["duration_seconds"])
    return {
        "date": base.date().isoformat(),
        "series": series,
        "best_hour": best["hour"],
        "worst_hour": worst["hour"],
        "spread_minutes": round(
            (worst["duration_seconds"] - best["duration_seconds"]) / 60.0, 1
        ),
    }


@app.get("/api/metrics")
def metrics() -> dict:
    """Held-out scores and permutation importances — the model's report card."""
    if not registry.metrics:
        raise HTTPException(503, detail="No metrics on disk; run `python -m taxi.train`.")
    return {"metrics": registry.metrics, "feature_importance": registry.importance[:18]}


@app.get("/api/zones")
def zones() -> dict:
    """KMeans centroids learned during training, drawn as zone markers."""
    bundle = registry.require()
    return {"clusters": bundle.builder.cluster_centers()}


@app.get("/api/places")
def places() -> dict:
    return {"places": QUICK_PLACES, "landmarks": LANDMARKS}


@app.get("/api/stats")
def stats() -> JSONResponse:
    path = WEB / "data" / "stats.json"
    if not path.exists():
        raise HTTPException(503, detail="No stats on disk; run `python -m taxi.eda`.")
    return JSONResponse(json.loads(path.read_text()))


# --------------------------------------------------------------------------- #
# static front end (mounted last so /api/* wins)
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
