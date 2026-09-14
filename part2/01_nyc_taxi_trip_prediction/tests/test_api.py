"""API contract tests, run against the real fitted bundle."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.main import app  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "bundle.joblib").exists(),
    reason="no trained model on disk — run `python -m taxi.train`",
)

TRIP = {
    "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,     # Times Square
    "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,   # JFK
    "pickup_datetime": "2016-06-15T18:00:00",
    "passenger_count": 2,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_a_loaded_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_a_sane_airport_estimate(client):
    body = client.post("/api/predict", json=TRIP).json()
    # Times Square to JFK at 6pm: anything outside 20-120 minutes is a bug.
    assert 20 * 60 < body["duration_seconds"] < 120 * 60
    assert body["low_seconds"] <= body["duration_seconds"] <= body["high_seconds"]
    assert body["grid_km"] >= body["straight_line_km"]
    assert body["eta"]
    assert any("Airport" in n for n in body["notes"])


def test_rush_hour_is_slower_than_the_dead_of_night(client):
    """A behavioural check, not a metric: the model must have learned the
    congestion clock, or the whole product is telling riders nothing."""
    evening = client.post("/api/predict", json={**TRIP, "pickup_datetime": "2016-06-15T18:00:00"}).json()
    overnight = client.post("/api/predict", json={**TRIP, "pickup_datetime": "2016-06-15T03:00:00"}).json()
    assert evening["duration_seconds"] > overnight["duration_seconds"]


def test_longer_trips_take_longer(client):
    short = client.post("/api/predict", json={
        **TRIP, "dropoff_latitude": 40.7527, "dropoff_longitude": -73.9772,   # Grand Central
    }).json()
    assert short["duration_seconds"] < client.post("/api/predict", json=TRIP).json()["duration_seconds"]


def test_coordinates_outside_nyc_are_rejected(client):
    res = client.post("/api/predict", json={**TRIP, "dropoff_latitude": 37.77,
                                            "dropoff_longitude": -122.42})   # San Francisco
    assert res.status_code == 422
    assert "service area" in res.text


def test_passenger_count_is_validated(client):
    assert client.post("/api/predict", json={**TRIP, "passenger_count": 0}).status_code == 422
    assert client.post("/api/predict", json={**TRIP, "passenger_count": 12}).status_code == 422


def test_by_hour_sweep_covers_the_whole_day(client):
    body = client.post("/api/predict/by-hour", json=TRIP).json()
    assert len(body["series"]) == 24
    assert [s["hour"] for s in body["series"]] == list(range(24))
    assert 0 <= body["best_hour"] <= 23
    assert body["spread_minutes"] > 0
    # The quietest hour of the day should be overnight, not the evening peak.
    assert body["best_hour"] in set(range(0, 7)) | {23}


def test_metrics_endpoint_beats_its_own_baselines(client):
    body = client.get("/api/metrics").json()
    results = {r["model"]: r for r in body["metrics"]["results"]}
    model = results["hist_gradient_boosting"]["rmsle"]
    assert all(model < r["rmsle"] for k, r in results.items() if k != "hist_gradient_boosting")
    assert body["feature_importance"]


def test_zones_endpoint_returns_cluster_centroids(client):
    clusters = client.get("/api/zones").json()["clusters"]
    assert len(clusters) == 40
    assert all(40.4 < c["lat"] < 41.0 and -74.4 < c["lon"] < -73.6 for c in clusters)


def test_frontend_is_served_from_the_same_process(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "NYC Taxi Trip Duration" in res.text
