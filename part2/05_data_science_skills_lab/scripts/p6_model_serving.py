"""CRISP-DM Phase 6 - Deployment (Track A).

Skill demonstrated:
  * model-serving -- FastAPI service over the *whole pipeline*, Pydantic validation,
                     model loaded once at startup, health/readiness, versioned
                     responses, batching, latency measurement, drift monitoring hook.

This module both defines the app and (when run directly) exercises it end to end
with FastAPI's TestClient, so the demonstration is a real request/response cycle.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent))
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common import MOD, PROC, banner, write_json

ARTIFACT = MOD / "titanic_pipeline.joblib"

# ---- Load the artifact ONCE at import/startup, never per request -------------
_bundle = joblib.load(ARTIFACT)
MODEL = _bundle["pipeline"]          # preprocessing + estimator travel together
THRESHOLD = _bundle["threshold"]     # the threshold chosen on validation in Phase 4
VERSION = _bundle["model_version"]
FEATURES = _bundle["features"]
# Training-time feature summary, used as the drift reference.
_train = pd.read_csv(PROC / "titanic_train.csv").drop(columns=["Survived"])
REFERENCE = {c: (float(_train[c].mean()), float(_train[c].std() or 1.0))
             for c in _bundle["num"]}

app = FastAPI(title="Titanic survival scorer", version=VERSION)


class Passenger(BaseModel):
    """Request schema. Pydantic returns 422 on a malformed payload before it
    ever reaches the model, so one bad request cannot take down the worker."""
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float = Field(ge=0, le=120)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float = Field(ge=0, le=1000)
    Embarked: Literal["C", "Q", "S"]
    Title: str = "Mr"
    Deck: str = "U"
    Age_was_missing: int = 0
    CabinKnown: int = 0


def _to_frame(items: list[Passenger]) -> pd.DataFrame:
    df = pd.DataFrame([i.model_dump() for i in items])
    # Derive the engineered columns exactly as Phase 3 did. Any divergence here
    # is training/serving skew -- the single most common production ML failure.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["FareLog"] = np.log1p(df["Fare"])
    df["AgeBand"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                           labels=[0, 1, 2, 3, 4]).astype(int)
    df["TicketPrefix_te"] = 0.3834      # train prior for an unseen prefix
    return df.reindex(columns=FEATURES)


def _drift(df: pd.DataFrame) -> dict:
    """ML-specific monitoring: how far is this request from the training distribution?"""
    out = {}
    for c, (mu, sd) in REFERENCE.items():
        if c in df:
            out[c] = round(float(abs(df[c].mean() - mu) / sd), 3)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_version": VERSION}


@app.get("/ready")
def ready():
    return {"ready": MODEL is not None, "model_version": VERSION,
            "threshold": round(THRESHOLD, 4), "n_features": len(FEATURES)}


@app.post("/predict")
def predict(p: Passenger):
    df = _to_frame([p])
    proba = float(MODEL.predict_proba(df)[0, 1])
    return {"probability": round(proba, 4),
            "label": int(proba >= THRESHOLD),
            "threshold": round(THRESHOLD, 4),
            "model_version": VERSION}


@app.post("/predict/batch")
def predict_batch(items: list[Passenger]):
    """Batching raises throughput: one vectorised transform+predict for N rows."""
    if not items:
        raise HTTPException(422, "empty batch")
    if len(items) > 1000:
        raise HTTPException(413, "batch too large (max 1000)")
    df = _to_frame(items)
    proba = MODEL.predict_proba(df)[:, 1]
    return {"n": len(items),
            "predictions": [{"probability": round(float(x), 4), "label": int(x >= THRESHOLD)}
                            for x in proba],
            "drift_zscores": _drift(df),
            "model_version": VERSION}


def demo() -> None:
    from fastapi.testclient import TestClient

    banner("model-serving", "Phase 6 - Deployment",
           "exercising the service end to end with TestClient")
    client = TestClient(app)
    results: dict = {}

    print("  GET /health ->", client.get("/health").json())
    print("  GET /ready  ->", client.get("/ready").json())

    print("\n  POST /predict")
    cases = [
        ("1st-class woman", {"Pclass": 1, "Sex": "female", "Age": 30, "SibSp": 0, "Parch": 0,
                             "Fare": 100.0, "Embarked": "C", "Title": "Mrs", "Deck": "C",
                             "CabinKnown": 1}),
        ("3rd-class man", {"Pclass": 3, "Sex": "male", "Age": 30, "SibSp": 0, "Parch": 0,
                           "Fare": 7.9, "Embarked": "S", "Title": "Mr", "Deck": "U"}),
        ("child with family", {"Pclass": 2, "Sex": "female", "Age": 6, "SibSp": 1, "Parch": 2,
                               "Fare": 27.0, "Embarked": "S", "Title": "Miss", "Deck": "U"}),
    ]
    preds = {}
    for label, body in cases:
        r = client.post("/predict", json=body).json()
        preds[label] = r
        print(f"     {label:<20} p(survive) = {r['probability']:.4f} -> "
              f"label {r['label']}  (v{r['model_version']})")

    print("\n  input validation (the skill's 'return 422 on bad payloads'):")
    for label, bad in [("Age = -5", {**cases[0][1], "Age": -5}),
                       ("Pclass = 9", {**cases[0][1], "Pclass": 9}),
                       ("missing Fare", {k: v for k, v in cases[0][1].items() if k != "Fare"})]:
        code = client.post("/predict", json=bad).status_code
        print(f"     {label:<16} -> HTTP {code} {'(rejected before the model)' if code == 422 else ''}")
        results.setdefault("validation", {})[label] = code

    print("\n  latency (model loaded once at startup, not per request):")
    single = cases[1][1]
    client.post("/predict", json=single)                       # warm
    t = [(lambda t0: (client.post("/predict", json=single), time.perf_counter() - t0)[1])
         (time.perf_counter()) for _ in range(50)]
    p50, p95 = np.percentile(t, 50) * 1000, np.percentile(t, 95) * 1000
    print(f"     single  p50 {p50:6.2f} ms   p95 {p95:6.2f} ms   (n=50)")

    batch = [single] * 200
    t0 = time.perf_counter()
    rb = client.post("/predict/batch", json=batch).json()
    bt = (time.perf_counter() - t0) * 1000
    print(f"     batch   200 rows in {bt:.2f} ms = {bt/200:.3f} ms/row "
          f"({p50/(bt/200):.0f}x cheaper per row than one-at-a-time)")

    print(f"\n  drift monitoring on the batch (|mean - train mean| / train sd):")
    for k, v in sorted(rb["drift_zscores"].items(), key=lambda x: -x[1])[:5]:
        flag = "  ** DRIFT **" if v > 2 else ""
        print(f"     {k:<16} z = {v:>6.3f}{flag}")
    mx = max(rb["drift_zscores"].values())
    print(f"     -> max |z| = {mx:.2f}; the alert threshold is 2.0, so this batch does NOT trip it.")
    print("        A batch of 200 identical 3rd-class men shifts the mean but stays inside one")
    print("        training standard deviation, because 3rd class is the modal class. The point")
    print("        is that the signal is computed and returned on every batch, so a genuine")
    print("        distribution shift would be visible before the metrics decay.")

    print("\n  production checklist:")
    for item, state in [
        ("model + preprocessing ship as ONE artifact", "yes - joblib holds the full Pipeline"),
        ("loaded once at startup", "yes - module-level load, not per request"),
        ("input validation", "yes - Pydantic, 422 before inference"),
        ("health / readiness endpoints", "yes - /health and /ready"),
        ("model version in every response", f"yes - v{VERSION}"),
        ("batch endpoint", "yes - /predict/batch, capped at 1000"),
        ("drift monitoring", "yes - per-feature z-scores returned with batches"),
        ("artifact provenance", f"yes - git {_bundle['git_sha'][:8]}, data {_bundle['data_sha256']}"),
        ("untrusted artifact loading", "N/A - we produced this artifact ourselves"),
    ]:
        print(f"     [x] {item:<44} {state}")

    print("\n  run for real:  uvicorn scripts.p6_model_serving:app --host 0.0.0.0 --port 8000 --workers 4")

    results.update({"predictions": preds, "latency_p50_ms": float(p50), "latency_p95_ms": float(p95),
                    "batch_ms_per_row": float(bt / 200), "drift": rb["drift_zscores"],
                    "model_version": VERSION, "threshold": THRESHOLD})
    write_json(results, "p6_serving_results.json")


if __name__ == "__main__":
    demo()
