# 🚕 NYC Taxi Trip Duration — end to end

Predict how long a New York yellow-cab trip will take, from a gradient-boosted
model trained on **1.16 million real 2016 taxi records** — with a live map, an
hour-by-hour "when should I leave" sweep, and the full CRISP-DM trail behind it.

Built on the Kaggle *New York City Taxi Trip Duration* problem, reproduced from
public data with no credentials required.

![RMSLE](https://img.shields.io/badge/RMSLE-0.3065-f7b500)
![MAE](https://img.shields.io/badge/MAE-3.1%20min-4c8dff)
![tests](https://img.shields.io/badge/tests-25%20passing-34d399)

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make data     # download + clean ~1.2M trips  (~4 min, cached after the first run)
make train    # fit the model + two quantile models  (~2 min)
make eda      # dashboard stats + report figures
make serve    # http://127.0.0.1:8000
```

Or all four at once:

```bash
make all && make serve
```

---

## What you get

**An interactive map.** Click to drop a pickup, click again for a dropoff, drag
either to adjust — or pick from twelve NYC landmarks. The route is drawn as a
staircase along Manhattan's 29°-rotated street grid, which is a faithful picture
of the distance feature the model actually consumes rather than an invented
turn-by-turn route.

**An estimate that admits what it doesn't know.** Every prediction comes with a
p10–p90 band: *"70 minutes, and 80% of trips like this take 54–90."* Two extra
quantile models produce the band, fitted in log space so it scales with the trip
instead of being a flat ±8 minutes on everything.

**The answer to the question riders actually ask.** The *When to go* panel
re-runs the same trip at all 24 departure hours. Times Square → JFK swings from
**29 minutes at 5am to 77 minutes at 4pm** — a 48-minute spread on identical
geometry. Knowing the duration is useful; knowing when to leave is the product.

**Two analysis layers over the city.** Toggle 12,000 sampled pickups (they
redraw the street grid and light up both airports) and the 40 KMeans zone
centroids the model learned, so the geography it uses is inspectable rather
than implicit.

**Its own report card.** The *Data* and *Model* tabs serve the live cleaning
audit, the held-out scores against three baselines, permutation importances, and
the interval's measured calibration — read from the same JSON the training run
wrote.

---

## Results

Scored on 232,968 held-out trips from the most recent period — trained on the
past, tested on the future.

| Model | RMSLE | MAE | Median err. | R² | Within 5 min |
|---|---:|---:|---:|---:|---:|
| Global median | 0.7427 | 7.9 min | 5.2 min | −0.103 | 47.8% |
| Constant speed (12.9 km/h) | 0.5309 | 6.8 min | 3.4 min | −0.034 | 62.2% |
| Ridge (log space) | 0.5436 | 6.4 min | 3.6 min | −0.177 | 64.4% |
| **Gradient-boosted trees** | **0.3065** | **3.1 min** | **1.9 min** | **0.816** | **83.4%** |

42% below the strongest baseline's RMSLE, with less than half its MAE.
Predictions take **7.5 ms** (≈21 ms end-to-end over HTTP).

---

## How it works

```
NYC Open Data  ──▶  ingest.py   ──▶  1.16M clean trips  (97.1% retained, audited)
                                          │
                                          ▼
                    features.py  ──▶  41 features   ← shared by training AND serving
                                          │
                                          ▼
                     train.py    ──▶  bundle.joblib  (point + p10 + p90)
                                          │
                                          ▼
                    api/main.py  ──▶  FastAPI  ──▶  web/  (Leaflet SPA)
```

**Data.** The Kaggle CSV needs credentials, and the TLC's current parquet mirror
re-published 2016 with zone IDs and *dropped the coordinates* — fatal for a
map-driven product. NYC Open Data still serves the original coordinate-level
2016 records, so `ingest.py` pulls from there and reshapes to the Kaggle schema.
Pages are stratified across all six months; an unstratified offset walk over
that API returns overwhelmingly February.

**Features.** Rotated-grid distance (NYC's grid runs 29° off north, so an L1
norm taken *after* rotating tracks driven distance), cyclical time encodings so
23:59 neighbours 00:01, weekday-gated rush-hour flags, airport proximity, and
40 learned KMeans zones.

**No train/serve skew.** The API imports the same `FeatureBuilder` that training
fitted, and the fitted KMeans travels inside the bundle. There is no second
implementation of the feature logic to drift — the usual way an accurate
notebook becomes an inaccurate service. A test pins it: one row scored alone
must equal that row scored inside a batch.

**Temporal split.** Train through 2016-05-25, test on everything after. A random
split leaks — trips from the same rush hour land on both sides, and the model
gets credit for memorising a specific evening.

---

## API

| Route | Purpose |
|---|---|
| `POST /api/predict` | Duration, interval, ETA, distances, plain-language notes |
| `POST /api/predict/by-hour` | The same trip at all 24 departure hours |
| `GET /api/metrics` | Held-out scores and permutation importances |
| `GET /api/zones` | The 40 learned KMeans centroids |
| `GET /api/stats` | Dataset aggregates for the dashboard |
| `GET /api/health` | Liveness plus model-load status |
| `GET /docs` | Generated OpenAPI documentation |

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"pickup_latitude":40.7580,"pickup_longitude":-73.9855,
       "dropoff_latitude":40.6413,"dropoff_longitude":-73.7781,
       "pickup_datetime":"2016-06-15T18:00:00","passenger_count":2}'
```

```json
{
  "duration_minutes": 69.5,
  "low_seconds": 3107.5,
  "high_seconds": 5427.1,
  "eta": "19:09",
  "grid_km": 28.827,
  "avg_speed_kmh": 24.9,
  "notes": [
    "Weekday evening rush — the slowest window of the week.",
    "Airport trip (JFK) — highway routing, and terminal access adds time."
  ]
}
```

Coordinates outside NYC are rejected with an explanation rather than silently
extrapolated — a model asked about San Francisco should decline, not guess.

---

## Layout

```
├── src/taxi/
│   ├── config.py      paths, thresholds, hyperparameters
│   ├── ingest.py      Phase 2/3 — download, reshape, clean (audited)
│   ├── features.py    Phase 3 — the shared transform
│   ├── model.py       the serialisable bundle
│   ├── train.py       Phase 4/5 — fit, score against baselines
│   └── eda.py         Phase 2 — dashboard stats + report figures
├── api/main.py        Phase 6 — FastAPI, serves the SPA too
├── web/               Leaflet + hand-rolled SVG charts, no build step
├── scripts/ablation.py  settles the collinear-importance question
├── docs/crisp-dm.md   the full six-phase write-up
└── tests/             25 tests — geometry, transform, API contract, behaviour
```

The front end has **no build step and no framework**: Leaflet for the map, and
about 200 lines of hand-written SVG for the charts. `make serve` is the whole
toolchain.

---

## Tests

```bash
make test
```

25 tests covering the geometry (haversine against known distances, grid distance
never shorter than the great circle, bearings against cardinal directions), the
transform (airport flags, weekend-gated rush hours, cyclical midnight wraparound,
single-row/batch equivalence), and the API — including two behavioural checks
that the model actually learned the city: rush hour must predict slower than 3am,
and longer trips must take longer.

---

## Known limitations

* **No live traffic.** The model knows 6pm Wednesday is usually slow; it cannot
  know the Midtown Tunnel is closed tonight.
* **No routing.** Distance is geometric, not driven. OSRM route features were
  the main edge of the Kaggle leaders and are most of the gap to their ~0.28.
* **2016 data.** Congestion pricing, ride-hailing and the pandemic have reshaped
  Manhattan traffic since. The pipeline retrains on newer data unchanged, but
  the shipped numbers describe 2016.
* **Sampled.** 1.2M of ~69M trips in the window — enough for stable estimates;
  a full pass would need out-of-core training.

Read [`docs/crisp-dm.md`](docs/crisp-dm.md) for the reasoning behind every
choice above.
