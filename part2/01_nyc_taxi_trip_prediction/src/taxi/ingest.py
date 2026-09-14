"""CRISP-DM Phase 2/3: acquire raw trips and turn them into a modelling table.

    python -m taxi.ingest            # download (cached) + clean
    python -m taxi.ingest --refresh  # ignore the cache and re-download

Downloading is deliberately separate from cleaning: raw shards land on disk
untouched so the cleaning rules can be re-run and audited without hitting the
network again.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import calendar
import io
import json
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .config import (
    DATA_INTERIM,
    DATA_PROCESSED,
    DATA_RAW,
    MAX_DISTANCE_KM,
    MAX_DURATION_S,
    MIN_DURATION_S,
    MONTHS,
    NYC_BBOX,
    PAGES_PER_MONTH,
    PAGE_SIZE,
    SEED,
    SOCRATA_URL,
)
from .features import haversine_km

SHARDS = DATA_RAW / "socrata"
RETRIES = 4


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _fetch(month: str, page: int) -> pd.DataFrame:
    """One 50k-row page for one month, cached on disk as CSV."""
    SHARDS.mkdir(parents=True, exist_ok=True)
    path = SHARDS / f"{month}_p{page}.csv"
    if path.exists() and path.stat().st_size > 1024:
        return pd.read_csv(path)

    year, mon = (int(x) for x in month.split("-"))
    last = calendar.monthrange(year, mon)[1]
    where = (
        f"tpep_pickup_datetime >= '{month}-01T00:00:00' "
        f"AND tpep_pickup_datetime <= '{month}-{last:02d}T23:59:59'"
    )
    query = urllib.parse.urlencode(
        {
            "$where": where,
            "$limit": PAGE_SIZE,
            "$offset": page * PAGE_SIZE,
            "$order": ":id",
        }
    )
    url = f"{SOCRATA_URL}?{query}"

    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                body = resp.read()
            df = pd.read_csv(io.BytesIO(body))
            path.write_bytes(body)
            print(f"  [{month} p{page}] {len(df):,} rows", flush=True)
            return df
        except Exception as exc:                     # transient 5xx / throttling
            if attempt == RETRIES - 1:
                print(f"  [{month} p{page}] giving up: {exc}", file=sys.stderr, flush=True)
                return pd.DataFrame()
            time.sleep(2 ** attempt * 3)
    return pd.DataFrame()


def download(refresh: bool = False) -> pd.DataFrame:
    """Stratified pull: equal pages from each of the six months.

    Stratifying by month matters — an unstratified offset walk over the API
    happens to return mostly February, which would leave the model blind to
    the spring ramp in traffic.
    """
    if refresh:
        for p in SHARDS.glob("*.csv"):
            p.unlink()

    jobs = [(m, p) for m in MONTHS for p in range(PAGES_PER_MONTH)]
    print(f"Fetching {len(jobs)} pages x {PAGE_SIZE:,} rows from NYC Open Data...")
    frames = []
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for df in pool.map(lambda a: _fetch(*a), jobs):
            if len(df):
                frames.append(df)

    if not frames:
        raise SystemExit("No data downloaded — check network access.")
    raw = pd.concat(frames, ignore_index=True)
    out = DATA_INTERIM / "trips_raw.parquet"
    raw.to_parquet(out, index=False)
    print(f"Raw: {len(raw):,} rows -> {out}")
    return raw


# --------------------------------------------------------------------------- #
# reshape + clean
# --------------------------------------------------------------------------- #
KAGGLE_SCHEMA = {
    "vendorid": "vendor_id",
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "passenger_count": "passenger_count",
    "pickup_longitude": "pickup_longitude",
    "pickup_latitude": "pickup_latitude",
    "dropoff_longitude": "dropoff_longitude",
    "dropoff_latitude": "dropoff_latitude",
    "store_and_fwd_flag": "store_and_fwd_flag",
    "trip_distance": "meter_distance_mi",
}


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the Phase-3 rules, recording how many rows each one removes.

    The audit trail is the point: every drop is a modelling decision, and the
    counts go straight into the CRISP-DM write-up and the dashboard.
    """
    df = raw.rename(columns=KAGGLE_SCHEMA)
    df = df[[c for c in KAGGLE_SCHEMA.values() if c in df.columns]].copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
    for c in ("pickup_longitude", "pickup_latitude", "dropoff_longitude",
              "dropoff_latitude", "meter_distance_mi", "passenger_count", "vendor_id"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # The Kaggle target: seconds between meter-on and meter-off.
    df["trip_duration"] = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds()

    audit: list[dict] = []
    n0 = len(df)

    def step(mask: pd.Series, why: str) -> None:
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        audit.append({"rule": why, "dropped": int(before - len(df)), "remaining": len(df)})

    step(df[["pickup_datetime", "dropoff_datetime", "pickup_latitude",
             "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]]
         .notna().all(axis=1),
         "missing timestamp or coordinate")

    # A stubborn ~1% of TLC rows carry (0, 0) coordinates — the GPS never fixed.
    lon_min, lon_max, lat_min, lat_max = NYC_BBOX
    in_box = (
        df["pickup_longitude"].between(lon_min, lon_max)
        & df["dropoff_longitude"].between(lon_min, lon_max)
        & df["pickup_latitude"].between(lat_min, lat_max)
        & df["dropoff_latitude"].between(lat_min, lat_max)
    )
    step(in_box, "coordinates outside the NYC bounding box")

    step(df["trip_duration"].between(MIN_DURATION_S, MAX_DURATION_S),
         f"duration outside [{MIN_DURATION_S}s, {MAX_DURATION_S}s]")

    df["haversine_km"] = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    step(df["haversine_km"] <= MAX_DISTANCE_KM, "straight-line distance over 100 km")

    # Implied average speed catches what duration and distance filters miss:
    # a 40-minute trip covering 200 m (meter left running at a curb) and a
    # 3-minute trip covering 30 km (corrupt coordinates) both land here.
    speed = df["haversine_km"] / (df["trip_duration"] / 3600.0)
    step(speed.between(1.0, 120.0), "implied speed outside [1, 120] km/h")

    df = df.drop(columns=["haversine_km"])
    audit.append({"rule": "TOTAL", "dropped": int(n0 - len(df)), "remaining": len(df)})

    df["passenger_count"] = df["passenger_count"].fillna(1).clip(1, 6)
    df["vendor_id"] = df["vendor_id"].fillna(1)
    df = df.sort_values("pickup_datetime").reset_index(drop=True)

    kept = 100.0 * len(df) / max(n0, 1)
    print(f"Clean: {n0:,} -> {len(df):,} rows ({kept:.1f}% kept)")
    for a in audit:
        print(f"  - {a['rule']}: dropped {a['dropped']:,}")
    return df, audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download, ignoring the cache")
    args = ap.parse_args()

    raw_path = DATA_INTERIM / "trips_raw.parquet"
    if raw_path.exists() and not args.refresh:
        print(f"Reusing {raw_path}")
        raw = pd.read_parquet(raw_path)
    else:
        raw = download(refresh=args.refresh)

    df, audit = clean(raw)
    out = DATA_PROCESSED / "trips.parquet"
    df.to_parquet(out, index=False)
    (DATA_PROCESSED / "cleaning_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"Wrote {out} ({len(df):,} rows, {df['pickup_datetime'].min()} .. {df['pickup_datetime'].max()})")


if __name__ == "__main__":
    main()
