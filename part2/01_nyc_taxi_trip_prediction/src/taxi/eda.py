"""CRISP-DM Phase 2 (data understanding), rendered twice.

    python -m taxi.eda

Writes ``web/data/stats.json`` for the live dashboard and PNG figures into
``docs/figures/`` for the written report.  One computation, two audiences —
the numbers in the write-up and the numbers on screen cannot disagree.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, FIGURES, WEB
from .features import grid_distance_km, haversine_km

INK = "#0f172a"
ACCENT = "#f5a524"
ACCENT2 = "#2563eb"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12, color=INK, pad=12, loc="left", fontweight="600")
    ax.set_xlabel(xlabel, fontsize=9, color="#64748b")
    ax.set_ylabel(ylabel, fontsize=9, color="#64748b")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cbd5e1")
    ax.tick_params(colors="#64748b", labelsize=8)


def compute(df: pd.DataFrame) -> dict:
    d = df["trip_duration"]
    km = haversine_km(df["pickup_latitude"], df["pickup_longitude"],
                      df["dropoff_latitude"], df["dropoff_longitude"])
    grid = grid_distance_km(df["pickup_latitude"], df["pickup_longitude"],
                            df["dropoff_latitude"], df["dropoff_longitude"])
    ts = df["pickup_datetime"]
    hour, dow = ts.dt.hour, ts.dt.dayofweek
    speed = grid / (d / 3600.0)

    # Speed by hour is the headline finding: the model is really learning a
    # congestion clock, and this is the chart that shows it.
    by_hour = (
        pd.DataFrame({"hour": hour, "dur": d, "speed": speed})
        .groupby("hour")
        .agg(trips=("dur", "size"), median_duration=("dur", "median"),
             median_speed=("speed", "median"))
        .reset_index()
    )
    by_dow = (
        pd.DataFrame({"dow": dow, "dur": d})
        .groupby("dow").agg(trips=("dur", "size"), median_duration=("dur", "median"))
        .reset_index()
    )
    heat = (
        pd.crosstab(dow, hour)
        .reindex(index=range(7), columns=range(24), fill_value=0)
        .to_numpy()
    )

    dur_hist, dur_edges = np.histogram(d[d <= 3600], bins=60)
    km_hist, km_edges = np.histogram(km[km <= 25], bins=60)

    # Thin the point cloud for the map layer; 12k markers is the most a browser
    # renders smoothly, and it is plenty to show where the density lives.
    sample = df.sample(min(12_000, len(df)), random_state=255)

    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "n_trips": int(len(df)),
        "period": [str(ts.min()), str(ts.max())],
        "duration": {
            "median_s": float(d.median()), "mean_s": float(d.mean()),
            "p10_s": float(d.quantile(0.10)), "p90_s": float(d.quantile(0.90)),
            "p99_s": float(d.quantile(0.99)),
        },
        "distance": {
            "median_haversine_km": float(np.median(km)),
            "median_grid_km": float(np.median(grid)),
            "median_detour_ratio": float(np.median(grid / np.maximum(km, 0.05))),
        },
        "speed": {
            "median_kmh": float(np.median(speed)),
            "rush_pm_kmh": float(np.median(speed[(hour >= 16) & (hour < 20) & (dow < 5)])),
            "overnight_kmh": float(np.median(speed[(hour >= 1) & (hour < 5)])),
        },
        "by_hour": by_hour.to_dict("records"),
        "by_dow": by_dow.to_dict("records"),
        "heatmap_dow_hour": heat.tolist(),
        "duration_hist": {"counts": dur_hist.tolist(), "edges": dur_edges.tolist()},
        "distance_hist": {"counts": km_hist.tolist(), "edges": km_edges.tolist()},
        "pickup_points": [
            [round(float(a), 5), round(float(b), 5)]
            for a, b in zip(sample["pickup_latitude"], sample["pickup_longitude"])
        ],
    }


def figures(df: pd.DataFrame, stats: dict) -> None:
    d = df["trip_duration"]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.hist(d[d <= 3600] / 60, bins=60, color=ACCENT2, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(d.median() / 60, color=ACCENT, linewidth=2,
               label=f"median {d.median()/60:.1f} min")
    _style(ax, "Trip duration is right-skewed — the long tail is the hard part",
           "minutes", "trips")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "duration_distribution.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.hist(np.log1p(d), bins=60, color="#7c3aed", alpha=0.85, edgecolor="white", linewidth=0.4)
    _style(ax, "log1p(duration) is near-Gaussian — which is why we model in log space",
           "log1p(seconds)", "trips")
    fig.tight_layout(); fig.savefig(FIGURES / "duration_log_distribution.png"); plt.close(fig)

    bh = pd.DataFrame(stats["by_hour"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), dpi=140, sharex=True)
    a1.bar(bh["hour"], bh["trips"], color=ACCENT2, alpha=0.85)
    _style(a1, "Demand by hour of day", "", "trips")
    a2.plot(bh["hour"], bh["median_speed"], color=ACCENT, marker="o", linewidth=2, markersize=4)
    _style(a2, "Median speed by hour — the congestion clock the model learns",
           "hour", "km/h")
    fig.tight_layout(); fig.savefig(FIGURES / "hourly_profile.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=140)
    im = ax.imshow(np.array(stats["heatmap_dow_hour"]), aspect="auto", cmap="magma")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [str(h) for h in range(0, 24, 2)])
    _style(ax, "Pickup volume by weekday x hour", "hour", "")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="trips")
    fig.tight_layout(); fig.savefig(FIGURES / "demand_heatmap.png"); plt.close(fig)

    s = df.sample(min(60_000, len(df)), random_state=255)
    fig, ax = plt.subplots(figsize=(6, 6.6), dpi=140)
    ax.scatter(s["pickup_longitude"], s["pickup_latitude"], s=0.35, alpha=0.25,
               color=ACCENT, linewidths=0)
    ax.set_facecolor("#0b1220")
    _style(ax, "Pickups trace the street grid without a basemap", "longitude", "latitude")
    ax.grid(False)
    fig.tight_layout(); fig.savefig(FIGURES / "pickup_scatter.png"); plt.close(fig)

    km = haversine_km(s["pickup_latitude"], s["pickup_longitude"],
                      s["dropoff_latitude"], s["dropoff_longitude"])
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=140)
    ax.scatter(km, s["trip_duration"] / 60, s=1.2, alpha=0.12, color=ACCENT2, linewidths=0)
    ax.set_xlim(0, 25); ax.set_ylim(0, 90)
    _style(ax, "Distance sets the floor; congestion sets the spread",
           "straight-line km", "minutes")
    fig.tight_layout(); fig.savefig(FIGURES / "distance_vs_duration.png"); plt.close(fig)

    print(f"Wrote 6 figures to {FIGURES}")


def main() -> None:
    src = DATA_PROCESSED / "trips.parquet"
    if not src.exists():
        raise SystemExit(f"{src} missing — run `python -m taxi.ingest` first.")
    df = pd.read_parquet(src)
    stats = compute(df)

    audit = DATA_PROCESSED / "cleaning_audit.json"
    if audit.exists():
        stats["cleaning_audit"] = json.loads(audit.read_text())

    out_dir = WEB / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats))
    print(f"Wrote {out_dir / 'stats.json'} ({len(df):,} trips)")
    figures(df, stats)


if __name__ == "__main__":
    main()
