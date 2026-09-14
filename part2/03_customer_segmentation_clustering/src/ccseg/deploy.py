"""CRISP-DM Phase 6 - Deployment.

A segmentation is deployed as a *function*, not as a column in a CSV. This module
produces the scoring artifact, measures its latency against SC-6, generates the
model card (Mitchell et al., FAT* 2019), and defines the drift monitor that decides
when the model has to be rebuilt.

The monitoring design starts from a fact specific to unsupervised learning: there
is no label to wait for, so there is no delayed ground truth that will eventually
tell you the model has gone stale. Everything must be inferred from the inputs and
from the geometry of the assignments:

  1. **Input drift**   - PSI per feature against the training distribution.
  2. **Assignment drift** - the segment mix, which moves before any single feature does.
  3. **Geometric drift** - mean distance to the assigned centroid, which rises when
     new records stop fitting the segments they are being forced into.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from datetime import date

import joblib
import numpy as np
import pandas as pd

from . import config
from .prepare import fixed_clean


# --------------------------------------------------------------------------- #
# Scoring artifact
# --------------------------------------------------------------------------- #
@dataclass
class ScoringArtifact:
    """Everything needed to assign a segment to an account that the model never saw."""
    preprocessor: object
    assigner: object
    columns: list[str]
    config: dict
    personas: dict            # cluster id -> persona name
    version: str
    seed: int
    trained_on: dict          # n rows, sha of the source file, date
    reference_stats: dict = field(default_factory=dict)   # for the drift monitor

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Raw source schema in, segment assignment out.

        ``fixed_clean`` is applied inside the artifact rather than expected of the
        caller: the engineered ratios and the domain clipping are part of the model,
        and a caller who reimplements them will eventually reimplement them wrong.
        That is training/serving skew, and it is the single most common way a
        segmentation quietly stops working in production.
        """
        cleaned, _ = fixed_clean(raw)
        X = np.asarray(self.preprocessor.transform(cleaned[self.columns]), dtype=float)
        labels = self.assigner.predict(X)
        margins = self.assigner.decision_margin(X)
        d = np.sqrt(((X[:, None, :] - self.assigner.centroids_[None, :, :]) ** 2).sum(axis=2))
        return pd.DataFrame({
            config.ID_COLUMN: raw[config.ID_COLUMN].to_numpy() if config.ID_COLUMN in raw else np.arange(len(raw)),
            "segment": labels,
            "persona": [self.personas.get(int(c), f"Segment {c}") for c in labels],
            "assignment_margin": np.round(margins, 4),
            "distance_to_centroid": np.round(d.min(axis=1), 4),
            "confident": margins > np.quantile(margins, 0.10),
            "model_version": self.version,
        })

    def save(self, path=None):
        path = path or (config.ARTIFACTS / f"segmentation_model_{self.version}.joblib")
        joblib.dump(self, path)
        return path


def build_artifact(fit: dict, cfg: dict, personas: dict, trained_on: dict,
                   reference_frame: pd.DataFrame, version: str = "1.0.0") -> ScoringArtifact:
    art = ScoringArtifact(
        preprocessor=fit["pre"], assigner=fit["assigner"], columns=fit["columns"],
        config=cfg, personas=personas, version=version, seed=config.RANDOM_SEED,
        trained_on=trained_on,
    )
    art.reference_stats = reference_distribution(reference_frame, fit)
    return art


# --------------------------------------------------------------------------- #
# Latency (SC-6)
# --------------------------------------------------------------------------- #
def latency_benchmark(art: ScoringArtifact, raw: pd.DataFrame,
                      batch_sizes=(1, 10, 100, 1000), repeats: int = 25) -> dict:
    """Per-record scoring latency at several batch sizes.

    Batch size is reported because it dominates: the per-record cost of a
    batch-of-one is almost entirely fixed overhead, and quoting only the
    batch-of-1000 number hides the latency of the real-time decisioning path.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    out = []
    for b in batch_sizes:
        ts = []
        for _ in range(repeats):
            idx = rng.choice(len(raw), b, replace=(b > len(raw)))
            chunk = raw.iloc[idx]
            t0 = time.perf_counter()
            art.score(chunk)
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        out.append({"batch_size": b,
                    "total_ms_p50": round(float(np.percentile(ts, 50)), 3),
                    "total_ms_p95": round(float(np.percentile(ts, 95)), 3),
                    "per_record_ms_p50": round(float(np.percentile(ts, 50) / b), 4),
                    "per_record_ms_p95": round(float(np.percentile(ts, 95) / b), 4)})
    return {"measurements": out,
            "environment": {"python": platform.python_version(),
                            "platform": platform.platform(),
                            "processor": platform.processor() or "n/a"},
            "note": "Single-process, cold cache excluded, no network or feature-store hop."}


# --------------------------------------------------------------------------- #
# Drift monitoring
# --------------------------------------------------------------------------- #
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index.

    Bin edges come from the *training* distribution's deciles, which is the point:
    PSI measures movement relative to what the model was fitted on. The industry
    convention - PSI < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant -
    is a heuristic from credit-scoring practice, not a hypothesis test, and it is
    used here as a trigger for investigation rather than as proof of anything.
    """
    e = np.asarray(expected, dtype=float); e = e[np.isfinite(e)]
    a = np.asarray(actual, dtype=float); a = a[np.isfinite(a)]
    if len(e) < 20 or len(a) < 20:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, bins=edges)[0] / len(e)
    pa = np.histogram(a, bins=edges)[0] / len(a)
    eps = 1e-4
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def reference_distribution(frame: pd.DataFrame, fit: dict) -> dict:
    """Frozen training-time statistics the monitor compares against."""
    from .profiling import PROFILE_FEATURES
    lab = np.asarray(fit["labels"])
    X = fit["X"]
    d = np.sqrt(((X[:, None, :] - fit["assigner"].centroids_[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return {
        "feature_quantiles": {c: [round(float(v), 4) for v in
                                  np.nanquantile(frame[c].to_numpy(float), np.linspace(0, 1, 11))]
                              for c in PROFILE_FEATURES if c in frame},
        "segment_mix": {int(c): round(float((lab == c).mean()), 4)
                        for c in np.unique(lab) if c != -1},
        "mean_distance_to_centroid": round(float(d.mean()), 4),
        "p95_distance_to_centroid": round(float(np.percentile(d, 95)), 4),
    }


def drift_report(art: ScoringArtifact, reference: pd.DataFrame, current: pd.DataFrame,
                 label: str = "current") -> dict:
    """Full three-signal drift check of ``current`` against the training reference."""
    from .profiling import PROFILE_FEATURES
    ref_clean, _ = fixed_clean(reference)
    cur_clean, _ = fixed_clean(current)

    feats = []
    for c in PROFILE_FEATURES:
        if c not in ref_clean or c not in cur_clean:
            continue
        v = psi(ref_clean[c].to_numpy(float), cur_clean[c].to_numpy(float))
        feats.append({"feature": c, "psi": round(v, 4) if np.isfinite(v) else None,
                      "verdict": ("significant" if v > 0.25 else
                                  "moderate" if v > 0.10 else "stable") if np.isfinite(v) else "n/a"})
    feats.sort(key=lambda d: -(d["psi"] or 0))

    scored = art.score(current)
    mix = {int(c): round(float((scored["segment"] == c).mean()), 4)
           for c in sorted(scored["segment"].unique())}
    ref_mix = art.reference_stats["segment_mix"]
    keys = sorted(set(mix) | {int(k) for k in ref_mix})
    p = np.array([max(ref_mix.get(k, ref_mix.get(str(k), 0.0)), 1e-4) for k in keys])
    q = np.array([max(mix.get(k, 0.0), 1e-4) for k in keys])
    mix_psi = float(((q - p) * np.log(q / p)).sum())

    dist_mean = float(scored["distance_to_centroid"].mean())
    ref_dist = art.reference_stats["mean_distance_to_centroid"]

    n_sig = sum(1 for f in feats if f["verdict"] == "significant")
    n_mod = sum(1 for f in feats if f["verdict"] == "moderate")
    triggered = n_sig > 0 or mix_psi > 0.25 or dist_mean > 1.25 * ref_dist
    return {
        "label": label, "n_records": int(len(current)),
        "feature_psi": feats,
        "n_significant": n_sig, "n_moderate": n_mod,
        "segment_mix": mix, "reference_mix": {int(k): v for k, v in ref_mix.items()},
        "segment_mix_psi": round(mix_psi, 4),
        "mean_distance_to_centroid": round(dist_mean, 4),
        "reference_mean_distance": ref_dist,
        "distance_ratio": round(dist_mean / max(ref_dist, 1e-9), 3),
        "alert": triggered,
        "verdict": ("REBUILD - inputs have moved materially" if triggered else
                    "HEALTHY - within monitoring tolerances"),
    }


def synthetic_shift(frame: pd.DataFrame, seed: int = config.RANDOM_SEED,
                    strength: float = 0.75) -> pd.DataFrame:
    """A deliberately shifted book, used to prove the monitor actually fires.

    A monitor that has never been observed to alarm is an untested monitor. This
    over-samples high-spend accounts and inflates cash advances, which is a
    plausible shape for a real shift (a promotional push plus a liquidity squeeze),
    and the drift report on it is included in the dashboard as the positive control.
    """
    rng = np.random.default_rng(seed)
    w = frame["PURCHASES"].rank(pct=True).to_numpy() ** 2 + 0.05
    idx = rng.choice(len(frame), len(frame), replace=True, p=w / w.sum())
    out = frame.iloc[idx].copy().reset_index(drop=True)
    out["CASH_ADVANCE"] = out["CASH_ADVANCE"] * (1 + strength)
    out["BALANCE"] = out["BALANCE"] * (1 + strength * 0.5)
    out[config.ID_COLUMN] = [f"SYN{i:06d}" for i in range(len(out))]
    return out


# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
def model_card(champion: dict, stability_res: dict, ref_space: dict, latency: dict,
               personas: list[dict], provenance: dict, search_summary: dict,
               criteria_results: list[dict]) -> dict:
    """Structured model card following Mitchell et al. (FAT* 2019)."""
    return {
        "model_details": {
            "name": "Credit-card behavioural segmentation",
            "version": "1.0.0",
            "date": str(date.today()),
            "type": "Unsupervised clustering pipeline (preprocessing + clusterer + induced nearest-centroid assigner)",
            "pipeline": champion.get("pipeline"),
            "selected_by": ("Hill-climbing CASH search over a "
                            f"{search_summary.get('space_size', {}).get('total_configurations', 0):,}-configuration space, "
                            f"{search_summary.get('unique_evaluations', 0)} evaluations"),
            "framework": "scikit-learn",
            "seed": config.RANDOM_SEED,
            "license_of_data": "Kaggle public dataset; see provenance",
        },
        "intended_use": {
            "primary": "Allocating marketing, retention and credit-line treatments across an "
                       "existing active credit-card book.",
            "users": "Portfolio marketing analysts and the campaign decisioning service.",
            "out_of_scope": [
                "Credit decisioning, pricing, or any adverse action. The segments are "
                "descriptive of behaviour and were never validated against repayment outcomes.",
                "Individual-level inference about a customer's circumstances. A segment is a "
                "statistical neighbourhood, not a fact about a person.",
                "Any use on a population other than active cardholders with >= 6 months tenure.",
            ],
        },
        "factors": {
            "relevant": "Behavioural only: spend, balance, cash advance, repayment, product mix.",
            "not_present": ("The source file carries no demographic, geographic or protected "
                            "attributes, so the model cannot condition on them directly."),
            "proxy_risk": ("Absence of protected attributes is not evidence of absence of "
                           "disparate impact - credit limit and cash-advance reliance are both "
                           "plausible proxies for income and, through it, for protected classes. "
                           "A disparate-impact review against the issuer's own demographic data "
                           "is a precondition of production use, and cannot be done from this file."),
        },
        "metrics": {
            "modelling_space": {k: champion["metrics"].get(k)
                                for k in ("silhouette", "davies_bouldin", "calinski_harabasz")},
            "reference_space": {k: ref_space.get(k)
                                for k in ("silhouette", "davies_bouldin", "calinski_harabasz")},
            "stability_ari_mean": stability_res.get("ari_mean"),
            "stability_ari_ci95": stability_res.get("ari_ci95"),
            "per_segment_jaccard": {k: v["mean"] for k, v in
                                    stability_res.get("per_cluster_jaccard", {}).items()},
            "scoring_latency_ms_per_record_p95": latency["measurements"][-1]["per_record_ms_p95"],
        },
        "training_data": provenance,
        "evaluation_data": "Same book. There is no held-out period; the source file is a single "
                           "six-month snapshot with no time index, which is a real limitation.",
        "ethical_considerations": [
            "Segment names are operational shorthand. 'Cash-Advance Reliant' describes an "
            "observed behaviour, not a judgement about the customer, and must not surface in "
            "any customer-facing message.",
            "The 'Dormant / Low-Engagement' action explicitly recommends limiting spend on "
            "reactivation. Applied without care this becomes systematic under-service of a "
            "group that may correlate with income - it needs a human sign-off, not an "
            "automated rule.",
            "Assignments near a segment boundary flip between refreshes; the artifact exposes "
            "an explicit confidence flag so those accounts can be excluded from treatment.",
        ],
        "caveats_and_recommendations": [
            "Six months of behaviour, no seasonality. A book scored in a promotional quarter "
            "will drift; the monitor is the mitigation, not the fix.",
            "Silhouette in the reduced modelling space overstates separation relative to the "
            "reference space; both numbers are reported and the reference-space value is the "
            "one to quote externally.",
            "Re-fit cadence: quarterly, or immediately on a drift alert. Segment identifiers "
            "are not stable across refits - map new segments to old by centroid nearest-match "
            "and publish the mapping, or campaign histories become uninterpretable.",
        ],
        "success_criteria_results": criteria_results,
        "personas": [{"cluster": p["cluster"], "persona": p["persona"], "share": p["share"],
                      "risk_posture": p["risk_posture"], "action": p["action"]} for p in personas],
    }
