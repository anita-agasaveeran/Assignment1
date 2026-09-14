"""Tests for the parts that would fail silently.

Deliberately not a coverage exercise. Each test pins an invariant whose violation
would produce a *plausible but wrong* number on the dashboard - the failure mode
that actually costs you, as opposed to a crash, which announces itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccseg import config, data, deploy, metrics, models, prepare  # noqa: E402
from ccseg import search_space as ss  # noqa: E402


@pytest.fixture(scope="module")
def clean():
    raw, _ = data.load_raw()
    df, _ = prepare.fixed_clean(raw)
    return df


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()[0]


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def test_inactive_dimensions_do_not_create_distinct_configs():
    """Two pipelines that build the same estimator must hash the same.

    Without this the memo cache and the tabu list both leak: the optimiser would
    re-evaluate identical pipelines and report a search far wider than it was.
    """
    base = ss.random_config(np.random.default_rng(0))
    a = dict(base, algorithm="hdbscan", k=3, covariance="full")
    b = dict(base, algorithm="hdbscan", k=11, covariance="diag")
    assert ss.config_key(a) == ss.config_key(b)


def test_neighbours_change_exactly_one_active_dimension():
    rng = np.random.default_rng(1)
    cfg = ss.random_config(rng)
    for cand, move in ss.neighbours(cfg, rng, 12):
        diff = [d for d in ss.active_dims(cand)
                if d in ss.canonical(cfg) and cand.get(d) != cfg.get(d)]
        assert len(diff) <= 1, f"move {move} changed {diff}"


def test_neighbours_are_distinct_and_exclude_self():
    rng = np.random.default_rng(2)
    cfg = ss.random_config(rng)
    keys = [ss.config_key(c) for c, _ in ss.neighbours(cfg, rng, 15)]
    assert len(keys) == len(set(keys))
    assert ss.config_key(cfg) not in keys


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def test_degenerate_partition_is_strictly_dominated():
    """One giant cluster plus a speck scores a near-perfect silhouette. It must
    still lose to a legitimate partition - this is the exact failure that an
    unconstrained objective produced during development (silhouette 1.0, one
    segment holding 83% of the book).

    Note the comparison partition has three clusters, not two: a binary split is
    itself infeasible under the charter's minimum-treatment rule, so comparing
    against one would not test the constraint that matters here.
    """
    rng = np.random.default_rng(3)
    blobs = np.vstack([rng.normal(c, 1.0, (168, 4)) for c in (0, 12, 24)])
    outliers = rng.normal(120, 0.01, (6, 4))
    X = np.vstack([blobs, outliers])

    degenerate = np.array([0] * 504 + [1] * 6)          # everything, plus a speck
    legitimate = np.array([0] * 168 + [1] * 168 + [2] * 168 + [0] * 6)

    calib = metrics.ObjectiveCalibration(300.0)
    d = metrics.score_partition(X, degenerate, calib)
    b = metrics.score_partition(X, legitimate, calib)

    assert d["silhouette"] > 0.85                       # the trap is real
    assert not d["feasible"] and b["feasible"]          # only one of them is usable
    assert d["objective"] < b["objective"]              # and the constraint decides it


def test_objective_is_stationary_under_calibration():
    """The CH mapping must not depend on the trial history, or the hill climber is
    comparing scores computed on two different scales."""
    c = metrics.ObjectiveCalibration(500.0)
    m = {"silhouette": 0.3, "calinski_harabasz": 1000.0, "davies_bouldin": 1.2}
    assert c.normalise(m) == c.normalise(m)
    assert metrics.ObjectiveCalibration.from_samples([]).kappa > 0


def test_composite_weights_sum_to_one():
    assert abs(sum(metrics.COMPOSITE_WEIGHTS.values()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def test_fixed_clean_repairs_the_documented_domain_violation(raw):
    assert (raw["CASH_ADVANCE_FREQUENCY"] > 1.0).sum() > 0     # the defect is real
    out, ledger = prepare.fixed_clean(raw)
    assert (out["CASH_ADVANCE_FREQUENCY"] > 1.0).sum() == 0    # and repaired
    assert any(a["action"] == "clip_to_domain" for a in ledger["actions"])  # and logged


def test_missing_indicator_precedes_imputation(raw, clean):
    assert clean["MINPAY_IMPUTED"].sum() == raw["MINIMUM_PAYMENTS"].isna().sum()


def test_variance_stabilising_transform_actually_stabilises(clean):
    rows = {r["transform"]: r["mean_abs_skew"]
            for r in prepare.transform_effect(clean, prepare.feature_view("raw17"))}
    assert rows["yeojohnson"] < rows["none"] / 3


def test_safe_pca_clamps_instead_of_raising(clean):
    """pca_n10 is legal for the 26-column view and illegal for the 8-column one.
    The optimiser treats feature view and reduction depth as independent, so the
    illegal combination has to degrade rather than throw."""
    cfg = dict(prepare.REFERENCE_SPEC, reducer="pca_n10")
    X = prepare.build_preprocessor(cfg).fit_transform(clean[prepare.feature_view("core8")])
    assert X.shape[1] == 8


# --------------------------------------------------------------------------- #
# The neutral evaluation space - the core invariant of the whole study
# --------------------------------------------------------------------------- #
def test_reference_space_is_independent_of_the_candidate(clean):
    sub = clean.sample(800, random_state=0)
    a, _ = prepare.reference_space(sub)
    b, _ = prepare.reference_space(sub)
    assert np.allclose(a, b)
    assert a.shape[1] == len(prepare.feature_view(prepare.REFERENCE_VIEW))


def test_self_scoring_would_have_inflated_the_silhouette(clean):
    """The finding the study is built around: score the same labels in a 2-D
    projection and in the neutral space, and the projection flatters them."""
    from sklearn.cluster import KMeans
    sub = clean.sample(1500, random_state=0)
    cfg = dict(prepare.REFERENCE_SPEC, reducer="pca_n2")
    Xm = prepare.build_preprocessor(cfg).fit_transform(sub[prepare.feature_view("raw17")])
    labels = KMeans(n_clusters=5, n_init=5, random_state=0).fit_predict(Xm)
    Xe, _ = prepare.reference_space(sub)
    self_sil = metrics.internal_indices(Xm, labels)["silhouette"]
    neutral_sil = metrics.internal_indices(Xe, labels)["silhouette"]
    assert self_sil > neutral_sil + 0.10


# --------------------------------------------------------------------------- #
# Deployment
# --------------------------------------------------------------------------- #
def test_assigner_is_faithful_for_centroid_models(clean):
    from sklearn.cluster import KMeans
    sub = clean.sample(1200, random_state=0)
    X, _ = prepare.reference_space(sub)
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    a = models.InductiveAssigner().fit(X, km.labels_)
    assert a.fidelity_ > 0.99                 # nearest centroid *is* k-means' rule
    assert np.array_equal(a.predict(X), km.predict(X))


def test_assigner_never_emits_noise(clean):
    """Every account must receive a segment at scoring time, including accounts a
    density model would have declined to label."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    labels = rng.choice([-1, 0, 1, 2], size=300)
    a = models.InductiveAssigner().fit(X, labels)
    assert -1 not in set(a.predict(X))


def test_scoring_artifact_applies_its_own_feature_engineering(raw, clean):
    """Guards against training/serving skew: the caller passes the raw source
    schema, never the engineered frame."""
    from ccseg import evaluate
    cfg = dict(prepare.REFERENCE_SPEC, algorithm="kmeans", k=4, n_init=10,
               init="k-means++", feature_view="raw17")
    fit = evaluate.fit_pipeline(clean, cfg)
    art = deploy.build_artifact(fit, cfg, {0: "A", 1: "B", 2: "C", 3: "D"},
                                {"n_rows": len(clean)}, clean)
    out = art.score(raw.head(50))             # raw columns only
    assert len(out) == 50
    assert set(out["segment"]).issubset({0, 1, 2, 3})
    assert out["assignment_margin"].between(0, 1).all()


def test_psi_is_zero_for_identical_distributions_and_fires_on_a_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert deploy.psi(a, a) < 0.01
    assert deploy.psi(a, rng.normal(loc=1.5, size=5000)) > 0.25


def test_synthetic_shift_is_actually_shifted(raw):
    """The positive control must be detectable, or the monitor is untested.

    PSI is measured on PURCHASES, the feature the shift over-samples on. It is
    deliberately *not* measured on CASH_ADVANCE: that column is 52% exact zeros,
    so its training deciles collapse onto a point mass and PSI stays under the
    0.10 band even after the values are inflated by 75%. That is a real limitation
    of PSI on zero-inflated features, not a flaw in the shift, and it is why the
    drift report reads three signals rather than trusting per-feature PSI alone.
    """
    shifted = deploy.synthetic_shift(raw)
    assert shifted["PURCHASES"].mean() > raw["PURCHASES"].mean() * 1.2
    assert deploy.psi(raw["PURCHASES"].to_numpy(float),
                      shifted["PURCHASES"].to_numpy(float)) > 0.25
    assert deploy.psi(raw["BALANCE"].to_numpy(float),
                      shifted["BALANCE"].to_numpy(float)) > 0.10


# --------------------------------------------------------------------------- #
# Data quality audit
# --------------------------------------------------------------------------- #
def test_audit_finds_the_known_defects(raw):
    ids = {(i["feature"], i["finding"].split(":")[0]) for i in data.audit(raw)}
    feats = {f for f, _ in ids}
    assert "MINIMUM_PAYMENTS" in feats          # 313 missing
    assert "CASH_ADVANCE_FREQUENCY" in feats    # values > 1.0
    assert "TENURE" in feats                    # near-constant


def test_every_issue_names_its_resolution(raw):
    for i in data.audit(raw):
        assert i["resolution"].strip(), f"{i['id']} has no stated resolution"
        assert i["severity"] in {"blocker", "high", "medium", "low"}
