"""Clustering algorithm zoo + the inductive-assignment wrapper deployment needs.

The zoo deliberately spans the four families that behave differently on this kind
of data, so algorithm selection is a real decision rather than a formality:

  centroid    KMeans, MiniBatchKMeans, BisectingKMeans  (MacQueen 1967; Sculley 2010)
  model-based GaussianMixture                           (Fraley & Raftery 1998)
  hierarchical Agglomerative (ward/average/complete), Birch (Zhang 1996)
  graph/density Spectral (von Luxburg 2007), HDBSCAN (Campello 2013)

Half of these are *transductive*: they produce labels for the data they saw and
expose no ``predict``. That is fine for a paper and fatal for a deployment, so
``InductiveAssigner`` induces a nearest-centroid rule from any partition and the
pipeline reports its fidelity - the agreement between the induced rule and the
original transductive labels. A segmentation you cannot apply to next month's
book is not a deliverable.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import (
    HDBSCAN, AgglomerativeClustering, Birch, BisectingKMeans, KMeans,
    MiniBatchKMeans, SpectralClustering,
)
from sklearn.mixture import GaussianMixture

from . import config

# Algorithms whose cost is super-linear; guarded so a single evaluation cannot
# blow the search budget.
MAX_N = {"spectral": 3000, "agglo_ward": 12000, "agglo_average": 12000,
         "agglo_complete": 12000}

ALGO_FAMILY = {
    "kmeans": "centroid", "minibatch_kmeans": "centroid", "bisecting_kmeans": "centroid",
    "gmm": "model-based",
    "agglo_ward": "hierarchical", "agglo_average": "hierarchical",
    "agglo_complete": "hierarchical", "birch": "hierarchical",
    "spectral": "graph", "hdbscan": "density",
}


def build_clusterer(cfg: dict, seed: int = config.RANDOM_SEED) -> BaseEstimator:
    """Instantiate the estimator described by a candidate configuration."""
    a, k = cfg["algorithm"], int(cfg.get("k", 4))
    if a == "kmeans":
        return KMeans(n_clusters=k, n_init=int(cfg.get("n_init", 10)),
                      init=cfg.get("init", "k-means++"), random_state=seed)
    if a == "minibatch_kmeans":
        return MiniBatchKMeans(n_clusters=k, n_init=int(cfg.get("n_init", 10)),
                               batch_size=1024, random_state=seed)
    if a == "bisecting_kmeans":
        return BisectingKMeans(n_clusters=k, n_init=1, random_state=seed,
                               bisecting_strategy=cfg.get("bisecting", "biggest_inertia"))
    if a == "gmm":
        return GaussianMixture(n_components=k, covariance_type=cfg.get("covariance", "full"),
                               reg_covar=float(cfg.get("reg_covar", 1e-6)),
                               n_init=2, random_state=seed)
    if a.startswith("agglo_"):
        linkage = a.split("_", 1)[1]
        metric = "euclidean" if linkage == "ward" else cfg.get("metric", "euclidean")
        return AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
    if a == "birch":
        return Birch(n_clusters=k, threshold=float(cfg.get("birch_threshold", 0.5)),
                     branching_factor=50)
    if a == "spectral":
        return SpectralClustering(n_clusters=k, affinity=cfg.get("affinity", "nearest_neighbors"),
                                  n_neighbors=int(cfg.get("n_neighbors", 15)),
                                  assign_labels="kmeans", random_state=seed, n_jobs=-1)
    if a == "hdbscan":
        return HDBSCAN(min_cluster_size=int(cfg.get("min_cluster_size", 150)),
                       min_samples=int(cfg.get("min_samples", 10)),
                       cluster_selection_method=cfg.get("selection", "eom"))
    raise ValueError(f"unknown algorithm {a!r}")


def fit_predict(est: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Uniform label extraction across sklearn's two different clustering APIs."""
    if isinstance(est, GaussianMixture):
        return est.fit(X).predict(X)
    return np.asarray(est.fit_predict(X))


def is_inductive(est: BaseEstimator) -> bool:
    return hasattr(est, "predict") and not isinstance(
        est, (AgglomerativeClustering, SpectralClustering, HDBSCAN))


class InductiveAssigner(BaseEstimator):
    """Nearest-centroid rule induced from an arbitrary partition.

    For centroid and mixture models this reproduces the native ``predict`` almost
    exactly. For hierarchical, spectral and density models it is an *approximation*,
    and ``fidelity_`` measures how good an approximation it is on the training
    partition, so the cost of making the model deployable is measured rather than
    assumed. Noise points (-1) are excluded from the centroid estimate; at scoring
    time nothing is ever labelled noise, since every account must receive a segment.
    """

    def __init__(self):
        self.centroids_: np.ndarray | None = None

    def fit(self, X: np.ndarray, labels: np.ndarray) -> "InductiveAssigner":
        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels)
        self.classes_ = np.array(sorted(int(c) for c in np.unique(labels) if c != -1))
        self.centroids_ = np.vstack([X[labels == c].mean(axis=0) for c in self.classes_])
        pred = self.predict(X)
        core = labels != -1
        self.fidelity_ = float((pred[core] == labels[core]).mean()) if core.any() else 0.0
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(d, axis=1)]

    def decision_margin(self, X: np.ndarray) -> np.ndarray:
        """Gap between nearest and second-nearest centroid, normalised.

        Low margin = an account sitting on a segment boundary. These are the records
        whose assignment will flip between monthly refreshes, and the ones a campaign
        manager should be told about before they are targeted.
        """
        X = np.asarray(X, dtype=float)
        d = np.sqrt(((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2))
        s = np.sort(d, axis=1)
        return (s[:, 1] - s[:, 0]) / (s[:, 1] + 1e-12)
