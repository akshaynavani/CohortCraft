"""K-means segmentation with k selected by silhouette score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from cohortcraft.features import SEGMENTATION_FEATURES

# Heavy-tailed features are log-transformed before scaling.
_LOG_FEATURES = {"recency_days", "purchases_per_month", "avg_order_value", "spend_last_90d", "tenure_days"}


@dataclass
class SegmentationResult:
    model: Pipeline
    k: int
    labels: np.ndarray
    k_search: pd.DataFrame  # k, inertia, silhouette


def _log_transform(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if col in _LOG_FEATURES:
            X[col] = np.log1p(X[col].clip(lower=0))
    return X


def _pipeline(k: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("log", FunctionTransformer(_log_transform, validate=False)),
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
        ]
    )


def search_k(
    X: pd.DataFrame, k_min: int = 2, k_max: int = 8, seed: int = 42, sample_size: int = 3000
) -> pd.DataFrame:
    """Evaluate inertia (elbow) and silhouette for each k."""
    rows = []
    for k in range(k_min, k_max + 1):
        pipe = _pipeline(k, seed).fit(X)
        Z = pipe[:-1].transform(X)
        labels = pipe.named_steps["kmeans"].labels_
        sil = silhouette_score(Z, labels, sample_size=min(sample_size, len(X)), random_state=seed)
        rows.append({"k": k, "inertia": pipe.named_steps["kmeans"].inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def choose_k(k_search: pd.DataFrame, min_k: int = 3) -> int:
    """Pick the k with best silhouette, preferring k >= ``min_k`` for actionable segments."""
    candidates = k_search[k_search["k"] >= min_k]
    if candidates.empty:
        candidates = k_search
    return int(candidates.loc[candidates["silhouette"].idxmax(), "k"])


def segment_customers(
    features: pd.DataFrame,
    k: int | None = None,
    k_min: int = 2,
    k_max: int = 8,
    seed: int = 42,
) -> SegmentationResult:
    X = features[SEGMENTATION_FEATURES]
    k_search = search_k(X, k_min, k_max, seed)
    k = k or choose_k(k_search)
    model = _pipeline(k, seed).fit(X)
    labels = model.named_steps["kmeans"].labels_
    return SegmentationResult(model=model, k=k, labels=labels, k_search=k_search)


def assign_segments(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features[SEGMENTATION_FEATURES])
