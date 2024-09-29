"""Training: stratified split, cross-validated AUC, final fit and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from cohortcraft.propensity.models import FEATURE_COLUMNS, MODEL_REGISTRY, TARGET, build_model


@dataclass
class TrainedModel:
    name: str
    model: Pipeline
    cv_auc_mean: float
    cv_auc_std: float


@dataclass
class TrainingResult:
    models: dict[str, TrainedModel]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_ids: pd.Series = field(repr=False)


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, df["customer_id"], test_size=test_size, stratify=y, random_state=seed
    )


def train_models(
    df: pd.DataFrame,
    model_names: list[str] | None = None,
    test_size: float = 0.25,
    seed: int = 42,
    cv_folds: int = 5,
) -> TrainingResult:
    model_names = model_names or list(MODEL_REGISTRY)
    X_train, X_test, y_train, y_test, _, ids_test = split(df, test_size, seed)

    pos = int(y_train.sum())
    neg = len(y_train) - pos
    spw = neg / max(pos, 1)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    trained: dict[str, TrainedModel] = {}
    for name in model_names:
        kwargs = {"scale_pos_weight": spw} if name == "xgboost" else {}
        model = build_model(name, seed=seed, **kwargs)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        trained[name] = TrainedModel(name, model, float(np.mean(scores)), float(np.std(scores)))

    return TrainingResult(trained, X_train, X_test, y_train, y_test, ids_test)


def save_models(result: TrainingResult, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir) / "models"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, tm in result.models.items():
        path = out / f"{name}.joblib"
        joblib.dump(tm.model, path)
        paths.append(path)
    return paths


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def score_customers(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Return customer_id with conversion propensity and decile (10 = highest)."""
    proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    out = pd.DataFrame({"customer_id": df["customer_id"].values, "propensity": proba})
    out["propensity_decile"] = (
        pd.qcut(out["propensity"].rank(method="first"), 10, labels=False) + 1
    )
    return out
