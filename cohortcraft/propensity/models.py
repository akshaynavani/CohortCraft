"""Model registry: logistic regression baseline and XGBoost."""

from __future__ import annotations

from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from cohortcraft.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "converted"


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def logistic_regression(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(scale=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
        ]
    )


def xgboost(seed: int = 42, scale_pos_weight: float = 1.0) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("prep", _preprocessor(scale=False)), ("clf", clf)])


MODEL_REGISTRY: dict[str, Callable[..., Pipeline]] = {
    "logistic_regression": logistic_regression,
    "xgboost": xgboost,
}


def build_model(name: str, seed: int = 42, **kwargs) -> Pipeline:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY)}")
    factory = MODEL_REGISTRY[name]
    if name == "xgboost":
        return factory(seed=seed, **kwargs)
    return factory(seed=seed)


def feature_names(model: Pipeline) -> list[str]:
    return list(model.named_steps["prep"].get_feature_names_out())
