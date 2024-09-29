"""Evaluation: AUC-ROC, PR-AUC, Brier score, lift by decile, ROC plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from cohortcraft.propensity.models import feature_names
from cohortcraft.propensity.train import TrainingResult


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "auc_roc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7))),
        "base_rate": float(np.mean(y_true)),
    }


def lift_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift table; decile 1 holds the highest-scored customers."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    base = df["y"].mean()
    table = df.groupby("decile").agg(
        customers=("y", "size"),
        conversions=("y", "sum"),
        avg_score=("p", "mean"),
    )
    table["conversion_rate"] = table["conversions"] / table["customers"]
    table["lift"] = table["conversion_rate"] / base if base > 0 else np.nan
    table["cum_capture"] = table["conversions"].cumsum() / max(table["conversions"].sum(), 1)
    return table.reset_index()


def evaluate(result: TrainingResult) -> pd.DataFrame:
    rows = []
    for name, tm in result.models.items():
        proba = tm.model.predict_proba(result.X_test)[:, 1]
        m = classification_metrics(result.y_test.values, proba)
        m.update(model=name, cv_auc_mean=tm.cv_auc_mean, cv_auc_std=tm.cv_auc_std)
        rows.append(m)
    cols = ["model", "auc_roc", "cv_auc_mean", "cv_auc_std", "pr_auc", "brier", "log_loss", "base_rate"]
    return pd.DataFrame(rows)[cols].sort_values("auc_roc", ascending=False).reset_index(drop=True)


def best_model_name(metrics: pd.DataFrame) -> str:
    return str(metrics.iloc[metrics["auc_roc"].idxmax()]["model"])


def feature_importance(result: TrainingResult, name: str, top_n: int = 15) -> pd.DataFrame:
    """Coefficients for LR, gain-based importance for XGBoost."""
    model = result.models[name].model
    names = feature_names(model)
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        values = clf.coef_.ravel()
        df = pd.DataFrame({"feature": names, "importance": values})
        df["abs"] = df["importance"].abs()
    else:
        values = clf.feature_importances_
        df = pd.DataFrame({"feature": names, "importance": values})
        df["abs"] = df["importance"]
    return df.sort_values("abs", ascending=False).drop(columns="abs").head(top_n).reset_index(drop=True)


def plot_roc(result: TrainingResult, path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, tm in result.models.items():
        proba = tm.model.predict_proba(result.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(result.y_test, proba)
        auc = roc_auc_score(result.y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Conversion propensity — ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
