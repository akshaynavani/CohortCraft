import numpy as np

from cohortcraft.propensity.evaluate import (
    best_model_name,
    classification_metrics,
    evaluate,
    feature_importance,
    lift_table,
)
from cohortcraft.propensity.train import score_customers, train_models


def test_models_beat_random(frame):
    result = train_models(frame, seed=0, cv_folds=3)
    metrics = evaluate(result)
    assert set(metrics["model"]) == {"logistic_regression", "xgboost"}
    assert (metrics["auc_roc"] > 0.65).all()
    best = best_model_name(metrics)
    imp = feature_importance(result, best)
    assert len(imp) > 0
    scores = score_customers(result.models[best].model, frame)
    assert scores["propensity"].between(0, 1).all()
    assert set(scores["propensity_decile"]) == set(range(1, 11))


def test_lift_table_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(1, 0, 100)
    table = lift_table(y, p)
    assert table.loc[0, "conversion_rate"] == 1.0
    assert np.isclose(table.loc[0, "lift"], 10.0)
    assert np.isclose(table["cum_capture"].iloc[-1], 1.0)


def test_classification_metrics_keys():
    m = classification_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert m["auc_roc"] == 1.0
    assert set(m) == {"auc_roc", "pr_auc", "brier", "log_loss", "base_rate"}
