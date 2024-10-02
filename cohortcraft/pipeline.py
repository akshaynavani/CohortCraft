"""End-to-end pipeline: data -> features -> propensity -> segments -> survival/CLV."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from cohortcraft.config import Config
from cohortcraft.data import db, synthetic
from cohortcraft.features import build_modeling_frame, cohort_summary
from cohortcraft.propensity import evaluate as peval
from cohortcraft.propensity.train import save_models, score_customers, train_models
from cohortcraft.segmentation.kmeans import segment_customers
from cohortcraft.segmentation.profile import plot_segments, profile_segments
from cohortcraft.survival import clv as clv_mod
from cohortcraft.survival.lifetime import fit_cox, kaplan_meier_by_group, logrank_by_group, plot_km

log = logging.getLogger("cohortcraft")


@dataclass
class PipelineOutputs:
    frame: pd.DataFrame
    metrics: pd.DataFrame
    best_model: str
    lift: pd.DataFrame
    importance: pd.DataFrame
    segments: pd.DataFrame
    k_search: pd.DataFrame
    km_by_channel: pd.DataFrame
    logrank: list[dict]
    cox_summary: pd.DataFrame
    cox_concordance: float
    cohort_clv: pd.DataFrame
    clv_by_segment: pd.DataFrame
    cohorts: pd.DataFrame
    artifacts: dict[str, str] = field(default_factory=dict)


def generate_data(cfg: Config) -> None:
    """Generate synthetic data and load it into the SQLite database."""
    cfg.ensure_dirs()
    data = synthetic.generate(cfg.n_customers, cfg.seed, snapshot_date=cfg.snapshot_date)
    with db.connect(cfg.db_path) as conn:
        db.load_tables(conn, data.tables())
    log.info(
        "Loaded %d customers, %d transactions, %d touches into %s",
        len(data.customers), len(data.transactions), len(data.campaign_touches), cfg.db_path,
    )


def run(cfg: Config, regenerate: bool = False) -> PipelineOutputs:
    cfg.ensure_dirs()
    out = cfg.output_dir
    if regenerate or not cfg.db_path.exists():
        generate_data(cfg)

    conn = db.connect(cfg.db_path)
    try:
        frame = build_modeling_frame(conn, cfg.snapshot_date)
        cohorts = cohort_summary(conn, cfg.snapshot_date)
    finally:
        conn.close()
    log.info("Feature frame: %d rows x %d cols", *frame.shape)

    # --- propensity
    training = train_models(frame, test_size=cfg.test_size, seed=cfg.seed)
    metrics = peval.evaluate(training)
    best = peval.best_model_name(metrics)
    best_proba = training.models[best].model.predict_proba(training.X_test)[:, 1]
    lift = peval.lift_table(training.y_test.values, best_proba)
    importance = peval.feature_importance(training, best)
    artifacts = {"roc_plot": str(peval.plot_roc(training, out / "roc_curve.png"))}
    for p in save_models(training, out):
        artifacts[f"model_{p.stem}"] = str(p)
    scores = score_customers(training.models[best].model, frame)
    frame = frame.merge(scores, on="customer_id")
    log.info("Best propensity model: %s (AUC %.3f)", best, metrics["auc_roc"].max())

    # --- segmentation
    seg = segment_customers(frame, k_min=cfg.k_min, k_max=cfg.k_max, seed=cfg.seed)
    frame["segment"] = seg.labels

    # --- survival + CLV
    km_channel = kaplan_meier_by_group(frame, "acquisition_channel")
    logrank = [logrank_by_group(frame, g) for g in ("acquisition_channel", "device", "segment")]
    cox = fit_cox(frame)
    clv = clv_mod.residual_clv(
        frame, cox, cfg.clv_horizon_months, cfg.annual_discount_rate, cfg.gross_margin
    )
    frame = frame.merge(clv, on="customer_id")
    cohort_clv = clv_mod.cohort_clv_from_km(
        frame, "cohort_month", cfg.clv_horizon_months, cfg.annual_discount_rate, cfg.gross_margin
    )
    segments = profile_segments(frame)
    name_map = dict(zip(segments["segment"], segments["name"]))
    frame["segment_name"] = frame["segment"].map(name_map)
    clv_by_segment = clv_mod.summarise_clv(frame, "segment_name")

    artifacts["segment_plot"] = str(plot_segments(frame, out / "segments.png", "segment_name"))
    artifacts["km_channel_plot"] = str(plot_km(frame, "acquisition_channel", out / "km_by_channel.png"))
    artifacts["km_segment_plot"] = str(plot_km(frame, "segment_name", out / "km_by_segment.png"))

    # --- persist tables
    frame.to_csv(out / "customer_scores.csv", index=False)
    metrics.to_csv(out / "propensity_metrics.csv", index=False)
    lift.to_csv(out / "lift_table.csv", index=False)
    segments.to_csv(out / "segment_profiles.csv", index=False)
    cohort_clv.to_csv(out / "cohort_clv.csv", index=False)
    cox.summary().to_csv(out / "cox_summary.csv")
    for key in ("customer_scores", "propensity_metrics", "lift_table", "segment_profiles", "cohort_clv"):
        artifacts[key] = str(out / f"{key}.csv")

    outputs = PipelineOutputs(
        frame=frame,
        metrics=metrics,
        best_model=best,
        lift=lift,
        importance=importance,
        segments=segments,
        k_search=seg.k_search,
        km_by_channel=km_channel,
        logrank=logrank,
        cox_summary=cox.summary(),
        cox_concordance=cox.concordance,
        cohort_clv=cohort_clv,
        clv_by_segment=clv_by_segment,
        cohorts=cohorts,
        artifacts=artifacts,
    )
    (out / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
    return outputs
