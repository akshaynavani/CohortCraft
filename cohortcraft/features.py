"""Feature matrix construction on top of the SQL feature store."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from cohortcraft.data.db import query, run_script_query

CATEGORICAL_FEATURES = ["acquisition_channel", "region", "device"]
NUMERIC_FEATURES = [
    "age",
    "tenure_days",
    "recency_days",
    "frequency",
    "monetary_total",
    "avg_order_value",
    "max_order_value",
    "txn_last_90d",
    "spend_last_90d",
    "purchases_per_month",
    "touches",
    "open_rate",
    "click_rate",
    "discount_clicks",
]

# Behavioural features used for clustering (no demographics / identifiers).
SEGMENTATION_FEATURES = [
    "recency_days",
    "purchases_per_month",
    "avg_order_value",
    "spend_last_90d",
    "open_rate",
    "click_rate",
    "tenure_days",
]


def build_features(conn: sqlite3.Connection, snapshot_date: str) -> pd.DataFrame:
    """Return one row per customer with engineered features."""
    df = run_script_query(conn, "features.sql", snapshot_date=snapshot_date)
    df["log_monetary_total"] = np.log1p(df["monetary_total"])
    return df


def load_outcomes(conn: sqlite3.Connection) -> pd.DataFrame:
    return query(conn, "SELECT customer_id, converted, churned, duration_days FROM outcomes")


def build_modeling_frame(conn: sqlite3.Connection, snapshot_date: str) -> pd.DataFrame:
    """Features joined with outcome labels, ready for modeling."""
    feats = build_features(conn, snapshot_date)
    return feats.merge(load_outcomes(conn), on="customer_id", how="inner")


def cohort_summary(conn: sqlite3.Connection, snapshot_date: str) -> pd.DataFrame:
    return run_script_query(conn, "cohorts.sql", snapshot_date=snapshot_date)
