"""Survival-based customer lifetime value.

Residual CLV for an active customer i with current tenure T_i:

    CLV_i = sum_{t=1..H} m_i * S_i(T_i + t) / S_i(T_i) / (1 + d)^t

where m_i is monthly gross margin (purchase rate x average order value x margin),
S_i is the Cox-model survival function and d the monthly discount rate.
Churned customers have zero residual value. Total value adds realised margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cohortcraft.survival.lifetime import CoxModel, DAYS_PER_MONTH, add_duration_months


def monthly_discount_rate(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def discounted_expected_months(
    survival: np.ndarray, annual_rate: float
) -> np.ndarray:
    """Sum of discounted survival probabilities.

    ``survival`` has shape (horizon, n_customers) with row t-1 = P(alive at month t).
    """
    d = monthly_discount_rate(annual_rate)
    t = np.arange(1, survival.shape[0] + 1)
    discount = (1.0 + d) ** -t
    return (survival * discount[:, None]).sum(axis=0)


def residual_clv(
    df: pd.DataFrame,
    cox: CoxModel,
    horizon_months: int = 36,
    annual_rate: float = 0.10,
    margin: float = 0.35,
) -> pd.DataFrame:
    """Per-customer residual and total CLV."""
    df = add_duration_months(df).reset_index(drop=True)
    monthly_margin = df["purchases_per_month"] * df["avg_order_value"] * margin
    realised_margin = df["monetary_total"] * margin

    residual = np.zeros(len(df))
    active = df["churned"].values == 0
    if active.any():
        X = cox.design(df.loc[active])
        times = np.arange(1, horizon_months + 1, dtype=float)
        # conditional_after makes the curve relative to each customer's current tenure.
        surv = cox.fitter.predict_survival_function(
            X, times=times, conditional_after=df.loc[active, "duration_months"].values
        ).values
        surv = np.clip(np.nan_to_num(surv, nan=0.0), 0.0, 1.0)
        residual[active] = monthly_margin[active].values * discounted_expected_months(surv, annual_rate)

    out = pd.DataFrame(
        {
            "customer_id": df["customer_id"].values,
            "monthly_margin": monthly_margin.values,
            "realised_margin": realised_margin.values,
            "residual_clv": residual,
        }
    )
    out["clv"] = out["realised_margin"] + out["residual_clv"]
    return out


def cohort_clv_from_km(
    df: pd.DataFrame,
    group_col: str = "cohort_month",
    horizon_months: int = 36,
    annual_rate: float = 0.10,
    margin: float = 0.35,
) -> pd.DataFrame:
    """Expected CLV of a *new* customer in each cohort/group, from signup.

    Uses the group's Kaplan-Meier curve (extended flat beyond the last observation)
    and the group's average monthly margin among observed active months.
    """
    from lifelines import KaplanMeierFitter

    df = add_duration_months(df)
    times = np.arange(1, horizon_months + 1, dtype=float)
    rows = []
    for group, part in df.groupby(group_col):
        kmf = KaplanMeierFitter().fit(part["duration_months"], part["churned"])
        surv = kmf.survival_function_at_times(times).values[:, None]
        exp_months = float(discounted_expected_months(surv, annual_rate)[0])
        observed_months = (part["duration_days"] / DAYS_PER_MONTH).clip(lower=1).sum()
        avg_monthly_margin = part["monetary_total"].sum() * margin / observed_months
        rows.append(
            {
                group_col: group,
                "customers": len(part),
                "avg_monthly_margin": avg_monthly_margin,
                "discounted_expected_months": exp_months,
                "expected_clv": avg_monthly_margin * exp_months,
                "retention_12m": float(kmf.survival_function_at_times([12]).iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def summarise_clv(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggregate per-customer CLV (``clv`` column) by a grouping column."""
    return (
        df.groupby(by)
        .agg(
            customers=("customer_id", "size"),
            mean_clv=("clv", "mean"),
            median_clv=("clv", "median"),
            total_clv=("clv", "sum"),
            mean_residual_clv=("residual_clv", "mean"),
        )
        .sort_values("mean_clv", ascending=False)
        .reset_index()
    )
