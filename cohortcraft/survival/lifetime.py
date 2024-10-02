"""Customer lifetime modeling: Kaplan-Meier curves, log-rank tests, Cox PH."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

DAYS_PER_MONTH = 30.44

COX_NUMERIC = ["age", "purchases_per_month", "avg_order_value", "open_rate", "click_rate"]
COX_CATEGORICAL = ["acquisition_channel", "device", "region"]


def add_duration_months(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration_months"] = np.maximum(df["duration_days"] / DAYS_PER_MONTH, 1e-3)
    return df


def kaplan_meier_by_group(
    df: pd.DataFrame, group_col: str, times: np.ndarray | None = None
) -> pd.DataFrame:
    """Survival probability at given month marks for each group (wide frame)."""
    df = add_duration_months(df)
    times = times if times is not None else np.array([3, 6, 12, 18, 24])
    out = {}
    for group, part in df.groupby(group_col):
        kmf = KaplanMeierFitter().fit(part["duration_months"], part["churned"], label=str(group))
        out[group] = kmf.survival_function_at_times(times).values
        out[group] = np.append(out[group], kmf.median_survival_time_)
    index = [f"S({t}m)" for t in times] + ["median_months"]
    return pd.DataFrame(out, index=index).T.rename_axis(group_col).reset_index()


def logrank_by_group(df: pd.DataFrame, group_col: str) -> dict[str, float]:
    df = add_duration_months(df)
    res = multivariate_logrank_test(df["duration_months"], df[group_col], df["churned"])
    return {"group": group_col, "test_statistic": float(res.test_statistic), "p_value": float(res.p_value)}


def cox_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[COX_NUMERIC].copy()
    X["purchases_per_month"] = np.log1p(X["purchases_per_month"])
    X["avg_order_value"] = np.log1p(X["avg_order_value"])
    dummies = pd.get_dummies(df[COX_CATEGORICAL], drop_first=True, dtype=float)
    return pd.concat([X, dummies], axis=1)


@dataclass
class CoxModel:
    fitter: CoxPHFitter
    columns: list[str]

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        return cox_design_matrix(df).reindex(columns=self.columns, fill_value=0.0)

    def summary(self) -> pd.DataFrame:
        cols = ["coef", "exp(coef)", "p"]
        return self.fitter.summary[cols].rename(columns={"exp(coef)": "hazard_ratio"})

    @property
    def concordance(self) -> float:
        return float(self.fitter.concordance_index_)


def fit_cox(df: pd.DataFrame, penalizer: float = 0.01) -> CoxModel:
    df = add_duration_months(df)
    X = cox_design_matrix(df)
    data = X.assign(duration_months=df["duration_months"].values, churned=df["churned"].values)
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(data, duration_col="duration_months", event_col="churned")
    return CoxModel(cph, list(X.columns))


def plot_km(df: pd.DataFrame, group_col: str, path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = add_duration_months(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    for group, part in df.groupby(group_col):
        KaplanMeierFitter().fit(part["duration_months"], part["churned"], label=str(group)).plot_survival_function(
            ax=ax, ci_show=False
        )
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Survival probability (still active)")
    ax.set_title(f"Kaplan-Meier retention by {group_col}")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
