"""Render a Markdown summary of a pipeline run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cohortcraft.pipeline import PipelineOutputs


def _table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal Markdown table renderer (avoids the optional `tabulate` dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def render(outputs: PipelineOutputs) -> str:
    seg_cols = ["segment", "name", "customers", "share", "recency_days", "purchases_per_month",
                "avg_order_value", "converted", "churned", "propensity", "clv"]
    seg_cols = [c for c in seg_cols if c in outputs.segments.columns]
    recent_cohorts = outputs.cohort_clv.sort_values("cohort_month").tail(12)

    parts = [
        "# CohortCraft run report",
        f"Customers modeled: **{len(outputs.frame):,}**  ",
        f"Conversion base rate: **{outputs.frame['converted'].mean():.1%}**  ",
        f"Observed churn rate: **{outputs.frame['churned'].mean():.1%}**",
        "## 1. Conversion propensity",
        f"Best model by held-out AUC-ROC: **{outputs.best_model}**",
        _table(outputs.metrics),
        "### Lift by decile (best model, test set)",
        _table(outputs.lift),
        "### Top features",
        _table(outputs.importance, ".4f"),
        "## 2. Segmentation (K-means)",
        "### k search",
        _table(outputs.k_search),
        "### Segment profiles",
        _table(outputs.segments[seg_cols], ".2f"),
        "## 3. Survival analysis",
        f"Cox PH concordance index: **{outputs.cox_concordance:.3f}**",
        "### Kaplan-Meier retention by acquisition channel",
        _table(outputs.km_by_channel),
        "### Log-rank tests",
        _table(pd.DataFrame(outputs.logrank), ".4g"),
        "### Cox hazard ratios",
        _table(outputs.cox_summary.reset_index().rename(columns={"covariate": "covariate"})),
        "## 4. Lifetime value",
        "### CLV by segment",
        _table(outputs.clv_by_segment, ".2f"),
        "### Expected new-customer CLV by cohort (last 12 cohorts)",
        _table(recent_cohorts, ".2f"),
        "## Artifacts",
        "\n".join(f"- `{k}`: {v}" for k, v in outputs.artifacts.items()),
    ]
    return "\n\n".join(parts) + "\n"


def write_report(outputs: PipelineOutputs, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(render(outputs))
    return path
