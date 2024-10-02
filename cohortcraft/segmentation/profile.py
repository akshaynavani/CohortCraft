"""Segment profiling and heuristic naming."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cohortcraft.features import SEGMENTATION_FEATURES


def profile_segments(df: pd.DataFrame, segment_col: str = "segment") -> pd.DataFrame:
    """Mean behaviour per segment plus size, share and outcome rates when available."""
    agg = df.groupby(segment_col)[SEGMENTATION_FEATURES].mean()
    agg.insert(0, "customers", df.groupby(segment_col).size())
    agg.insert(1, "share", agg["customers"] / len(df))
    for col in ("converted", "churned", "propensity", "clv"):
        if col in df.columns:
            agg[col] = df.groupby(segment_col)[col].mean()
    agg["name"] = name_segments(agg)
    return agg.reset_index()


def name_segments(profile: pd.DataFrame) -> pd.Series:
    """Assign readable names by comparing each segment to the population median.

    Rules use relative ranks so names stay stable across datasets:
    - high frequency + recent      -> Champions
    - high spend, less frequent    -> Big Spenders
    - recent but low activity      -> Promising Newcomers
    - long since last purchase     -> At Risk / Lapsed
    - otherwise                    -> Steady Regulars
    """
    ranks = profile[SEGMENTATION_FEATURES].rank(pct=True)
    names = {}
    used: dict[str, int] = {}
    for seg, r in ranks.iterrows():
        if r["purchases_per_month"] >= 0.75 and r["recency_days"] <= 0.5:
            name = "Champions"
        elif r["recency_days"] >= 0.75:
            name = "Lapsed" if r["purchases_per_month"] <= 0.5 else "At Risk"
        elif r["avg_order_value"] >= 0.75:
            name = "Big Spenders"
        elif r["tenure_days"] <= 0.25:
            name = "Promising Newcomers"
        else:
            name = "Steady Regulars"
        used[name] = used.get(name, 0) + 1
        names[seg] = name if used[name] == 1 else f"{name} {used[name]}"
    return pd.Series(names)


def plot_segments(df: pd.DataFrame, path: str | Path, segment_col: str = "segment") -> Path:
    """Scatter of purchase frequency vs recency coloured by segment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for seg, part in df.groupby(segment_col):
        ax.scatter(part["recency_days"], part["purchases_per_month"], s=6, alpha=0.5, label=str(seg))
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Purchases per month")
    ax.set_yscale("symlog")
    ax.set_title("Customer segments")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
