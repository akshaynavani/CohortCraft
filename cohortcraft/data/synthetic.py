"""Seeded synthetic customer data with realistic, recoverable structure.

Each customer has a latent ``engagement`` score that drives purchase frequency,
spend, campaign responsiveness, churn hazard and conversion probability. Channel
and region shift the latent score so that cohorts differ in a known way.

Tables produced
---------------
customers         one row per customer, incl. acquisition cohort
transactions      purchase events up to churn or snapshot date
campaign_touches  marketing touches with open / click flags
outcomes          conversion label and churn (duration, event) per customer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CHANNELS = ["organic", "paid_search", "social", "referral", "email"]
CHANNEL_PROBS = [0.30, 0.25, 0.20, 0.10, 0.15]
CHANNEL_EFFECT = {"organic": 0.2, "paid_search": -0.1, "social": -0.4, "referral": 0.6, "email": 0.1}

REGIONS = ["north", "south", "east", "west"]
REGION_EFFECT = {"north": 0.1, "south": -0.1, "east": 0.0, "west": 0.05}

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_PROBS = [0.55, 0.35, 0.10]

CAMPAIGN_TYPES = ["discount", "newsletter", "product_launch", "winback"]


@dataclass
class SyntheticData:
    customers: pd.DataFrame
    transactions: pd.DataFrame
    campaign_touches: pd.DataFrame
    outcomes: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "customers": self.customers,
            "transactions": self.transactions,
            "campaign_touches": self.campaign_touches,
            "outcomes": self.outcomes,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_customers: int = 5000,
    seed: int = 42,
    start_date: str = "2023-01-01",
    snapshot_date: str = "2025-12-31",
) -> SyntheticData:
    """Generate a full synthetic dataset.

    Signups are spread uniformly between ``start_date`` and six months before
    ``snapshot_date`` so every customer has some observation history.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    snapshot = pd.Timestamp(snapshot_date)
    last_signup = snapshot - pd.DateOffset(months=6)
    signup_span = (last_signup - start).days

    ids = np.arange(1, n_customers + 1)
    channel = rng.choice(CHANNELS, size=n_customers, p=CHANNEL_PROBS)
    region = rng.choice(REGIONS, size=n_customers)
    device = rng.choice(DEVICES, size=n_customers, p=DEVICE_PROBS)
    age = np.clip(rng.normal(38, 11, n_customers).round(), 18, 80).astype(int)
    signup = start + pd.to_timedelta(rng.integers(0, signup_span + 1, n_customers), unit="D")

    engagement = (
        rng.normal(0, 1, n_customers)
        + np.vectorize(CHANNEL_EFFECT.get)(channel)
        + np.vectorize(REGION_EFFECT.get)(region)
        + np.where(device == "desktop", 0.15, 0.0)
    )

    # --- churn: exponential lifetime (days) with engagement-dependent hazard
    base_monthly_hazard = 0.06
    hazard = base_monthly_hazard * np.exp(-0.7 * engagement)
    lifetime_days = rng.exponential(30.44 / hazard)
    churn_date = signup + pd.to_timedelta(lifetime_days.round(), unit="D")
    churned = churn_date <= snapshot
    end_obs = pd.Series(np.where(churned, churn_date, snapshot), dtype="datetime64[ns]")
    duration_days = (end_obs.values - signup.values) / np.timedelta64(1, "D")

    customers = pd.DataFrame(
        {
            "customer_id": ids,
            "signup_date": signup.strftime("%Y-%m-%d"),
            "cohort_month": signup.strftime("%Y-%m"),
            "acquisition_channel": channel,
            "region": region,
            "device": device,
            "age": age,
        }
    )

    # --- transactions: Poisson process while active
    monthly_rate = np.exp(0.2 + 0.5 * engagement)  # purchases per month
    avg_ticket = np.exp(3.6 + 0.25 * engagement + rng.normal(0, 0.2, n_customers))
    txn_rows = []
    for i in range(n_customers):
        active_days = max(duration_days[i], 1.0)
        n_txn = rng.poisson(monthly_rate[i] * active_days / 30.44)
        if n_txn == 0:
            continue
        offsets = np.sort(rng.uniform(0, active_days, n_txn))
        amounts = rng.lognormal(np.log(avg_ticket[i]), 0.45, n_txn)
        dates = signup[i] + pd.to_timedelta(offsets.round(), unit="D")
        for d, a in zip(dates, amounts):
            txn_rows.append((ids[i], d.strftime("%Y-%m-%d"), round(float(a), 2)))
    transactions = pd.DataFrame(txn_rows, columns=["customer_id", "txn_date", "amount"])
    transactions.insert(0, "txn_id", np.arange(1, len(transactions) + 1))

    # --- campaign touches: roughly 1-2 per active month
    touch_rows = []
    open_p = _sigmoid(-0.6 + 0.8 * engagement)
    for i in range(n_customers):
        active_days = max(duration_days[i], 1.0)
        n_touch = rng.poisson(1.5 * active_days / 30.44)
        if n_touch == 0:
            continue
        offsets = rng.uniform(0, active_days, n_touch)
        types = rng.choice(CAMPAIGN_TYPES, n_touch)
        opened = rng.random(n_touch) < open_p[i]
        clicked = opened & (rng.random(n_touch) < _sigmoid(-1.0 + 0.6 * engagement[i]))
        dates = signup[i] + pd.to_timedelta(offsets.round(), unit="D")
        for d, t, o, c in zip(dates, types, opened, clicked):
            touch_rows.append((ids[i], d.strftime("%Y-%m-%d"), t, int(o), int(c)))
    touches = pd.DataFrame(
        touch_rows, columns=["customer_id", "touch_date", "campaign_type", "opened", "clicked"]
    )

    # --- conversion label (e.g. upgrade to paid plan in the next period)
    tenure_months = duration_days / 30.44
    conv_logit = (
        -1.6
        + 1.1 * engagement
        + 0.03 * np.minimum(tenure_months, 24)
        - 1.2 * churned.astype(float)
        + np.where(channel == "referral", 0.4, 0.0)
        + np.where(device == "mobile", -0.2, 0.0)
        - 0.01 * (age - 38)
    )
    converted = (rng.random(n_customers) < _sigmoid(conv_logit)).astype(int)

    outcomes = pd.DataFrame(
        {
            "customer_id": ids,
            "converted": converted,
            "churned": churned.astype(int),
            "duration_days": duration_days.round().astype(int),
            "churn_date": np.where(churned, churn_date.strftime("%Y-%m-%d"), None),
        }
    )
    return SyntheticData(customers, transactions, touches, outcomes)
