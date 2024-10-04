import numpy as np

from cohortcraft.survival.clv import (
    cohort_clv_from_km,
    discounted_expected_months,
    monthly_discount_rate,
    residual_clv,
)
from cohortcraft.survival.lifetime import fit_cox, kaplan_meier_by_group, logrank_by_group


def test_discounting():
    assert np.isclose((1 + monthly_discount_rate(0.10)) ** 12, 1.10)
    surv = np.ones((12, 1))
    em = discounted_expected_months(surv, 0.0)
    assert np.isclose(em[0], 12.0)
    assert discounted_expected_months(surv, 0.10)[0] < 12.0


def test_km_and_logrank(frame):
    km = kaplan_meier_by_group(frame, "acquisition_channel")
    s_cols = [c for c in km.columns if c.startswith("S(")]
    vals = km[s_cols].values
    assert ((vals >= 0) & (vals <= 1)).all()
    assert (np.diff(vals, axis=1) <= 1e-12).all()  # survival is non-increasing
    lr = logrank_by_group(frame, "acquisition_channel")
    assert 0 <= lr["p_value"] <= 1


def test_cox_and_clv(frame):
    cox = fit_cox(frame)
    assert 0.5 < cox.concordance <= 1.0
    # higher purchase frequency should reduce churn hazard in the synthetic data
    assert cox.summary().loc["purchases_per_month", "coef"] < 0
    clv = residual_clv(frame, cox, horizon_months=24)
    assert len(clv) == len(frame)
    assert (clv["residual_clv"] >= 0).all()
    churned_ids = frame.loc[frame["churned"] == 1, "customer_id"]
    assert (clv.set_index("customer_id").loc[churned_ids, "residual_clv"] == 0).all()
    cohort = cohort_clv_from_km(frame, horizon_months=24)
    assert (cohort["expected_clv"] >= 0).all()
    assert cohort["retention_12m"].between(0, 1).all()
