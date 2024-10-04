from cohortcraft.features import NUMERIC_FEATURES, cohort_summary
from tests.conftest import SNAPSHOT


def test_one_row_per_customer(frame, data):
    assert len(frame) == len(data.customers)
    assert frame["customer_id"].is_unique


def test_numeric_features_present_and_finite(frame):
    for col in NUMERIC_FEATURES:
        assert col in frame.columns
        assert frame[col].notna().all(), col


def test_frequency_matches_transactions(frame, data):
    counts = data.transactions.groupby("customer_id").size()
    merged = frame.set_index("customer_id")["frequency"]
    assert (merged.loc[counts.index] == counts).all()


def test_rates_bounded(frame):
    assert frame["open_rate"].between(0, 1).all()
    assert frame["click_rate"].between(0, 1).all()
    assert (frame["click_rate"] <= frame["open_rate"] + 1e-9).all()


def test_cohort_summary(conn):
    cohorts = cohort_summary(conn, SNAPSHOT)
    assert {"cohort_month", "customers", "conversion_rate"} <= set(cohorts.columns)
    assert cohorts["conversion_rate"].between(0, 1).all()
