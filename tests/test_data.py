import pandas as pd

from cohortcraft.data import synthetic
from tests.conftest import SNAPSHOT


def test_generator_is_deterministic():
    a = synthetic.generate(200, seed=1)
    b = synthetic.generate(200, seed=1)
    pd.testing.assert_frame_equal(a.customers, b.customers)
    pd.testing.assert_frame_equal(a.transactions, b.transactions)


def test_tables_are_consistent(data):
    ids = set(data.customers["customer_id"])
    assert set(data.transactions["customer_id"]) <= ids
    assert set(data.campaign_touches["customer_id"]) <= ids
    assert len(data.outcomes) == len(data.customers)
    assert data.outcomes["converted"].between(0, 1).all()
    assert (data.outcomes["duration_days"] >= 0).all()
    assert (data.transactions["txn_date"] <= SNAPSHOT).all()


def test_label_has_both_classes(data):
    rate = data.outcomes["converted"].mean()
    assert 0.05 < rate < 0.95
