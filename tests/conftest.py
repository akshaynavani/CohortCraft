import pytest

from cohortcraft.data import db, synthetic
from cohortcraft.features import build_modeling_frame

SNAPSHOT = "2025-12-31"


@pytest.fixture(scope="session")
def data():
    return synthetic.generate(n_customers=800, seed=7, snapshot_date=SNAPSHOT)


@pytest.fixture(scope="session")
def conn(data):
    connection = db.connect(":memory:")
    db.load_tables(connection, data.tables())
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def frame(conn):
    return build_modeling_frame(conn, SNAPSHOT)
