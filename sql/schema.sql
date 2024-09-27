-- Raw tables loaded from the generator (or an external extract with the same shape).
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
    customer_id         INTEGER PRIMARY KEY,
    signup_date         TEXT NOT NULL,
    cohort_month        TEXT NOT NULL,
    acquisition_channel TEXT NOT NULL,
    region              TEXT NOT NULL,
    device              TEXT NOT NULL,
    age                 INTEGER
);

DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    txn_id      INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    txn_date    TEXT NOT NULL,
    amount      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);

DROP TABLE IF EXISTS campaign_touches;
CREATE TABLE campaign_touches (
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    touch_date    TEXT NOT NULL,
    campaign_type TEXT NOT NULL,
    opened        INTEGER NOT NULL,
    clicked       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_touch_customer ON campaign_touches(customer_id);

DROP TABLE IF EXISTS outcomes;
CREATE TABLE outcomes (
    customer_id   INTEGER PRIMARY KEY REFERENCES customers(customer_id),
    converted     INTEGER NOT NULL,
    churned       INTEGER NOT NULL,
    duration_days INTEGER NOT NULL,
    churn_date    TEXT
);
