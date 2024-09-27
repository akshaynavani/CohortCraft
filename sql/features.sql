-- Customer-level feature snapshot as of {snapshot_date}.
-- Recency / frequency / monetary, engagement and campaign response features.

DROP VIEW IF EXISTS v_txn_agg;
CREATE TEMP VIEW v_txn_agg AS
SELECT
    t.customer_id,
    COUNT(*)                                                  AS frequency,
    SUM(t.amount)                                             AS monetary_total,
    AVG(t.amount)                                             AS avg_order_value,
    MAX(t.amount)                                             AS max_order_value,
    MIN(t.txn_date)                                           AS first_txn_date,
    MAX(t.txn_date)                                           AS last_txn_date,
    SUM(CASE WHEN julianday('{snapshot_date}') - julianday(t.txn_date) <= 90
             THEN 1 ELSE 0 END)                               AS txn_last_90d,
    SUM(CASE WHEN julianday('{snapshot_date}') - julianday(t.txn_date) <= 90
             THEN t.amount ELSE 0 END)                        AS spend_last_90d
FROM transactions t
WHERE t.txn_date <= '{snapshot_date}'
GROUP BY t.customer_id;

DROP VIEW IF EXISTS v_touch_agg;
CREATE TEMP VIEW v_touch_agg AS
SELECT
    ct.customer_id,
    COUNT(*)                                                  AS touches,
    SUM(ct.opened)                                            AS opens,
    SUM(ct.clicked)                                           AS clicks,
    SUM(CASE WHEN ct.campaign_type = 'discount' THEN ct.clicked ELSE 0 END) AS discount_clicks
FROM campaign_touches ct
WHERE ct.touch_date <= '{snapshot_date}'
GROUP BY ct.customer_id;

SELECT
    c.customer_id,
    c.cohort_month,
    c.acquisition_channel,
    c.region,
    c.device,
    c.age,
    julianday('{snapshot_date}') - julianday(c.signup_date)             AS tenure_days,
    COALESCE(julianday('{snapshot_date}') - julianday(ta.last_txn_date),
             julianday('{snapshot_date}') - julianday(c.signup_date))   AS recency_days,
    COALESCE(ta.frequency, 0)                                           AS frequency,
    COALESCE(ta.monetary_total, 0.0)                                    AS monetary_total,
    COALESCE(ta.avg_order_value, 0.0)                                   AS avg_order_value,
    COALESCE(ta.max_order_value, 0.0)                                   AS max_order_value,
    COALESCE(ta.txn_last_90d, 0)                                        AS txn_last_90d,
    COALESCE(ta.spend_last_90d, 0.0)                                    AS spend_last_90d,
    COALESCE(ta.frequency, 0) * 30.44
        / MAX(julianday('{snapshot_date}') - julianday(c.signup_date), 1) AS purchases_per_month,
    COALESCE(tc.touches, 0)                                             AS touches,
    CASE WHEN COALESCE(tc.touches, 0) = 0 THEN 0.0
         ELSE 1.0 * tc.opens / tc.touches END                           AS open_rate,
    CASE WHEN COALESCE(tc.touches, 0) = 0 THEN 0.0
         ELSE 1.0 * tc.clicks / tc.touches END                          AS click_rate,
    COALESCE(tc.discount_clicks, 0)                                     AS discount_clicks
FROM customers c
LEFT JOIN v_txn_agg   ta ON ta.customer_id = c.customer_id
LEFT JOIN v_touch_agg tc ON tc.customer_id = c.customer_id
ORDER BY c.customer_id;
