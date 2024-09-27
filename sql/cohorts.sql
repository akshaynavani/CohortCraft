-- Cohort-level summary: size, conversion, churn and revenue per acquisition cohort.
SELECT
    c.cohort_month,
    c.acquisition_channel,
    COUNT(*)                                     AS customers,
    AVG(o.converted)                             AS conversion_rate,
    AVG(o.churned)                               AS churn_rate,
    AVG(o.duration_days) / 30.44                 AS avg_observed_months,
    COALESCE(SUM(r.revenue), 0) / COUNT(*)       AS revenue_per_customer
FROM customers c
JOIN outcomes o ON o.customer_id = c.customer_id
LEFT JOIN (
    SELECT customer_id, SUM(amount) AS revenue
    FROM transactions
    WHERE txn_date <= '{snapshot_date}'
    GROUP BY customer_id
) r ON r.customer_id = c.customer_id
GROUP BY c.cohort_month, c.acquisition_channel
ORDER BY c.cohort_month, c.acquisition_channel;
