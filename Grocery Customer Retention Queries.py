"""PostgreSQL-dialect SQL (also runs on DuckDB). {asof} is injected as a date literal."""

RETENTION = """
WITH first_order AS (
  SELECT customer_id, MIN(order_date) AS first_date FROM orders GROUP BY customer_id),
second AS (                      -- days from first order to next order on a LATER day
  SELECT f.customer_id, f.first_date,
         MIN(o.order_date - f.first_date) FILTER (WHERE o.order_date > f.first_date) AS d2
  FROM first_order f JOIN orders o USING (customer_id) GROUP BY f.customer_id, f.first_date)
SELECT DATE_TRUNC('week', first_date)::date AS cohort_week,
       COUNT(*) AS customers,
       -- conditional aggregation; denominator only includes customers whose window has fully elapsed
       100.0 * COUNT(*) FILTER (WHERE d2 <= 7)  / NULLIF(COUNT(*) FILTER (WHERE first_date <= DATE '{asof}' - 7), 0)  AS d7,
       100.0 * COUNT(*) FILTER (WHERE d2 <= 30) / NULLIF(COUNT(*) FILTER (WHERE first_date <= DATE '{asof}' - 30), 0) AS d30,
       100.0 * COUNT(*) FILTER (WHERE d2 <= 60) / NULLIF(COUNT(*) FILTER (WHERE first_date <= DATE '{asof}' - 60), 0) AS d60
FROM second GROUP BY 1 ORDER BY 1"""

RFM = """
WITH base AS (
  SELECT customer_id, DATE '{asof}' - MAX(order_date) AS recency_days,
         COUNT(*) AS frequency, SUM(amount) AS monetary FROM orders GROUP BY customer_id),
scored AS (
  SELECT *, NTILE(5) OVER (ORDER BY recency_days DESC) AS r,   -- 5 = most recent
            NTILE(5) OVER (ORDER BY frequency, monetary)  AS f,  -- monetary breaks frequency ties
            NTILE(5) OVER (ORDER BY monetary)             AS m
  FROM base)
SELECT *, CASE WHEN r >= 4 AND f >= 4 THEN 'Champions'
               WHEN r >= 3 AND f >= 3 THEN 'Loyal'
               WHEN r >= 4 AND f <= 2 THEN 'New / promising'
               WHEN r <= 2 AND f >= 3 THEN 'At risk (was good)'
               WHEN r <= 2 AND f <= 2 THEN 'Lost / one-and-done'
               ELSE 'Needs attention' END AS segment
FROM scored"""

# Transparent rules-based risk score (0-100). Each component is shown so ops can see WHY.
RISK = """
WITH seq AS (
  SELECT customer_id, order_id, order_date, late,
         order_date - LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS gap,
         ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC, order_id DESC) AS rn
  FROM orders),
agg AS (
  SELECT customer_id, COUNT(*) AS n_orders, MAX(order_date) AS last_date, AVG(gap) AS avg_gap,
         MAX(late::int) FILTER (WHERE rn = 1) AS last_late
  FROM seq GROUP BY customer_id),
tix AS (
  SELECT customer_id,
         COUNT(*) FILTER (WHERE category = 'quality'       AND created_date >= DATE '{asof}' - 30) AS q30,
         COUNT(*) FILTER (WHERE category <> 'quality'      AND created_date >= DATE '{asof}' - 30) AS other30
  FROM tickets GROUP BY customer_id),
parts AS (
  SELECT a.customer_id, a.n_orders, DATE '{asof}' - a.last_date AS days_since_last,
         LEAST(45, 25.0 * (DATE '{asof}' - a.last_date) / COALESCE(a.avg_gap, 21)) AS pts_overdue,
         CASE WHEN a.n_orders = 1 THEN 15 ELSE 0 END                 AS pts_new,
         CASE WHEN COALESCE(t.q30, 0) > 0 THEN 25 ELSE 0 END         AS pts_quality,
         CASE WHEN a.last_late = 1 THEN 10 ELSE 0 END                AS pts_late,
         CASE WHEN COALESCE(t.other30, 0) > 0 THEN 5 ELSE 0 END      AS pts_other
  FROM agg a LEFT JOIN tix t USING (customer_id)
  WHERE DATE '{asof}' - a.last_date <= 120)       -- beyond 120 days treat as already churned
SELECT *, LEAST(100, pts_overdue + pts_new + pts_quality + pts_late + pts_other) AS risk_score,
       CASE GREATEST(pts_overdue, pts_new, pts_quality, pts_late, pts_other)
            WHEN pts_quality THEN 'quality_issue' WHEN pts_late THEN 'late_delivery'
            WHEN pts_overdue THEN 'overdue_vs_usual_cadence' ELSE 'new_customer' END AS top_driver
FROM parts"""

# Base table for the quality-issue question: first order, exposure, outcome, pre-exposure covariates.
QUALITY_BASE = """
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date, order_id) AS rn FROM orders),
first_o AS (SELECT customer_id, order_id, order_date AS first_date, amount AS first_amount, late AS first_late
            FROM ranked WHERE rn = 1 AND order_date <= DATE '{asof}' - 60)   -- full 60d outcome window
SELECT f.customer_id, f.first_amount, f.first_late::int AS first_late,
       -- EXPOSURE: quality ticket within 2 days of first order
       EXISTS (SELECT 1 FROM tickets t WHERE t.customer_id = f.customer_id AND t.category = 'quality'
               AND t.created_date <= f.first_date + 2)::int AS quality_issue,
       -- OUTCOME: any order from day 3 to day 60 (starts after the exposure window closes)
       EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = f.customer_id
               AND o.order_date BETWEEN f.first_date + 3 AND f.first_date + 60)::int AS reordered
FROM first_o f"""

SUPPORT = """SELECT category, COUNT(*) AS tickets, COUNT(DISTINCT customer_id) AS customers
FROM tickets GROUP BY category ORDER BY tickets DESC"""
