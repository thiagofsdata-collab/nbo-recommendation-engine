-- =============================================================
-- Project : NBO Recommendation Engine
-- Script  : sql/transforms/01_feature_engineering.sql
-- Purpose : Build training dataset with features + target
-- =============================================================


-- -----------------------------------------------------------
-- Step 1 — Target: new product acquisitions in May 2016
-- -----------------------------------------------------------
DROP TABLE IF EXISTS mart_targets;
CREATE TABLE mart_targets AS
SELECT
    curr.customer_id,
    curr.product_name,
    1 AS target
FROM customer_products curr
LEFT JOIN customer_products prev
    ON  prev.customer_id = curr.customer_id
    AND prev.product_name = curr.product_name
    AND prev.snapshot_date = '2016-04-28'
WHERE curr.snapshot_date = '2016-05-28'
  AND curr.has_product = TRUE
  AND (prev.has_product = FALSE OR prev.has_product IS NULL);


-- -----------------------------------------------------------
-- Step 2 — Customer profile features (snapshot Abril 2016)
-- -----------------------------------------------------------
DROP TABLE IF EXISTS mart_customer_features;
CREATE TABLE mart_customer_features AS
SELECT
    cus_snap.customer_id,
    cus_snap.gross_income,
    cus_snap.seniority_months,
    cus_snap.new_customer,
    cus_snap.residence_index,
    cus_snap.segment,
    cus_snap.age,
    cus_snap.active_customer,
    cus.country,
    cus.sex,
    cus.acquisition_channel
FROM customer_snapshots cus_snap
INNER JOIN customers cus
    ON cus.customer_id = cus_snap.customer_id
WHERE cus_snap.snapshot_date = '2016-04-28';


-- -----------------------------------------------------------
-- Step 3 — Portfolio features (Abril 2016)
-- -----------------------------------------------------------
DROP TABLE IF EXISTS mart_portfolio_features;
CREATE TABLE mart_portfolio_features AS
SELECT
    cp.customer_id,
    COUNT(*) AS total_products,
    COUNT(*) FILTER (WHERE pc.product_category = 'accounts') AS total_accounts,
    COUNT(*) FILTER (WHERE pc.product_category = 'credit') AS total_credit,
    COUNT(*) FILTER (WHERE pc.product_category = 'investments') AS total_investments,
    MAX(CASE WHEN cp.product_name = 'current_account' THEN 1 ELSE 0 END) AS has_current_account,
    MAX(CASE WHEN cp.product_name = 'credit_card' THEN 1 ELSE 0 END) AS has_credit_card
FROM customer_products cp
INNER JOIN product_catalog pc
    ON pc.product_name = cp.product_name
WHERE cp.has_product = TRUE
  AND cp.snapshot_date = '2016-04-28'
GROUP BY cp.customer_id;


-- -----------------------------------------------------------
-- Step 4 — Behavioral features (últimos 3 meses)
-- -----------------------------------------------------------
DROP TABLE IF EXISTS mart_behavioral_features;
CREATE TABLE mart_behavioral_features AS
WITH new_acquisitions AS (
    SELECT
        curr.customer_id,
        curr.snapshot_date,
        COUNT(*) AS new_products_this_month
    FROM customer_products curr
    LEFT JOIN customer_products prev
        ON  prev.customer_id = curr.customer_id
        AND prev.product_name = curr.product_name
        AND prev.snapshot_date = curr.snapshot_date - INTERVAL '1 month'
    WHERE curr.snapshot_date IN ('2016-02-28', '2016-03-28', '2016-04-28')
      AND curr.has_product = TRUE
      AND (prev.has_product = FALSE OR prev.has_product IS NULL)
    GROUP BY curr.customer_id, curr.snapshot_date
)
SELECT
    customer_id,
    SUM(new_products_this_month) AS total_new_products_3m,
    SUM(CASE WHEN snapshot_date = '2016-04-28'
             THEN new_products_this_month ELSE 0 END) AS new_products_last_month
FROM new_acquisitions
GROUP BY customer_id;


-- -----------------------------------------------------------
-- Step 5 — Final training dataset
-- -----------------------------------------------------------
DROP TABLE IF EXISTS mart_training_dataset;
CREATE TABLE mart_training_dataset AS
SELECT
    base.customer_id,
    base.product_name,
    COALESCE(mt.target, 0) AS target,
    mcf.gross_income,
    mcf.seniority_months,
    mcf.new_customer,
    mcf.residence_index,
    mcf.segment,
    mcf.age,
    mcf.active_customer,
    mcf.country,
    mcf.sex,
    mcf.acquisition_channel,
    COALESCE(mpf.total_products, 0) AS total_products,
    COALESCE(mpf.total_accounts, 0) AS total_accounts,
    COALESCE(mpf.total_credit, 0) AS total_credit,
    COALESCE(mpf.total_investments, 0) AS total_investments,
    COALESCE(mpf.has_current_account, 0) AS has_current_account,
    COALESCE(mpf.has_credit_card, 0) AS has_credit_card,
    COALESCE(mbf.total_new_products_3m, 0) AS total_new_products_3m,
    COALESCE(mbf.new_products_last_month, 0) AS new_products_last_month
FROM (
    SELECT
        cs.customer_id,
        pc.product_name
    FROM customer_snapshots cs
    CROSS JOIN product_catalog pc
    LEFT JOIN customer_products cp
        ON cp.customer_id = cs.customer_id
        AND cp.product_name = pc.product_name
        AND cp.snapshot_date = '2016-04-28'
    WHERE cs.snapshot_date = '2016-04-28'
      AND pc.is_active = TRUE
      AND (cp.has_product = FALSE OR cp.has_product IS NULL)
) base
LEFT JOIN mart_targets mt
    ON  mt.customer_id = base.customer_id
    AND mt.product_name = base.product_name
LEFT JOIN mart_customer_features mcf
    ON mcf.customer_id = base.customer_id
LEFT JOIN mart_portfolio_features mpf
    ON mpf.customer_id = base.customer_id
LEFT JOIN mart_behavioral_features mbf
    ON mbf.customer_id = base.customer_id;