-- =============================================================
-- Project : Next Best Offer — Financial Product Recommendation
-- Script  : 01_create_tables.sql
-- Purpose : Raw schema — normalized from Santander flat CSV
-- Author  : Thiago Silva
-- Notes   : Run inside the `nbo` database as the app user.
--           Tables are idempotent (DROP IF EXISTS + CREATE).
-- =============================================================
 
 
-- -----------------------------------------------------------
-- 0. Extensions
-- -----------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- future-proofing for UUIDs
 
 
-- -----------------------------------------------------------
-- 1. customers
--    One row per customer. STATIC attributes only —
--    fields that do not change across monthly snapshots.
--    Mutable fields (income, segment, age, etc.) live in
--    customer_snapshots to preserve full history.
-- -----------------------------------------------------------
DROP TABLE IF EXISTS customer_products CASCADE;
DROP TABLE IF EXISTS customer_snapshots CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
 
CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,   
    employee_index VARCHAR(2),                 
    country VARCHAR(5),                  
    sex VARCHAR(2),                   
    first_contract_date DATE,                      
    foreigner_index BOOLEAN,                      
    spouse_of_employee BOOLEAN,                     
    acquisition_channel VARCHAR(10),                   
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
 
COMMENT ON TABLE  customers IS 'Static customer profile. One row per customer. Never updated after insert.';
COMMENT ON COLUMN customers.employee_index IS 'A=active employee, B=ex-employee, F=branch, N=not employee, P=passive';
COMMENT ON COLUMN customers.acquisition_channel IS 'Channel used by customer to join the bank (~160 distinct values)';
 
 
-- -----------------------------------------------------------
-- 2. customer_snapshots
--    One row per (customer, month). MUTABLE attributes —
--    everything that can change across snapshots.
--    This is the behavioural spine of the feature store.
-- -----------------------------------------------------------
CREATE TABLE customer_snapshots (
    snapshot_date DATE NOT NULL,  
    customer_id BIGINT NOT NULL,  
    age SMALLINT,                 
    seniority_months INTEGER,                
    new_customer BOOLEAN,                  
    customer_type VARCHAR(2),             
    customer_type_month_start VARCHAR(2),               
    relation_type_month_start VARCHAR(1),             
    last_date_as_primary DATE,                     
    residence_index BOOLEAN,                
    active_customer BOOLEAN,                  
    gross_income NUMERIC(15,2),            
    segment VARCHAR(2),               
    province_code SMALLINT,                 
    province_name VARCHAR(100),             
    deceased BOOLEAN,                   
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
 
    PRIMARY KEY (snapshot_date, customer_id),
    CONSTRAINT fk_snapshots_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers (customer_id)
        ON DELETE CASCADE
);
 
COMMENT ON TABLE  customer_snapshots IS 'Monthly behavioural snapshot per customer. Append-only — never update existing rows.';
COMMENT ON COLUMN customer_snapshots.seniority_months IS 'Raw data contains negative values — clamp to 0 during ingestion.';
COMMENT ON COLUMN customer_snapshots.gross_income IS 'Household gross income. ~30% NaN in raw — median-imputed per segment on load.';
COMMENT ON COLUMN customer_snapshots.segment IS '01=VIP, 02=Individuals, 03=College graduate';
 
 
-- -----------------------------------------------------------
-- 3. customer_products
--    One row per (customer, month, product).
--    LONG FORMAT — not wide.
--
--    Why long format?
--    Wide (24 bool columns): counting new products requires
--      comparing 24 column pairs across two snapshots.
--    Long (1 row per product): new products = simple WHERE +
--      GROUP BY. Window functions, CTEs, self-joins all
--      become O(1) in complexity instead of O(24 columns).
-- -----------------------------------------------------------
CREATE TABLE customer_products (
    snapshot_date DATE NOT NULL,
    customer_id BIGINT NOT NULL,
    product_name VARCHAR(50) NOT NULL, 
    has_product BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
 
    PRIMARY KEY (snapshot_date, customer_id, product_name),
    CONSTRAINT fk_products_snapshot
        FOREIGN KEY (snapshot_date, customer_id)
        REFERENCES customer_snapshots (snapshot_date, customer_id)
        ON DELETE CASCADE
);
 
COMMENT ON TABLE  customer_products IS 'Monthly product ownership in long format. 24 rows per (customer, month).';
COMMENT ON COLUMN customer_products.product_name IS 'Standardised English product name. See docs/product_catalog.md for full list.';
 
 
-- -----------------------------------------------------------
-- 4. product_catalog
--    Reference table — the 24 products and their metadata.
--    Decouples product names from magic strings in code.
-- -----------------------------------------------------------
CREATE TABLE product_catalog (
    product_name VARCHAR(50) PRIMARY KEY,
    original_column VARCHAR(30) NOT NULL,  
    display_name VARCHAR(100) NOT NULL,
    product_category VARCHAR(50),               
    is_active BOOLEAN DEFAULT TRUE,
 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
 
COMMENT ON TABLE product_catalog IS 'Reference catalog for the 24 Santander products. Join to customer_products on product_name.';
 
  ALTER TABLE customer_products
    ADD CONSTRAINT fk_products_catalog
        FOREIGN KEY (product_name)
        REFERENCES product_catalog (product_name)
        ON DELETE RESTRICT;
        
INSERT INTO product_catalog (product_name, original_column, display_name, product_category) VALUES
    ('saving_account',          'ind_ahor_fin_ult1',   'Saving Account',              'accounts'),
    ('guarantees',              'ind_aval_fin_ult1',   'Guarantees',                  'credit'),
    ('current_account',         'ind_cco_fin_ult1',    'Current Account',             'accounts'),
    ('derivada_account',        'ind_cder_fin_ult1',   'Derivada Account',            'accounts'),
    ('payroll_account',         'ind_cno_fin_ult1',    'Payroll Account',             'accounts'),
    ('junior_account',          'ind_ctju_fin_ult1',   'Junior Account',              'accounts'),
    ('mas_particular_account',  'ind_ctma_fin_ult1',   'Mas Particular Account',      'accounts'),
    ('particular_account',      'ind_ctop_fin_ult1',   'Particular Account',          'accounts'),
    ('particular_plus_account', 'ind_ctpp_fin_ult1',   'Particular Plus Account',     'accounts'),
    ('short_term_deposit',      'ind_deco_fin_ult1',   'Short-Term Deposit',          'investments'),
    ('medium_term_deposit',     'ind_deme_fin_ult1',   'Medium-Term Deposit',         'investments'),
    ('long_term_deposit',       'ind_dela_fin_ult1',   'Long-Term Deposit',           'investments'),
    ('e_account',               'ind_ecue_fin_ult1',   'E-Account',                   'accounts'),
    ('funds',                   'ind_fond_fin_ult1',   'Investment Funds',            'investments'),
    ('mortgage',                'ind_hip_fin_ult1',    'Mortgage',                    'credit'),
    ('pension_plan',            'ind_plan_fin_ult1',   'Pension Plan',                'investments'),
    ('loans',                   'ind_pres_fin_ult1',   'Personal Loans',              'credit'),
    ('taxes',                   'ind_reca_fin_ult1',   'Tax Account',                 'accounts'),
    ('credit_card',             'ind_tjcr_fin_ult1',   'Credit Card',                 'credit'),
    ('securities',              'ind_valo_fin_ult1',   'Securities',                  'investments'),
    ('home_account',            'ind_viv_fin_ult1',    'Home Account',                'accounts'),
    ('payroll',                 'ind_nomina_ult1',     'Payroll',                     'accounts'),
    ('pensions',                'ind_nom_pens_ult1',   'Pensions',                    'investments'),
    ('direct_debit',            'ind_recibo_ult1',     'Direct Debit',                'accounts');
 
 
-- -----------------------------------------------------------
-- 5. Indexes
--    Design rationale:
--    - customer_id on snapshots/products → join performance
--    - snapshot_date on snapshots/products → time-range scans
--    - product_name on products → product-level aggregations
--    - (customer_id, snapshot_date) on products → per-customer
--      time series access pattern (most common query shape)
-- -----------------------------------------------------------
 
CREATE INDEX idx_snapshots_customer_id
    ON customer_snapshots (customer_id);
 
CREATE INDEX idx_snapshots_date
    ON customer_snapshots (snapshot_date);
 
CREATE INDEX idx_snapshots_segment
    ON customer_snapshots (segment);          
 
CREATE INDEX idx_products_customer_id
    ON customer_products (customer_id);
 
CREATE INDEX idx_products_date
    ON customer_products (snapshot_date);
 
CREATE INDEX idx_products_product_name
    ON customer_products (product_name);
 
CREATE INDEX idx_products_customer_date       
    ON customer_products (customer_id, snapshot_date);
 
CREATE INDEX idx_products_has_product          
    ON customer_products (has_product)
    WHERE has_product = TRUE;                



-- -----------------------------------------------------------
-- scores_table — pre-computed recommendations
-- populated daily by Airflow, consumed by FastAPI
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommendation_scores (
    customer_id BIGINT NOT NULL,
    product_name VARCHAR(50) NOT NULL,
    score NUMERIC(6,4) NOT NULL,
    rank SMALLINT NOT NULL,
    scored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),

    PRIMARY KEY (customer_id, product_name),
    CONSTRAINT fk_scores_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers (customer_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_scores_product
        FOREIGN KEY (product_name)
        REFERENCES product_catalog (product_name)
        ON DELETE RESTRICT
);

CREATE INDEX idx_scores_customer_id
    ON recommendation_scores (customer_id);

CREATE INDEX idx_scores_rank
    ON recommendation_scores (customer_id, rank);