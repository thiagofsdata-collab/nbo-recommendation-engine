# =============================================================
# Project : NBO Recommendation Engine
# Script  : src/monitoring/drift_report.py
# Purpose : Detect data drift between training reference period
#           and current customer profiles using KS test.
# =============================================================

import os
import logging
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

NUMERIC_FEATURES = [
    "gross_income",
    "seniority_months",
    "age",
    "total_accounts",
    "total_credit",
    "total_investments",
    "total_new_products_3m",
    "new_products_last_month",
]

CATEGORICAL_FEATURES = ["segment"]

DRIFT_THRESHOLD = 0.05  # p-value threshold for KS test


def load_snapshot(engine, snapshot_date: str, limit: int = 50_000) -> pd.DataFrame:
    """Load customer features for a given snapshot date."""
    log.info(f"Loading snapshot {snapshot_date}...")
    df = pd.read_sql(
        f"""
        SELECT DISTINCT ON (cs.customer_id)
            cs.customer_id,
            cs.gross_income,
            cs.seniority_months,
            cs.age,
            cs.segment,
            COALESCE(mpf.total_accounts, 0) AS total_accounts,
            COALESCE(mpf.total_credit, 0) AS total_credit,
            COALESCE(mpf.total_investments, 0) AS total_investments,
            COALESCE(mbf.total_new_products_3m, 0) AS total_new_products_3m,
            COALESCE(mbf.new_products_last_month, 0) AS new_products_last_month
        FROM customer_snapshots cs
        LEFT JOIN mart_portfolio_features mpf
            ON mpf.customer_id = cs.customer_id
        LEFT JOIN mart_behavioral_features mbf
            ON mbf.customer_id = cs.customer_id
        WHERE cs.snapshot_date = '{snapshot_date}'
        LIMIT {limit}
        """,
        engine
    )
    log.info(f"Loaded {len(df):,} rows for {snapshot_date}")
    return df


def check_numeric_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DRIFT_THRESHOLD
) -> pd.DataFrame:
    """
    KS test for numeric features.
    H0: distributions are the same.
    p < threshold → reject H0 → drift detected.
    """
    results = []
    for col in NUMERIC_FEATURES:
        ref = reference[col].dropna()
        cur = current[col].dropna()
        stat, pvalue = stats.ks_2samp(ref, cur)
        drift = pvalue < threshold
        results.append({
            "feature": col,
            "type": "numeric",
            "ks_statistic": round(stat, 4),
            "p_value": round(pvalue, 6),
            "drift_detected": drift,
            "ref_mean": round(ref.mean(), 2),
            "cur_mean": round(cur.mean(), 2),
            "mean_delta_pct": round((cur.mean() - ref.mean()) / (ref.mean() + 1e-9) * 100, 1),
        })
        status = "DRIFT ⚠" if drift else "OK ✓"
        log.info(
            f"{col:30} | KS={stat:.4f} | p={pvalue:.4f} | "
            f"ref_mean={ref.mean():.1f} | cur_mean={cur.mean():.1f} | {status}"
        )
    return pd.DataFrame(results)


def check_categorical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DRIFT_THRESHOLD
) -> pd.DataFrame:
    """
    Chi-square test for categorical features.
    Compares proportion of each category between reference and current.
    """
    results = []
    for col in CATEGORICAL_FEATURES:
        ref_counts = reference[col].value_counts(normalize=True)
        cur_counts = current[col].value_counts(normalize=True)

        # align categories
        all_cats = set(ref_counts.index) | set(cur_counts.index)
        ref_freq = [ref_counts.get(c, 0) for c in all_cats]
        cur_freq = [cur_counts.get(c, 0) for c in all_cats]

        stat, pvalue = stats.chisquare(
            f_obs=[x * len(current) for x in cur_freq],
            f_exp=[x * len(current) for x in ref_freq]
        )
        drift = pvalue < threshold
        results.append({
            "feature": col,
            "type": "categorical",
            "chi2_statistic": round(stat, 4),
            "p_value": round(pvalue, 6),
            "drift_detected": drift,
        })
        status = "DRIFT ⚠" if drift else "OK ✓"
        log.info(f"{col:30} | chi2={stat:.4f} | p={pvalue:.4f} | {status}")
    return pd.DataFrame(results)


def save_report(numeric_results: pd.DataFrame, categorical_results: pd.DataFrame):
    """Save drift report as CSV."""
    report = pd.concat([numeric_results, categorical_results], ignore_index=True)
    output_path = "data/processed/drift_report.csv"
    report.to_csv(output_path, index=False)

    n_drift = report["drift_detected"].sum()
    log.info("=" * 50)
    log.info(f"Features with drift: {n_drift} of {len(report)}")
    if n_drift > 0:
        log.info("ALERT: Consider retraining the model.")
    else:
        log.info("No significant drift detected.")
    log.info(f"Report saved to {output_path}")
    log.info("=" * 50)
    return report


if __name__ == "__main__":
    engine = create_engine(DB_URL, pool_pre_ping=True)
    reference = load_snapshot(engine, snapshot_date="2015-12-28")
    current = load_snapshot(engine, snapshot_date="2016-04-28")

    numeric_results = check_numeric_drift(reference, current)
    categorical_results = check_categorical_drift(reference, current)
    report = save_report(numeric_results, categorical_results)

    print("\n", report.to_string(index=False))