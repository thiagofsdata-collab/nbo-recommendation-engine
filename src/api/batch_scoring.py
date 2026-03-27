# =============================================================
# Project : NBO Recommendation Engine
# Script  : src/api/batch_scoring.py
# Purpose : Generate daily recommendation scores for all
#           propense customers and store in PostgreSQL.
#           Called by Airflow DAG daily.
# =============================================================

import os
import json
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from propensity_model import FEATURES as PROPENSITY_FEATURES
from lgbm_ranker import ALL_FEATURES as RANKING_FEATURES

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

PROPENSITY_MODEL_PATH  = "data/processed/propensity_model.txt"
RANKING_MODEL_PATH = "data/processed/lgbm_model.txt"
THRESHOLD_PATH = "data/processed/propensity_threshold.json"
MODEL_VERSION = "lgbm_v2_propensity"
TOP_K = 5

from propensity_model import FEATURES as PROPENSITY_FEATURES
from lgbm_ranker import ALL_FEATURES as RANKING_FEATURES


def prepare_features_for_predict(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Prepare features for prediction with a loaded LightGBM model.
    Converts categoricals to numeric codes — required for .txt loaded models.
    """
    X = df[features].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.Categorical(X[col]).codes
    return X


# -----------------------------------------------------------
# Step 1 — Load models
# -----------------------------------------------------------
def load_models():
    log.info("Loading propensity model...")
    propensity_model = lgb.Booster(model_file=PROPENSITY_MODEL_PATH)

    log.info("Loading ranking model...")
    ranking_model = lgb.Booster(model_file=RANKING_MODEL_PATH)

    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]

    log.info(f"Models loaded. Propensity threshold: {threshold:.2f}")
    return propensity_model, ranking_model, threshold


# -----------------------------------------------------------
# Step 2 — Load features from PostgreSQL
# -----------------------------------------------------------
def load_features(engine) -> pd.DataFrame:
    log.info("Loading features from mart_training_dataset...")
    df = pd.read_sql(
        """
        SELECT *
        FROM mart_training_dataset
        """,
        engine
    )
    log.info(f"Loaded {len(df):,} rows")

    for col in ["new_customer", "residence_index", "active_customer"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["segment"] = df["segment"].fillna("UNKNOWN").astype(str)
    df["sex"]= df["sex"].fillna("UNKNOWN").astype(str)
    df["country"] = df["country"].fillna("UNKNOWN").astype(str)
    df["gross_income"] = df["gross_income"].fillna(df["gross_income"].median())

    channel_freq = (
        df["acquisition_channel"]
        .fillna("UNKNOWN")
        .value_counts(normalize=True)
    )
    df["acquisition_channel_freq"] = (
        df["acquisition_channel"]
        .fillna("UNKNOWN")
        .map(channel_freq)
        .fillna(0)
    )
    return df


# -----------------------------------------------------------
# Step 3 — Filter propense customers
# -----------------------------------------------------------
def filter_propense(df, propensity_model, threshold) -> pd.DataFrame:
    customer_features = (
        df.drop_duplicates(subset=["customer_id"])
        [["customer_id"] + PROPENSITY_FEATURES]
        .reset_index(drop=True)
    )
    
    X_propensity = prepare_features_for_predict(customer_features, PROPENSITY_FEATURES)
    scores = propensity_model.predict(X_propensity)
    customer_features["propensity_score"] = scores
    propense = customer_features[scores >= threshold]["customer_id"]

    log.info(f"Propense customers: {len(propense):,} of {len(customer_features):,}")
    return df[df["customer_id"].isin(propense)].reset_index(drop=True)


# -----------------------------------------------------------
# Step 4 — Generate top-K scores per customer
# -----------------------------------------------------------
def generate_scores(df, ranking_model, top_k=TOP_K) -> pd.DataFrame:
    log.info("Generating ranking scores...")

    X = prepare_features_for_predict(df, RANKING_FEATURES)
    df["score"] = ranking_model.predict(X, num_iteration=ranking_model.best_iteration)

    df["rank"] = (
        df.groupby("customer_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    top_k_df = (
        df[df["rank"] <= top_k]
        [["customer_id", "product_name", "score", "rank"]]
        .sort_values(["customer_id", "rank"])
        .reset_index(drop=True)
    )

    log.info(f"Generated {len(top_k_df):,} recommendations for {top_k_df['customer_id'].nunique():,} customers")
    return top_k_df


# -----------------------------------------------------------
# Step 5 — Save to recommendation_scores
# -----------------------------------------------------------
def save_scores(scores_df, engine, model_version=MODEL_VERSION):
    log.info("Saving scores to recommendation_scores...")

    scores_df["scored_at"] = pd.Timestamp.now()
    scores_df["model_version"] = model_version
    scores_df["score"] = scores_df["score"].round(4)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM recommendation_scores"))
        log.info("Cleared existing scores.")

    scores_df.to_sql(
        "recommendation_scores",
        engine,
        if_exists="append",
        index=False,
        chunksize=10_000,
        method="multi"
    )
    log.info(f"Saved {len(scores_df):,} scores.")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    engine = create_engine(DB_URL, pool_pre_ping=True)

    propensity_model, ranking_model, threshold = load_models()
    log.info(f"Propensity pandas_categorical: {propensity_model.pandas_categorical}")
    log.info(f"Ranking pandas_categorical: {ranking_model.pandas_categorical}")
    
    df = load_features(engine)
    df = filter_propense(df, propensity_model, threshold)
    scores = generate_scores(df, ranking_model)
    save_scores(scores, engine)

    log.info("Batch scoring complete.")
    log.info(f"Total customers scored: {scores['customer_id'].nunique():,}")