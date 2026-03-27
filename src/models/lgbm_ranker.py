# =============================================================
# Project : NBO Recommendation Engine
# Script  : src/models/lgbm_ranker.py
# Purpose : Train LightGBM model for product recommendation
#           and compare against baseline MAP@5
# =============================================================

import os
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import shap
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from propensity_model import FEATURES as PROPENSITY_FEATURES

from baseline import compute_popularity, recommend_baseline, evaluate_baseline

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

# -----------------------------------------------------------
# Features
# -----------------------------------------------------------
CATEGORICAL_FEATURES = ["segment", "sex", "country"]

NUMERIC_FEATURES = [
    "gross_income",
    "seniority_months",
    "age",
    #"total_products", # removed — dominates trivially
    "total_accounts",
    "total_credit",
    "total_investments",
    "has_current_account",
    "has_credit_card",
    "total_new_products_3m",
    "new_products_last_month",
]

BOOL_FEATURES = [
    "new_customer",
    "residence_index",
    "active_customer",
]

ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES + ["acquisition_channel_freq"]

PARQUET_PATH = "data/processed/training_dataset.parquet"

def load_training_data(engine, propense_customers=None):
    """
    Load mart_training_dataset from the database, preprocess features,
    split by customer_id (80/20), and return training/validation sets.
    Steps
    -----
    1. Load mart_training_dataset from PostgreSQL
    2. Validate raw columns from database
    3. Apply frequency encoding to acquisition_channel
    4. Convert boolean columns to int
    5. Prepare categoricals
    6. Validate ALL_FEATURES exist
    7. Split by customer_id using GroupShuffleSplit
    8. Return X_train, X_val, y_train, y_val, groups_train
    """
    log.info("Loading mart_training_dataset from database...")
    log.info("Loading from parquet cache...")
    df = pd.read_parquet(PARQUET_PATH)
    log.info(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")

    if propense_customers is not None:
        before = len(df)
        df = df[df["customer_id"].isin(propense_customers)].reset_index(drop=True)
        log.info(f"Propensity filter: {before:,} → {len(df):,} rows")

    required_raw = (
        NUMERIC_FEATURES
        + BOOL_FEATURES
        + CATEGORICAL_FEATURES
        + ["customer_id", "target", "acquisition_channel"]
    )
    missing_raw = [col for col in required_raw if col not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing raw columns from database: {missing_raw}")

    channel_freq = (
        df["acquisition_channel"]
        .fillna("UNKNOWN")
        .value_counts(normalize=True)
    )
    df["acquisition_channel_freq"] = (
        df["acquisition_channel"]
        .fillna("UNKNOWN")
        .map(channel_freq).fillna(0)
    )

    df["segment"] = df["segment"].fillna("UNKNOWN").astype("category")
    df["sex"] = df["sex"].fillna("UNKNOWN").astype("category")
    df["country"] = df["country"].fillna("UNKNOWN").astype("category")

    assert "acquisition_channel_freq" in df.columns, "frequency encoding failed"

    for col in NUMERIC_FEATURES:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log.info(f"Imputed {col} NaN with median={median:.2f}")

    for col in BOOL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in CATEGORICAL_FEATURES:
        df[col] = pd.Categorical(df[col]).codes

    missing_features = [col for col in ALL_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing engineered features: {missing_features}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    X = df[ALL_FEATURES].copy()
    y = df["target"].astype(int).copy()
    groups = df["customer_id"].copy()

    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    log.info(
        f"Train: {len(X_train):,} rows | "
        f"Val: {len(X_val):,} rows"
    )
    log.info(
        f"Unique customers — train: {groups_train.nunique():,} | "
        f"val: {groups.iloc[val_idx].nunique():,}"
    )

    val_customer_ids = groups.iloc[val_idx].unique()

    return X_train, X_val, y_train, y_val, groups_train, val_customer_ids


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.Booster:
    """
    Train LightGBM binary classifier for product recommendation.
    Uses scale_pos_weight to handle class imbalance.
    Early stopping based on validation AUC.
    """

    # class imbalance ratio — 22M negatives vs 35k positives
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = min(neg / pos, 100)
    log.info(f"Class ratio — neg: {neg:,} | pos: {pos:,} | scale_pos_weight: {scale_pos_weight:.1f}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "scale_pos_weight": 50, # reduzi de 100 para 50
        "learning_rate" : 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 50, # reduzi de 100 para 50
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
        "is_unbalance": False, # desliga - usei scale_pos_weight manual
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    log.info("Training LightGBM model...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        train_set=train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    log.info(f"Best iteration: {model.best_iteration}")
    log.info(f"Best validation AUC: {model.best_score['valid_0']['auc']:.4f}")

    return model


def evaluate_model(
    model: lgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
    baseline_score: float,
) -> float:


    y_pred = model.predict(X_val, num_iteration=model.best_iteration)

    # AUC 
    auc = roc_auc_score(y_val, y_pred)
    log.info(f"Validation AUC: {auc:.4f}")

    # df ranking
    df_eval = df_val[["customer_id", "product_name"]].copy()
    df_eval["y_true"] = y_val.values
    df_eval["y_pred"] = y_pred
 
    df_eval = df_eval.sort_values(["customer_id", "y_pred"], ascending=[True, False])
    df_eval["rank"] = df_eval.groupby("customer_id").cumcount() + 1
    recommendations = df_eval[df_eval["rank"] <= 5][["customer_id", "product_name", "rank"]]

    # ground truth 
    ground_truth = df_val[df_val["target"] == 1][["customer_id", "product_name"]]

    # MAP@5 
    map5 = evaluate_baseline(recommendations, ground_truth, k=5)
    log.info(f"Model  MAP@5 : {map5:.4f}")
    log.info(f"Baseline MAP@5: {baseline_score:.4f}")
    log.info(f"Improvement   : {((map5 - baseline_score) / baseline_score * 100):+.1f}%")

    for k in [1, 3, 5]:
        topk = df_eval[df_eval["rank"] <= k]
        precision = topk["y_true"].sum() / len(topk)
        total_relevant = df_eval["y_true"].sum()
        recall = topk["y_true"].sum() / total_relevant
        log.info(f"Precision@{k}: {precision:.4f} | Recall@{k}: {recall:.4f}")

    return map5   


if __name__ == "__main__":

    engine = create_engine(DB_URL, pool_pre_ping=True)

    # -----------------------------------------------------------
    # 1. Load and prepare data
    # -----------------------------------------------------------

    log.info("Loading raw dataset for propensity filter...")
    df = pd.read_parquet(PARQUET_PATH)

    for col in ["new_customer", "residence_index", "active_customer"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["segment"] = df["segment"].fillna("UNKNOWN").astype("category")
    df["sex"] = df["sex"].fillna("UNKNOWN").astype("category")
    df["gross_income"] = df["gross_income"].fillna(df["gross_income"].median())
    channel_freq = df["acquisition_channel"].fillna("UNKNOWN").value_counts(normalize=True)
    df["acquisition_channel_freq"] = df["acquisition_channel"].fillna("UNKNOWN").map(channel_freq).fillna(0)

    # -----------------------------------------------------------
    # 2. Propensity filter
    # -----------------------------------------------------------
    import json
    log.info("Loading propensity model...")
    propensity_model_lgb = lgb.Booster(model_file="data/processed/propensity_model.txt")
    with open("data/processed/propensity_threshold.json") as f:
        propensity_threshold = json.load(f)["threshold"]
    log.info(f"Propensity threshold: {propensity_threshold:.2f}")

    customer_features = (
        df.drop_duplicates(subset=["customer_id"])
        [["customer_id"] + PROPENSITY_FEATURES]
        .reset_index(drop=True)
    )

    propensity_scores = propensity_model_lgb.predict(customer_features[PROPENSITY_FEATURES])
    customer_features["is_propense"] = (propensity_scores >= propensity_threshold).astype(int)
    propense_customers = set(customer_features[customer_features["is_propense"] == 1]["customer_id"])
    log.info(f"Propense customers: {len(propense_customers):,}")

    # -----------------------------------------------------------
    # 3. Evaluation metadata — only propense customers in val set
    # -----------------------------------------------------------

    X_train, X_val, y_train, y_val, groups_train, val_customer_ids  = load_training_data(engine,propense_customers=propense_customers)

    df_val_propense = (
        df[df["customer_id"].isin(val_customer_ids)]
        [["customer_id", "product_name", "target"]]
        .reset_index(drop=True)
    )
    log.info(f"Val customers: {df_val_propense['customer_id'].nunique():,}")

    log.info(f"Val customers after propensity filter: {df_val_propense['customer_id'].nunique():,}")

    # recalculate baseline on same population
    log.info("Recalculating baseline on propense val set...")
    df_propense_full = df[df["customer_id"].isin(propense_customers)].reset_index(drop=True)
    popularity = compute_popularity(df_propense_full)
    recs_baseline = recommend_baseline(df_propense_full, popularity, top_k=5)
    gt_baseline = df_val_propense[df_val_propense["target"] == 1][["customer_id", "product_name"]]
    BASELINE_MAP5 = evaluate_baseline(recs_baseline, gt_baseline, k=5)
    log.info(f"Baseline MAP@5 (propense val): {BASELINE_MAP5:.4f}")

    # -----------------------------------------------------------
    # 4. MLflow experiment
    # -----------------------------------------------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("nbo-recommendation")

    with mlflow.start_run(run_name="lgbm_v2_propensity"):

        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("n_features", len(ALL_FEATURES))
        mlflow.log_param("train_customers", groups_train.nunique())
        mlflow.log_param("propense_customers", len(propense_customers))
        mlflow.log_param("baseline_map5", BASELINE_MAP5)

        # -----------------------------------------------------------
        # 5. Train
        # -----------------------------------------------------------
        model = train_model(X_train, y_train, X_val, y_val)

        mlflow.log_params({
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "best_iteration": model.best_iteration,
        })

        # -----------------------------------------------------------
        # 6. Evaluate on propense customers only
        # -----------------------------------------------------------
        # filter X_val and y_val to propense customers
        val_customer_ids_mask = pd.Series(X_val.index).isin(df_val_propense.index)
        propense_mask = pd.Series(val_customer_ids).isin(propense_customers).values

        X_val_propense = X_val.reset_index(drop=True)
        y_val_propense = y_val.reset_index(drop=True)
        
        map5 = evaluate_model(
            model = model,
            X_val = X_val_propense,
            y_val = y_val_propense,
            df_val = df_val_propense,
            baseline_score = BASELINE_MAP5,
        )

        mlflow.log_metric("val_auc", model.best_score["valid_0"]["auc"])
        mlflow.log_metric("val_map5", map5)
        mlflow.log_metric("baseline_map5", BASELINE_MAP5)
        mlflow.log_metric("map5_improvement_pct", (map5 - BASELINE_MAP5) / BASELINE_MAP5 * 100)

        # -----------------------------------------------------------
        # 7. SHAP
        # -----------------------------------------------------------
        log.info("Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        sample = X_val.sample(5_000, random_state=42)
        shap_values = explainer.shap_values(sample)

        shap_importance = pd.DataFrame({
            "feature": ALL_FEATURES,
            "importance": np.abs(shap_values).mean(axis=0)
        }).sort_values("importance", ascending=False)

        log.info("Top 10 features by SHAP:")
        log.info(shap_importance.head(10).to_string(index=False))

        shap_path = "data/processed/shap_importance.csv"
        shap_importance.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path)

        # -----------------------------------------------------------
        # 8. Save model
        # -----------------------------------------------------------
        model_path = "data/processed/lgbm_model.txt"
        model.save_model(model_path)
        mlflow.log_artifact(model_path, artifact_path="model")
        log.info(f"Model saved to {model_path}")

        # -----------------------------------------------------------
        # 9. Final summary
        # -----------------------------------------------------------
        log.info("=" * 50)
        log.info(f"BASELINE MAP@5: {BASELINE_MAP5:.4f}")
        log.info(f"MODEL MAP@5: {map5:.4f}")
        log.info(f"IMPROVEMENT: {(map5 - BASELINE_MAP5) / BASELINE_MAP5 * 100:+.1f}%")
        log.info("=" * 50)