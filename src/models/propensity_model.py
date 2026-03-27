# =============================================================
# Project : NBO Recommendation Engine
# Script  : src/models/propensity_model.py
# Purpose : Predict which customers will acquire ANY new product
#           Used as a pre-filter before the ranking model
# =============================================================

import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score

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

PARQUET_PATH = "data/processed/training_dataset.parquet"

FEATURES = [
    "gross_income",
    "seniority_months",
    "age",
    "total_accounts",
    "total_credit",
    "total_investments",
    "has_current_account",
    "has_credit_card",
    "total_new_products_3m",
    "new_products_last_month",
    "new_customer",
    "residence_index",
    "active_customer",
    "segment",
    "sex",
    "acquisition_channel_freq",
]


# -----------------------------------------------------------
# Step 1 — Build propensity dataset
# One row per customer (not per customer-product pair)
# -----------------------------------------------------------
def build_propensity_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the training dataset to one row per customer.
    Target = 1 if customer acquired ANY product in May, else 0.
    """
    propensity_target = (
        df.groupby("customer_id")["target"]
        .max()
        .reset_index()
        .rename(columns={"target": "will_acquire"})
    )

    customer_features = (
        df.drop_duplicates(subset=["customer_id"])
        [["customer_id"] + FEATURES]
        .reset_index(drop=True)
    )

    propensity_df = customer_features.merge(
        propensity_target,
        on="customer_id",
        how="inner"
    )

    pos = propensity_df["will_acquire"].sum()
    neg = (propensity_df["will_acquire"] == 0).sum()
    log.info(f"Propensity dataset: {len(propensity_df):,} customers")
    log.info(f"will_acquire=1: {pos:,} ({pos/len(propensity_df)*100:.1f}%)")
    log.info(f"will_acquire=0: {neg:,} ({neg/len(propensity_df)*100:.1f}%)")

    return propensity_df


def train_propensity_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.Booster:

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = min(neg / pos, 50)
    log.info(f"Propensity ratio — neg: {neg:,} | pos: {pos:,} | weight: {scale_pos_weight:.1f}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "scale_pos_weight": scale_pos_weight,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        train_set=train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    log.info(f"Best iteration: {model.best_iteration}")
    log.info(f"Best AUC: {model.best_score['valid_0']['auc']:.4f}")
    return model



def evaluate_propensity(
    model: lgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Evaluate propensity model and find optimal threshold.
    Returns the threshold that maximizes F1 on validation set.
    """
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred_proba)
    log.info(f"Propensity AUC: {auc:.4f}")

    best_f1, best_threshold = 0, 0.5
    for threshold in np.arange(0.01, 0.5, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        if y_pred.sum() == 0:
            continue
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall    = recall_score(y_val, y_pred, zero_division=0)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_pred_best = (y_pred_proba >= best_threshold).astype(int)
    log.info(f"Optimal threshold : {best_threshold:.2f}")
    log.info(f"F1 at threshold : {best_f1:.4f}")
    log.info(f"Precision : {precision_score(y_val, y_pred_best):.4f}")
    log.info(f"Recall : {recall_score(y_val, y_pred_best):.4f}")
    log.info(f"Customers flagged : {y_pred_best.sum():,} of {len(y_pred_best):,}")

    return best_threshold


def baseline_propensity(df: pd.DataFrame, top_pct: float = 0.20) -> pd.Series:
    """
    Rule-based baseline for propensity.
    Score = total_new_products_3m × active_customer × log(seniority_months + 1)
    Flag top_pct% of customers as will_acquire = 1.
    Returns Series with customer_id as index and prediction as value.
    """
    prop_df = df.drop_duplicates(subset=["customer_id"]).copy()

    prop_df["propensity_score"] = (
        prop_df["total_new_products_3m"]
        * prop_df["active_customer"]
        * np.log1p(prop_df["seniority_months"].fillna(0))
    )

    threshold = prop_df["propensity_score"].quantile(1 - top_pct)
    prop_df["baseline_pred"] = (prop_df["propensity_score"] >= threshold).astype(int)

    return prop_df.set_index("customer_id")["baseline_pred"]


if __name__ == "__main__":

    engine = create_engine(DB_URL, pool_pre_ping=True)

    log.info("Loading dataset...")
    df = pd.read_parquet(PARQUET_PATH)

    for col in ["new_customer", "residence_index", "active_customer"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["segment"] = df["segment"].fillna("UNKNOWN").astype("category")
    df["sex"] = df["sex"].fillna("UNKNOWN").astype("category")
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
    df["gross_income"] = df["gross_income"].fillna(df["gross_income"].median())

    propensity_df = build_propensity_dataset(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    X = propensity_df[FEATURES]
    y = propensity_df["will_acquire"].astype(int)
    groups = propensity_df["customer_id"]

    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    for col in X_train.select_dtypes(include=["category", "object"]).columns:
        X_train[col] = pd.Categorical(X_train[col]).codes
        X_val[col]   = pd.Categorical(X_val[col]).codes
        
    log.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

    # Baseline propensity evaluation
    log.info("Evaluating rule-based baseline...")
    propensity_val = propensity_df.iloc[val_idx].reset_index(drop=True)

    baseline_preds = baseline_propensity(propensity_val, top_pct=0.20)
    baseline_preds = baseline_preds.reindex(propensity_val["customer_id"]).fillna(0).values

    baseline_precision = precision_score(y_val, baseline_preds, zero_division=0)
    baseline_recall = recall_score(y_val, baseline_preds, zero_division=0)
    baseline_f1 = 2 * baseline_precision * baseline_recall / (baseline_precision + baseline_recall + 1e-9)

    log.info(f"Baseline Precision: {baseline_precision:.4f}")
    log.info(f"Baseline Recall: {baseline_recall:.4f}")
    log.info(f"Baseline F1: {baseline_f1:.4f}")
    log.info(f"Baseline customers flagged: {int(baseline_preds.sum()):,}")

    model = train_propensity_model(X_train, y_train, X_val, y_val)

    threshold = evaluate_propensity(model, X_val, y_val)

    model.save_model("data/processed/propensity_model.txt")
    pd.Series({"threshold": threshold}).to_json(
        "data/processed/propensity_threshold.json"
    )
    log.info("Propensity model saved.")
    log.info(f"Use threshold={threshold:.2f} to filter customers before ranking.")