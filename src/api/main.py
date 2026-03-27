# =============================================================
# Project : NBO Recommendation Engine
# Script  : src/api/main.py
# Purpose : FastAPI serving layer — returns pre-computed
#           top-5 product recommendations per customer.
# =============================================================

import os
import logging
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------
# Database connection
# -----------------------------------------------------------
DB_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
engine = create_engine(DB_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)

# -----------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------
app = FastAPI(
    title="NBO Recommendation Engine",
    description="Next Best Offer — returns top-5 financial product recommendations per customer.",
    version="1.0.0"
)

# -----------------------------------------------------------
# Response schemas
# -----------------------------------------------------------
class ProductRecommendation(BaseModel):
    rank: int
    product_name: str
    score: float
    category: Optional[str] = None

class RecommendationResponse(BaseModel):
    customer_id: int
    recommendations: List[ProductRecommendation]
    scored_at: Optional[str] = None
    model_version: Optional[str] = None
    total_results: int

class HealthResponse(BaseModel):
    status: str
    db: str
    scores: int

# -----------------------------------------------------------
# Endpoints
# -----------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health and database connectivity."""
    try:
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM recommendation_scores")
            ).scalar()
        return HealthResponse(status="ok", db="connected", scores=count)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def get_recommendations(customer_id: int, top_k: int = 5):
    """
    Return top-K product recommendations for a customer.
    Scores are pre-computed by the daily batch scoring pipeline.
    """
    query = text("""
        SELECT
            rs.rank,
            rs.product_name,
            rs.score,
            pc.product_category  AS category,
            rs.scored_at,
            rs.model_version
        FROM recommendation_scores rs
        LEFT JOIN product_catalog pc
            ON pc.product_name = rs.product_name
        WHERE rs.customer_id = :customer_id
          AND rs.rank <= :top_k
        ORDER BY rs.rank ASC
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"customer_id": customer_id, "top_k": top_k}
        ).fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for customer_id={customer_id}. "
                   f"Customer may not exist or was filtered by propensity model."
        )

    recommendations = [
        ProductRecommendation(
            rank = row.rank,
            product_name = row.product_name,
            score = float(row.score),
            category = row.category,
        )
        for row in rows
    ]

    return RecommendationResponse(
        customer_id = customer_id,
        recommendations = recommendations,
        scored_at = str(rows[0].scored_at),
        model_version = rows[0].model_version,
        total_results = len(recommendations)
    )


@app.get("/customers/{customer_id}/profile")
def get_customer_profile(customer_id: int):
    """
    Return customer profile and current products.
    Useful for understanding why certain products were recommended.
    """
    query = text("""
        SELECT
            c.customer_id,
            c.country,
            c.sex,
            cs.age,
            cs.segment,
            cs.gross_income,
            cs.seniority_months,
            cs.active_customer
        FROM customers c
        INNER JOIN customer_snapshots cs
            ON cs.customer_id = c.customer_id
        WHERE c.customer_id = :customer_id
          AND cs.snapshot_date = '2016-04-28'
    """)

    with engine.connect() as conn:
        row = conn.execute(query, {"customer_id": customer_id}).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")

    return {
        "customer_id": row.customer_id,
        "country": row.country,
        "sex": row.sex,
        "age": row.age,
        "segment": row.segment,
        "gross_income": float(row.gross_income) if row.gross_income else None,
        "seniority_months": row.seniority_months,
        "active_customer": row.active_customer,
    }