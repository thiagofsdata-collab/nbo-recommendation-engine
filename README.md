# NBO Recommendation Engine
### Next Best Offer — Financial Product Recommendation System

A production-grade machine learning system that recommends the top-5 financial products for each bank customer, built end-to-end from raw data to a serving API.

---

## Business Problem

Banks traditionally recommend products based on segment-level popularity — every VIP customer gets the same list, every individual customer gets the same list. This ignores individual behavior, purchase history, and financial profile.

This project builds a **personalized Next Best Offer (NBO) system** that predicts which financial products each customer is most likely to acquire next, enabling targeted campaigns with measurably higher conversion rates.

---

## Results

| Model | MAP@5 | vs Baseline |
|---|---|---|
| Popularity baseline (segment-level) | 0.0417 | — |
| Propensity + LightGBM ranking model | 0.0982 | **+135.8%** |

- **211,205 customers** scored daily
- **1,056,025 recommendations** pre-computed and served via REST API
- **3 of 9 features** flagged with data drift between Dec 2015 and Apr 2016

---

## Architecture
```
Raw Data (Kaggle)
      ↓
PostgreSQL (raw schema)
      ↓ SQL transforms (dbt-style)
Feature Store (mart tables)
      ↓ export to Parquet
Propensity Model (LightGBM) → filters 928k → 211k propense customers
      ↓
Ranking Model (LightGBM) → top-5 products per customer
      ↓
Batch Scoring (daily) → recommendation_scores table
      ↓
FastAPI → GET /recommend/{customer_id}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data storage | PostgreSQL 16 |
| Data format | Parquet (pyarrow) |
| Feature engineering | SQL (window functions, CTEs) |
| ML models | LightGBM |
| Experiment tracking | MLflow |
| Model explainability | SHAP |
| API | FastAPI + uvicorn |
| Monitoring | KS test + chi-square (scipy) |
| Containerization | Docker + docker-compose |
| Language | Python 3.11 |

---

## Dataset

[Santander Product Recommendation](https://www.kaggle.com/competitions/santander-product-recommendation) — Kaggle competition dataset.

- **928,274** unique customers
- **17 months** of monthly snapshots (Jan 2015 – May 2016)
- **24 financial products** (accounts, credit, investments)
- **13M+ rows** in raw format

---

## Statistical Concepts Applied

| Concept | Where applied |
|---|---|
| Conditional probability P(product\|customer) | Core ranking model objective |
| Bayes theorem | Popularity baseline — P(product\|segment) |
| Bernoulli trials | Binary target — acquired or not |
| Expected value | MAP@5 — expected precision across ranked list |
| Normal approximation | KS test drift detection |
| Binomial distribution | Propensity model target |

---

## Project Structure
```
nbo-recommendation-engine/
├── data/
│   ├── raw/                  # Kaggle CSV (not committed)
│   └── processed/            # Parquet, models, reports
├── sql/
│   ├── schema/               # DDL — CREATE TABLE
│   ├── transforms/           # Feature engineering SQL
│   └── marts/                # Final mart views
├── src/
│   ├── ingestion/            # CSV → PostgreSQL
│   ├── models/
│   │   ├── baseline.py       # Popularity model + MAP@5
│   │   ├── propensity_model.py
│   │   └── lgbm_ranker.py    # LightGBM + SHAP + MLflow
│   ├── api/
│   │   ├── main.py           # FastAPI endpoints
│   │   └── batch_scoring.py  # Daily scoring pipeline
│   └── monitoring/
│       └── drift_report.py   # KS + chi-square drift detection
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/
│   └── 01_eda_ingestion.ipynb
├── .env.example
├── Makefile
└── requirements.txt
```

---

## How to Run

### Prerequisites
- Docker Desktop
- Python 3.11+
- Kaggle account (to download dataset)

### 1. Clone and setup
```bash
git clone https://github.com/thiagofsdata-collab/nbo-recommendation-engine
cd nbo-recommendation-engine
cp .env.example .env
```

### 2. Start infrastructure
```bash
make up
# PostgreSQL ready at localhost:5432
# MLflow UI ready at http://localhost:5001
```

### 3. Download dataset

Download `train_ver2.csv` from [Kaggle](https://www.kaggle.com/competitions/santander-product-recommendation/data) and place in `data/raw/`.

### 4. Run pipeline
```bash
make ingest    # load raw CSV into PostgreSQL (~3h)
```

Run feature engineering:
```bash
# SQL transforms
Get-Content sql/transforms/01_feature_engineering.sql | docker exec -i nbo_postgres psql -U nbo_user -d nbo
```
```bash
make train     # train propensity + ranking models
```

### 5. Generate recommendations
```bash
python src/api/batch_scoring.py
```

### 6. Start API
```bash
make serve
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 7. Test
```bash
curl http://localhost:8000/health
curl http://localhost:8000/recommend/15889
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check + scores count |
| GET | `/recommend/{customer_id}` | Top-5 product recommendations |
| GET | `/customers/{customer_id}/profile` | Customer profile |
| GET | `/docs` | Interactive API documentation |

### Example response
```json
{
  "customer_id": 15889,
  "recommendations": [
    {"rank": 1, "product_name": "junior_account", "score": 0.4492, "category": "accounts"},
    {"rank": 2, "product_name": "home_account",   "score": 0.4492, "category": "accounts"},
    {"rank": 3, "product_name": "pension_plan",   "score": 0.4421, "category": "investments"},
    {"rank": 4, "product_name": "credit_card",    "score": 0.4380, "category": "credit"},
    {"rank": 5, "product_name": "funds",          "score": 0.4312, "category": "investments"}
  ],
  "scored_at": "2026-03-27 10:44:44",
  "model_version": "lgbm_v2_propensity",
  "total_results": 5
}
```

---

## Model Details

### Propensity Model
- **Objective**: Binary classification — will customer acquire any product?
- **Algorithm**: LightGBM
- **AUC**: 0.876
- **F1**: 0.191 vs baseline 0.058 (+229%)
- **Threshold**: 0.49 — filters 928k → 211k propense customers

### Ranking Model
- **Objective**: Binary classification — probability of acquiring each specific product
- **Algorithm**: LightGBM with scale_pos_weight
- **AUC**: 0.712
- **MAP@5**: 0.0982 vs baseline 0.0417 (+135.8%)
- **Top features** (SHAP): total_new_products_3m, has_current_account, seniority_months

---

## Monitoring

Data drift is detected weekly by comparing feature distributions between the training reference period (Dec 2015) and the current period using:
- **KS test** for numeric features (p < 0.05 = drift)
- **Chi-square test** for categorical features

Current status: **3/9 features with drift** (gross_income, seniority_months, age) — model retraining recommended.

---

## Author

**Thiago Silva**  
[GitHub](https://github.com/thiagofsdata-collab) · [LinkedIn](https://linkedin.com/in/seu-linkedin)