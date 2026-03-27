# =============================================================
# NBO Recommendation Engine — API
# Base image: Python 3.11 slim
# =============================================================

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/processed/lgbm_model.txt ./data/processed/
COPY data/processed/propensity_model.txt ./data/processed/
COPY data/processed/propensity_threshold.json ./data/processed/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]