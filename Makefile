.PHONY: help up down ingest train serve test

help:
	@echo "Commands:"
	@echo "  make up       → start PostgreSQL + MLflow (Docker)"
	@echo "  make down     → stop all containers"
	@echo "  make ingest   → load raw CSV into PostgreSQL"
	@echo "  make train    → train baseline + LightGBM model"
	@echo "  make serve    → start FastAPI recommendation endpoint"
	@echo "  make test     → run unit tests"

up:
	docker compose -f docker/docker-compose.yml --env-file .env up -d
	@echo "PostgreSQL ready at localhost:5432"
	@echo "MLflow UI ready at http://localhost:5001"

down:
	docker compose -f docker/docker-compose.yml down

ingest:
	python src/ingestion/load_raw.py

train:
	python src/models/baseline.py
	python src/models/lgbm_ranker.py

serve:
	uvicorn src.api.main:app --reload --port 8000

test:
	pytest tests/ -v