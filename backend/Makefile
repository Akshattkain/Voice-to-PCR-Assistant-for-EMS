.PHONY: run dev test lint format install

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=app --cov-report=html

lint:
	ruff check app/ tests/ evaluation/ training/

format:
	ruff format app/ tests/ evaluation/ training/
