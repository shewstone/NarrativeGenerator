# Narrative Engine Makefile

.PHONY: help install install-dev test test-unit test-integration lint format type-check clean db-up db-down migrate

# Default target
help:
	@echo "Narrative Engine - Available commands:"
	@echo ""
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make lint             - Run linters (ruff, black --check)"
	@echo "  make format           - Format code (black, ruff --fix)"
	@echo "  make type-check       - Run mypy type checker"
	@echo "  make clean            - Remove build artifacts"
	@echo "  make db-up            - Start PostgreSQL with docker-compose"
	@echo "  make db-down          - Stop PostgreSQL"
	@echo "  make migrate          - Run database migrations"
	@echo "  make migrate-create   - Create new migration (prompts for message)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,llm]"

# Testing
test:
	pytest -v --cov=narrative_engine --cov-report=term-missing

test-unit:
	pytest -v -m "not integration" --cov=narrative_engine --cov-report=term-missing

test-integration:
	pytest -v -m integration

test-coverage:
	pytest -v --cov=narrative_engine --cov-report=html --cov-report=term-missing
	@echo "Open htmlcov/index.html for detailed coverage report"

# Linting and formatting
lint:
	ruff check src tests
	black --check src tests

format:
	black src tests
	ruff check --fix src tests

type-check:
	mypy src/narrative_engine

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Database operations
db-up:
	docker-compose up -d postgres

db-down:
	docker-compose down

db-logs:
	docker-compose logs -f postgres

migrate:
	alembic upgrade head

migrate-create:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

migrate-down:
	alembic downgrade -1

# Development server
run-api:
	uvicorn narrative_engine.api:app --reload --port 8000

# Documentation
docs-serve:
	cd docs && mkdocs serve

docs-build:
	cd docs && mkdocs build
