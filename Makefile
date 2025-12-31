# --- Configuration ---
# Allow switching between podman-compose and docker-compose
# Usage: make run-platform COMPOSE=docker-compose
COMPOSE ?= podman-compose

# --- Production Commands ---
.PHONY: run-platform stop logs

# Starts the full platform in "Production Mode" (Detached)
# Explicitly uses -f compose.yaml to IGNORE any lab overrides
run-platform:
	$(COMPOSE) -f compose.yaml up --build -d
	@echo "Platform is running in background."
	@echo " - Dagster: http://localhost:3000"
	@echo " - Streamlit: http://localhost:8501"
	@echo " - MLflow: http://localhost:5000"

# Stops all services and removes networks
stop:
	$(COMPOSE) down

# Tail logs for all services (Press Ctrl+C to exit)
logs:
	$(COMPOSE) logs -f

# --- Development / Lab Commands ---
.PHONY: lab lab-stop

# Starts the platform in "Lab Mode" (Interactive)
# Merges compose.yaml (Base) + compose.lab.yaml (Overrides)
# model_training will run Jupyter instead of the gRPC server
lab:
	@echo "Starting Lab Environment..."
	@echo " - Dagster: http://localhost:3000"
	@echo " - Streamlit: http://localhost:8501"
	@echo " - MLflow: http://localhost:5000"
	@echo " - Jupyter: (Check logs below for URL/Token)"
	$(COMPOSE) -f compose.yaml -f compose.lab.yaml up --build
# Note: We run in foreground so you can see Jupyter URL/Token

# --- CI / Testing Commands ---
.PHONY: test test-all lint

# Run unit tests for a specific service
# Usage: make test service=model_training
test:
	cd services/$(service) && poetry run pytest

# Run all tests (Good for pre-commit checks)
test-all:
	@echo "Running Data Gen Tests..."
	cd services/data_gen && poetry run pytest
	@echo "Running ETL Tests..."
	cd services/etl && poetry run pytest
	@echo "Running Model Training Tests..."
	cd services/model_training && poetry run pytest

# Run containerized integration tests (Requires podman/docker and compose)
test-integration:
	@echo "Running containerized pipeline integration tests..."
	python3 -m unittest discover tests

# Check code style (Optional, requires ruff/flake8 installed)
lint:
	poetry run ruff check .

# --- Utilities ---
.PHONY: clean help

# Clean up pycache and artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

help:
	@echo "Retail Clustering Platform - Makefile"
	@echo "====================================="
	@echo "run-platform : Start full production pipeline (detached)"
	@echo "lab          : Start Lab Mode (Jupyter on port 8888)"
	@echo "stop         : Stop all containers"
	@echo "logs         : Tail logs"
	@echo "test         : Run tests for specific service (e.g., make test service=etl)"
	@echo "test-all     : Run all unit tests"
	@echo "test-integration : Run containerized end-to-end integration tests"