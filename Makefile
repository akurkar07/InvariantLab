# InvariantLab — Makefile

.PHONY: help install install-dev lint format test test-unit test-acceptance typecheck coverage clean validate-tasks

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install with development dependencies
	pip install -e ".[dev]"
	pre-commit install

lint: ## Run Ruff over active benchmark code
	ruff check src/ tests/ tasks/ scripts/validate_task.py

format: ## Format active benchmark code
	ruff format src/ tests/ tasks/ scripts/validate_task.py

typecheck: ## Run mypy
	mypy src/invariantlab/

test: ## Run top-level tests
	pytest tests/ -v

test-unit: ## Run unit tests
	pytest tests/unit/ -v

test-acceptance: ## Run acceptance tests
	pytest tests/acceptance/ -v

coverage: ## Run top-level tests with coverage
	pytest tests/ --cov=invariantlab --cov-report=term-missing --cov-report=html

validate-tasks: ## Validate all four V1 task packages
	python scripts/validate_task.py --task-dir tasks/

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
