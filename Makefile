# InvariantLab — Makefile

.PHONY: help install install-dev lint format test test-unit test-property test-integration test-acceptance typecheck coverage clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install with dev dependencies
	pip install -e ".[dev,models,export]"
	pre-commit install

lint: ## Run ruff linter
	ruff check src/ tests/

format: ## Format code with ruff
	ruff format src/ tests/

typecheck: ## Run mypy type checking
	mypy src/invariantlab/

test: ## Run full test suite
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-property: ## Run property tests only
	pytest tests/property/ -v -m "not slow"

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-acceptance: ## Run acceptance tests only
	pytest tests/acceptance/ -v

coverage: ## Run tests with coverage report
	pytest tests/ --cov=invariantlab --cov-report=term-missing --cov-report=html

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

validate-tasks: ## Validate all task contracts
	python scripts/validate_task.py --task-dir tasks/

validate-mutants: ## Validate all mutants
	python scripts/validate_mutants.py --task-dir tasks/

reproduce: ## Reproduce the smoke run
	python scripts/reproduce_report.py

export-hf: ## Export dataset to Hugging Face
	python scripts/export_hf_dataset.py
