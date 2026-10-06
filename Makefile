# InvariantLab — Makefile (targets mirror .github/workflows/tests.yml and task-validation.yml)

.PHONY: help install install-dev lint format typecheck test test-unit test-acceptance test-tasks coverage validate-tasks clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies from uv.lock
	uv sync --locked

install-dev: ## Install development dependencies from uv.lock and the pre-commit hooks
	uv sync --locked --extra dev
	uv run pre-commit install

lint: ## Run the CI lint checks
	uv lock --check
	uv run ruff check src tests tasks scripts
	uv run ruff format --check src tests tasks scripts

format: ## Format code with Ruff
	uv run ruff format src tests tasks scripts

typecheck: ## Run mypy
	uv run mypy src/invariantlab/

test: ## Run top-level tests (excluding acceptance)
	uv run pytest tests/ -v --ignore=tests/acceptance

test-unit: ## Run unit tests
	uv run pytest tests/unit/ -v

test-acceptance: ## Run acceptance tests
	uv run pytest tests/acceptance/ -v

test-tasks: ## Run each task package suite in its own directory
	for t in oscillator kepler heat1d wave1d; do (cd tasks/$$t && uv run pytest tests -q) || exit 1; done

coverage: ## Run top-level tests with coverage (fail_under from pyproject.toml)
	uv run pytest tests --cov=invariantlab --cov-report=term -v --ignore=tests/acceptance

validate-tasks: ## Validate all four V1 task packages
	uv run python scripts/validate_task.py --task-dir tasks/

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
