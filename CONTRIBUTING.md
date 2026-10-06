# Contributing to InvariantLab

## Branches

- `main` is the only integration branch. There is no `develop` branch.
- Cut every branch from the latest `main` and open the pull request against `main`.
- Name branches after the kind of change: `feat/...`, `fix/...`, `docs/...` or `chore/...`
  (for example `fix/heat1d-boundary`). Automated agent branches use `devin/issue-<n>-<slug>`.
- Never push directly to `main` or force-push it. Merge through a pull request with green CI.

## Required checks

A pull request can merge into `main` only when these status checks pass on its head commit:

| Check | Workflow | What it runs |
| --- | --- | --- |
| `lint` | Tests | `uv lock --check`, `ruff check`, `ruff format --check` and `cffconvert --validate -i CITATION.cff` |
| `test (3.10)`, `test (3.11)`, `test (3.12)` | Tests | top-level `pytest tests/` (without `tests/acceptance`) |
| `task-tests (oscillator)`, `task-tests (kepler)`, `task-tests (heat1d)`, `task-tests (wave1d)` | Tests | each task suite in its own directory |
| `typecheck` | Tests | `mypy src/invariantlab/` |
| `acceptance` | Tests | `pytest tests/acceptance/` |
| `validate` | Task Validation | `python scripts/validate_task.py --task-dir tasks/`; `python scripts/validate_mutants.py --task-dir tasks/` |

## Local verification

Set up the environment (Python 3.10-3.12):

```bash
uv sync --extra dev        # or: pip install -e ".[dev]"
```

Run the same commands as CI before opening a pull request:

```bash
uv lock --check
uv run ruff check src tests tasks scripts
uv run ruff format --check src tests tasks scripts
uv run mypy src/invariantlab/
uv run pytest tests/ -q
uv run python scripts/validate_task.py --task-dir tasks/
uv run python scripts/validate_mutants.py --task-dir tasks/
```

Run each task suite from its own directory (`pytest tasks` fails because the suites share
test module names):

```bash
for t in oscillator kepler heat1d wave1d; do (cd tasks/$t && uv run pytest tests -q) || exit 1; done
```

`make lint`, `make typecheck`, `make test`, `make test-tasks` and `make validate-tasks` wrap
the same commands.
