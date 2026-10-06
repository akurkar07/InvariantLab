# InvariantLab development scripts

Run every script from the repository root inside the uv environment
(`uv sync --extra dev`).

## `validate_task.py`

Validates the fixed V1 task set (`oscillator`, `kepler`, `heat1d`, `wave1d`): each
package must exist with a `contract.yaml` that passes the schema, the on-disk artifacts
the contract declares, and a `specification.md`. CI runs it as the `validate` check.

```bash
uv run python scripts/validate_task.py --task-dir tasks/
```

Exit codes:

- `0` — all task contracts and artifacts are valid.
- `1` — at least one validation error; each error is printed to stderr.
- `2` — invalid command line (for example, missing `--task-dir`).

## `validate_mutants.py`

Validates every registered task mutant against its reference implementation and declared
behavior. The optional `--task` argument limits validation to one task directory.

```bash
uv run python scripts/validate_mutants.py --task-dir tasks/ [--task <name>]
```

Exit codes:

- `0` — every mutant and reference passed.
- `1` — any validation failure, registry error, missing task directory, or zero mutants.
- `2` — invalid command line.

## `check_v1_acceptance.py`

Fail-closed release gate for the V1 acceptance criteria. It runs pytest, keeps only tests
marked `@pytest.mark.v1_acceptance("V1-ACn")`, and prints a status for each criterion
`V1-AC1`..`V1-AC8`. See [docs/v1-acceptance.md](../docs/v1-acceptance.md) for the
criteria-to-tests map.

```bash
uv run python scripts/check_v1_acceptance.py --report v1.json [PATH ...]
```

`PATH` defaults to the repository `tests/` tree; `--report` writes a JSON report.

Exit codes:

- `0` — every criterion passed, no unknown or invalid marker ids, and pytest exited cleanly.
- `1` — any criterion is failed, skipped or missing, an unknown or invalid id was seen,
  or pytest failed.
- `2` — invalid command line.

## `check_release_metadata.py`

Checks that the release tag agrees with `CITATION.cff`, `CHANGELOG.md`, and (for final
releases) the development-status classifier in `pyproject.toml`. It also validates the CFF
schema with `cffconvert`.

```bash
uv run python scripts/check_release_metadata.py --tag v1.0.0
uv run python scripts/check_release_metadata.py --tag v1.0.0-rc.1 --notes-output release-notes.md
```

Exit codes:

- `0` — release metadata is consistent.
- `1` — a metadata or CFF validation check failed.
- `2` — invalid command line.
