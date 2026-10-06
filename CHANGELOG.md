# Changelog

All notable changes to InvariantLab are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versions come
from git tags via hatch-vcs; see [docs/releasing.md](docs/releasing.md).

## [Unreleased]

## [1.0.0] - 2026-10-06

First release (V1). Numbers are pull requests on
<https://github.com/akurkar07/InvariantLab>.

### Added

- **Release workflow.** Tag-triggered `.github/workflows/release.yml` (with a `workflow_dispatch`
  dry run) and `scripts/check_release_metadata.py`, which checks the tag against `CITATION.cff`,
  `CHANGELOG.md` and the development-status classifier (#180).
- **Task packages.** Four self-contained tasks behind one subprocess/NPZ boundary, each with
  `contract.yaml`, `specification.md`, `src/solver.py`, committed example inputs and public
  and scientific test suites: `oscillator` (#23), `kepler` (#25, with the independent DOP853
  oracle #24), `heat1d` (#26) and `wave1d` (#27, CFL fix #22). Task/verifier boundaries
  (#21), complete contracts (#41), output-based scientific checks (#42) and artifact/path
  validation (#30). Landed on `main` with fail-closed validation in #150; example inputs
  #160; trusted heat references honour domain length (#145); all four suites run in CI (#136).
- **Verification layers.** Reusable gates in `invariantlab.verification`: Layer 0 execution
  and archive schema (`run_task`, #138), Layer 2 oracle comparison (`check_oracle`, #147),
  Layer 3 physical invariants (`check_invariants`, #148), Layer 4 observed-order convergence
  through the CLI/NPZ boundary (`check_convergence`, #166), Layer 5 metamorphic relations
  (`check_metamorphic`, #149) and Layer 6 held-out robustness cases (`check_robustness`,
  #163), composed into one `VerificationResult` by `verify_candidate` / `invariantlab verify`
  (#171). Agent workspaces are built without hidden tests and candidates are evaluated against
  trusted task files (#137); the oscillator study verifier no longer trusts candidate stdout
  (#129) and its thresholds are single-sourced from `task.yaml` (#134).
- **Mutants.** Mutant manifest and registry (`invariantlab.mutations.discover_mutants`,
  #139); validation that each package mutant passes the public tests and fails its declared
  scientific gate (#158); the curated mutants `oscillator/non-conservative-damping`,
  `wave1d/sign-error-startup` and `wave1d/update-order-overwrite`, plus the legacy
  oscillator `update-order` and `sign-error` study mutants (#165); the oscillator
  `hard-coded-fixture-shortcut`, `float32-position` and `early-termination` mutants (#175);
  wave1d `courant-not-squared`, `dirichlet-wrong-node` and `unstable-time-recurrence`
  mutants (#179); Kepler `unit-error-au-rounding` and `non-conservative-velocity-damping`
  (#172);
  heat1d `discretisation-grid-spacing` and `sixth-order-stencil-ftcs-limit` mutants (#178).
- **Repair runner and replay.** Generic task/mutation/repair experiment runner with Docker
  sandboxed evaluation (#53) and resilient, resumable provider runs (#46); local-first model
  configs (#55). The `replay` adapter replays recorded responses by prompt hash and the fixed
  solver is the `reference_stub` adapter (#142); supported backends are `reference_stub`,
  `replay`, `ollama` and `openai_compatible` (#135), all returning token usage (#125). Run
  manifest with pinned image digest and provenance (#141); timeouts and infrastructure
  errors pause instead of crashing (#127); public pass rate and verification gap per
  condition (#123); `first_model` runner and dead config retired, `--dry-run` checks what a
  run needs (#144); `audit-run` resolves the model id like the runner (#130); ASCII CLI status
  markers (#131). Package-task repair runs evaluate mutants through `invariantlab verify`
  with candidate code isolated in Docker (#120). Docker-free runner integration test (#152),
  repeated replay yields identical evaluator outcomes (V1-AC4, #156) and a real replay smoke
  run is reproduced in CI (#167).
- **Reports and export.** `invariantlab report` rebuilds summary and CSV tables from
  `events.jsonl` (#140), with an optional static `report.html` (#155); `invariantlab export-hf`
  writes a local Hugging Face-loadable dataset with per-record provenance and a credential scan
  (#153). Report totals match sample records and tables rebuild offline (V1-AC5/AC6, #159);
  the exported dataset has full provenance and no credentials (V1-AC8, #164).
- **Acceptance, CI and docs.** V1 acceptance-criteria map and fail-closed release gate
  `scripts/check_v1_acceptance.py` (#133); oracle/agent code separation proved (V1-AC3,
  #157); CI installs from `uv.lock`, checks formatting and enforces a coverage ratchet (#154);
  Makefile through uv plus pre-commit config (#168); schema-valid `CITATION.cff` (#122);
  status-first README (#151), docs index (#161), research-track roadmap (#162), Study 2
  corrections (#124, #128) and the documented evaluation protocol (#132).
- Mutant catalogue and validation documentation, plus an acceptance test requiring package
  mutant coverage for all ten mutation families or a documented infeasibility issue (issue #93).
- This changelog and the [release procedure](docs/releasing.md).

### Changed

- `main` and `develop` were reconciled into a single integration branch, `main` (#126, for
  issue #94); `develop` is retired and CI checks are required on `main` (#146).
- `invariantlab run` exits with "run InvariantLab from a source checkout (tasks/ not found)"
  instead of a traceback when started outside a source checkout.

### Known limitations

See [docs/limitations.md](docs/limitations.md). In short: V1 runs from a source checkout, the
Docker sandbox is single-machine, the legacy oscillator `update-order` / `sign-error` study
mutants are still graded by the oscillator study harness, and the Study 1/2 results are not
yet reproducible from the repository
([#100](https://github.com/akurkar07/InvariantLab/issues/100)).

[Unreleased]: https://github.com/akurkar07/InvariantLab/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/akurkar07/InvariantLab/releases/tag/v1.0.0
