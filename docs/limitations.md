# Known V1 limitations

What V1 deliberately does not do, or does only partially. Each item links the issue or page
that tracks it. The [CHANGELOG](../CHANGELOG.md) links this page from its
"Known limitations" section.

## Installation and packaging

- **Runs from a source checkout only.** The wheel ships the `invariantlab` library and CLI;
  `tasks/` and `configs/` are read from a `git clone`, relative to the current directory.
  `invariantlab run` outside a checkout exits with "run InvariantLab from a source checkout
  (tasks/ not found)". Packaging tasks/configs is deferred to post-V1; see
  [Releasing](releasing.md#packaging-decision-for-v1).
- **Windows and vLLM.** The former `models` extra (vllm/torch) was removed because vllm had no
  Windows wheel (`uv sync --all-extras` failed on Windows); vLLM is now used only as an
  external OpenAI-compatible server through the `openai_compatible` adapter, so it must run
  on a platform vLLM supports (Linux, or WSL on Windows). Install with
  `uv sync --extra dev`; see [Model execution](local-models.md).

## Evaluation harness

- **Single-machine Docker sandbox.** Every candidate is evaluated in a local Docker container
  (`--network none`, 256 MB memory, 1 CPU, 64 pids, read-only root, 60 s timeout; see
  [Methodology](methodology.md#evaluation-protocol)). There is no distributed or remote
  execution, `invariantlab run` needs a working Docker daemon on the same machine, and the
  container limits are a resource and network boundary that has not been reviewed as a
  hardened security sandbox.
- **Oscillator-only repair-study harness.** The repair runner evaluates repairs through the
  legacy oscillator study harness (`tasks/oscillator/candidate_runner.py` + `verifier.py`),
  so repair studies run on the oscillator only and the `update-order` / `sign-error` mutants
  are `interface: legacy_study`. Evaluating candidates through the M2 task packages for any
  registered mutant is tracked in
  [#120](https://github.com/akurkar07/InvariantLab/issues/120); until then `task_suite` is
  ignored.
- **Mutant coverage.** Package mutants exist for `oscillator`, `kepler` and `wave1d`; `heat1d`
  has none yet. Covered families: `sign_error`, `update_order_error`,
  `non_conservative_update`, `unit_error`, `hard_coded_shortcut`, `precision_defect`,
  `termination_defect`, `discretisation_error`, `boundary_error` and `stability_error`.
- **Reporting.** The static `report.html` is the V1 dashboard: no plots, interactive
  filtering or served dashboard.

## Study reproducibility

Status from [#100](https://github.com/akurkar07/InvariantLab/issues/100), which is still
open; see the [Research Track](research-track.md) study table.

- **Study 1** is not reproducible from the repository: no Study 1 experiment config or prompt
  is committed and its `hardened` condition does not exist in the runner.
- **Study 2** configs are committed (the DeepSeek configs were reconstructed) and protocol
  deviations are recorded in [Update-order feedback replication](update-order-feedback-replication.md),
  but its audited run directories are not published, so the reported aggregates cannot yet be
  rebuilt from the repository.
