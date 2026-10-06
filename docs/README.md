# InvariantLab Documentation

The repository [README](../README.md) gives the implementation status and a quickstart.

## Getting started

- [Local Models](local-models.md) — Running experiments against local model servers (Ollama, vLLM, llama.cpp)
- [First Model Experiment](first-model-experiment.md) — Repair-runner configuration and how it maps onto Study 1

## Authoring

- [Task Authoring](task-authoring.md) — Task package layout, subprocess protocol, NPZ output contract and trust boundary
- [Experiment Authoring](experiment-authoring.md) — Config-driven tasks, mutations, models and repair studies

## Design

- [Methodology](methodology.md) — V1 design target: benchmark design, evaluation protocol, metrics, scientific evidence, contracts and run outputs
- [Verification](verification.md) — Layers 0-6 (seven layers): status table, per-task gates and thresholds, trust boundary and `invariantlab verify`

## Reference

- [Model Adapters](model-adapters.md) — Adapter interface, supported backends, config keys and adding a backend
- [V1 Acceptance Criteria Map](v1-acceptance.md) — Which tests prove each V1 acceptance criterion and how to run the release gate
- [Releasing](releasing.md) — Release procedure, hatch-vcs versioning and the V1 run-from-checkout packaging decision
- [Known Limitations](limitations.md) — What V1 does not do yet: checkout-only install, single-machine Docker sandbox, oscillator-only study harness, study reproducibility
- [Changelog](../CHANGELOG.md) — What V1 contains, by pull request

## Study reports

- [First Multi-Condition Study Results](first-multi-condition-study-results.md) — Study 1 results
- [Update-Order Feedback Replication](update-order-feedback-replication.md) — Study 2 protocol
- [Study 2: Two-Model Comparison](study-two-model-comparison.md) — Study 2 results
- [Research Track](research-track.md) — How Studies 1-3 relate to the V1 verification-gap question; Study 3 preregistration rules

Development scripts are described in [scripts/README.md](../scripts/README.md).
