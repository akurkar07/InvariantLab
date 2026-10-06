# V1 acceptance criteria map

The V1 acceptance criteria stated in the README are authoritative. Every
criterion must be proved by at least one test marked
`@pytest.mark.v1_acceptance("V1-ACn")` under `tests/`.

Check the gate with:

```bash
uv run python scripts/check_v1_acceptance.py --report v1.json
```

The checker runs pytest, keeps only `v1_acceptance`-marked tests, and reports
each criterion as `passed`, `failed`, `skipped` or `missing`. It exits 0 only
when all eight criteria are `passed`; skipped, xfailed, failed and missing
criteria — as well as tests marked with unknown acceptance ids — all fail the
gate. CI wiring for the checker is tracked in
[#69](https://github.com/akurkar07/InvariantLab/issues/69).

| ID | Criterion | Prerequisite milestones | Test issue | Pytest node id(s) |
| --- | --- | --- | --- | --- |
| V1-AC1 | all reference implementations pass every scientific gate | M2, M3 | [#63](https://github.com/akurkar07/InvariantLab/issues/63) | _pending (#63)_ |
| V1-AC2 | every controlled mutant passes its designated weak profile and fails its expected scientific gate | M3, M4 | [#64](https://github.com/akurkar07/InvariantLab/issues/64) | _pending (#64)_ |
| V1-AC3 | independent oracle and agent-facing code paths share no numerical update implementation | M2, M3 | [#65](https://github.com/akurkar07/InvariantLab/issues/65) | `tests/acceptance/test_v1_oracle_separation.py` (whole module)<br>`tests/acceptance/test_task_boundaries.py` (whole module) |
| V1-AC4 | repeated replay produces identical evaluator outcomes | M5 | [#66](https://github.com/akurkar07/InvariantLab/issues/66) | `tests/integration/test_v1_replay_determinism.py::test_repeated_replay_yields_identical_evaluator_outcomes`, `tests/integration/test_v1_replay_determinism.py::test_perturbed_replay_response_changes_evaluator_outcomes` |
| V1-AC5 | report totals equal the number of enumerated sample records | M5, M6 | [#67](https://github.com/akurkar07/InvariantLab/issues/67) | _pending (#67)_ |
| V1-AC6 | result tables can be regenerated without API access | M6 | [#67](https://github.com/akurkar07/InvariantLab/issues/67) | _pending (#67)_ |
| V1-AC7 | CI exercises task validation, a complete smoke run and report reconstruction | M6 | [#69](https://github.com/akurkar07/InvariantLab/issues/69) | _pending (#69)_ |
| V1-AC8 | the public dataset contains task metadata, trajectories, patches, measurements and provenance without hidden credentials | M6 | [#68](https://github.com/akurkar07/InvariantLab/issues/68) | _pending (#68)_ |

V1-AC4 requires a working Docker daemon and fails rather than skips when Docker is unavailable.
