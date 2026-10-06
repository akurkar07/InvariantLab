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
| V1-AC1 | all reference implementations pass every scientific gate | M2, M3 | [#63](https://github.com/akurkar07/InvariantLab/issues/63) | `tests/acceptance/test_v1_reference_gates.py` (whole module) |
| V1-AC2 | every controlled mutant passes its designated weak profile and fails its expected scientific gate | M3, M4 | [#64](https://github.com/akurkar07/InvariantLab/issues/64) | `tests/acceptance/test_v1_mutants.py::test_mutation_registry_is_non_empty`, `tests/acceptance/test_v1_mutants.py::test_mutant_passes_weak_profile_and_fails_expected_gate` (one case per registered package mutant) |
| V1-AC3 | independent oracle and agent-facing code paths share no numerical update implementation | M2, M3 | [#65](https://github.com/akurkar07/InvariantLab/issues/65) | `tests/acceptance/test_v1_oracle_separation.py` (whole module)<br>`tests/acceptance/test_task_boundaries.py` (whole module) |
| V1-AC4 | repeated replay produces identical evaluator outcomes | M5 | [#66](https://github.com/akurkar07/InvariantLab/issues/66) | `tests/integration/test_v1_replay_determinism.py::test_repeated_replay_yields_identical_evaluator_outcomes`, `tests/integration/test_v1_replay_determinism.py::test_perturbed_replay_response_changes_evaluator_outcomes` |
| V1-AC5 | report totals equal the number of enumerated sample records | M5, M6 | [#67](https://github.com/akurkar07/InvariantLab/issues/67) | `tests/acceptance/test_v1_report_reconstruction.py::test_report_totals_equal_enumerated_sample_records` |
| V1-AC6 | result tables can be regenerated without API access | M6 | [#67](https://github.com/akurkar07/InvariantLab/issues/67) | `tests/acceptance/test_v1_report_reconstruction.py::test_report_tables_regenerate_offline_byte_for_byte` |
| V1-AC7 | CI exercises task validation, a complete smoke run and report reconstruction | M6 | [#69](https://github.com/akurkar07/InvariantLab/issues/69) | _pending (#69)_ |
| V1-AC8 | the public dataset contains task metadata, trajectories, patches, measurements and provenance without hidden credentials | M6 | [#68](https://github.com/akurkar07/InvariantLab/issues/68) | `tests/acceptance/test_v1_dataset_export.py::test_exported_records_have_full_provenance`, `tests/acceptance/test_v1_dataset_export.py::test_exported_dataset_passes_credential_scan`, `tests/acceptance/test_v1_dataset_export.py::test_credential_scan_fails_on_injected_canary` |

V1-AC4 requires a working Docker daemon and fails rather than skips when Docker is unavailable.

V1-AC2 is parametrised over the mutation registry (`discover_mutants`) at collection time and
prints the number of mutants per task. Mutants with `interface: legacy_study` are excluded and
reported as skipped by `test_legacy_study_mutants_are_excluded`, which is not marked for V1-AC2;
they are graded by the legacy verifier until [#120](https://github.com/akurkar07/InvariantLab/issues/120).
