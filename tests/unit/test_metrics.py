"""Tests for repair-run pass-rate and verification-gap metrics."""

import pytest

from invariantlab.config import ExperimentConfig
from invariantlab.experiments.repair import _summary
from invariantlab.metrics import pass_rate, verification_gap, wilson_interval

BASELINE = {
    "public_passed": True,
    "scientific_passed": False,
    "metrics": {
        "max_state_relative_error": 0.021,
        "max_energy_relative_drift": 0.046,
    },
}


def test_wilson_interval_all_successes_matches_reference():
    interval = wilson_interval(30, 30)

    assert interval is not None
    assert round(interval[0], 3) == 0.886
    assert round(interval[1], 3) == 1.000


def test_empty_totals_yield_none():
    assert wilson_interval(0, 0) is None
    assert pass_rate(0, 0) is None
    assert verification_gap(None, 0.5) is None
    assert verification_gap(0.5, None) is None


def test_pass_rate_and_gap_are_hand_computed():
    assert pass_rate(3, 4) == 0.75
    assert verification_gap(0.75, 0.25) == 0.5


def _experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="metrics-test",
        model="configs/models/replay.yaml",
        runner="repair",
        mutation="update-order",
        conditions=["weak", "metrics", "interpreted"],
        n_attempts=4,
        seed=1729,
    )


def _record(condition: str, trial: int, public: bool, scientific: bool):
    return {
        "experiment": "metrics-test",
        "model": "test/model",
        "mutation": "update-order",
        "seed": 1729,
        "condition": condition,
        "trial": trial,
        "repaired": {"public_passed": public, "scientific_passed": scientific},
        "successful_repair": scientific,
        "scientific_regression": False,
        "severity": {"worst_scientific_ratio": 0.1},
    }


def test_summary_reports_public_rate_and_verification_gap():
    outcomes = {
        "weak": [(True, False), (True, False), (True, True), (False, False)],
        "metrics": [(True, True), (True, True), (True, True), (True, False)],
    }
    records = [
        _record(condition, trial, public, scientific)
        for condition, cells in outcomes.items()
        for trial, (public, scientific) in enumerate(cells, start=1)
    ]

    summary = _summary(_experiment(), "test/model", BASELINE, records)
    weak = summary["by_condition"]["weak"]
    metrics = summary["by_condition"]["metrics"]
    empty = summary["by_condition"]["interpreted"]

    assert summary["baseline_verification_gap"] == 1
    assert summary["primary_endpoint"] == "scientific_pass_rate"

    assert weak["completed"] == 4
    assert weak["public_passes"] == 3
    assert weak["public_pass_rate"] == 0.75
    assert weak["scientific_passes"] == 1
    assert weak["scientific_pass_rate"] == 0.25
    assert weak["verification_gap"] == pytest.approx(0.5)
    assert weak["public_pass_rate_wilson95"] == list(wilson_interval(3, 4) or ())
    assert weak["scientific_pass_rate_wilson95"] == list(wilson_interval(1, 4) or ())

    assert metrics["public_pass_rate"] == 1.0
    assert metrics["scientific_pass_rate"] == 0.75
    assert metrics["verification_gap"] == pytest.approx(0.25)
    assert metrics["pass_rate_difference_vs_weak"] == pytest.approx(0.5)

    assert empty["completed"] == 0
    assert empty["public_pass_rate"] is None
    assert empty["public_pass_rate_wilson95"] is None
    assert empty["scientific_pass_rate_wilson95"] is None
    assert empty["verification_gap"] is None


def test_failed_candidates_stay_in_denominator():
    records = [
        _record("weak", 1, True, True),
        _record("weak", 2, False, False),
    ]
    records[1]["repaired"] = {
        "public": {},
        "scientific": {},
        "public_passed": False,
        "scientific_passed": False,
        "metrics": {"error": "TimeoutError: candidate timed out"},
    }

    summary = _summary(_experiment(), "test/model", BASELINE, records)
    weak = summary["by_condition"]["weak"]

    assert weak["completed"] == 2
    assert weak["public_pass_rate"] == 0.5
    assert weak["scientific_pass_rate"] == 0.5
    assert weak["verification_gap"] == 0.0
