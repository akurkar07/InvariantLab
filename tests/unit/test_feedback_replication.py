"""Tests for the update-order feedback replication study."""

from invariantlab.experiments.feedback_replication import (
    CONDITIONS,
    _build_schedule,
    _condition_context,
    _severity_ratios,
)


BASELINE = {
    "metrics": {
        "max_state_relative_error": 0.021,
        "max_energy_relative_drift": 0.046,
    }
}


def test_schedule_is_balanced_and_reproducible():
    first = _build_schedule(list(CONDITIONS), 30, 1729, True)
    second = _build_schedule(list(CONDITIONS), 30, 1729, True)

    assert first == second
    assert len(first) == 120
    for condition in CONDITIONS:
        assert sum(cell_condition == condition for cell_condition, _ in first) == 30


def test_schedule_randomisation_changes_grouped_order():
    grouped = _build_schedule(list(CONDITIONS), 3, 1729, False)
    randomised = _build_schedule(list(CONDITIONS), 3, 1729, True)

    assert grouped != randomised
    assert sorted(grouped) == sorted(randomised)


def test_feedback_conditions_are_distinct():
    contexts = {
        condition: _condition_context(condition, BASELINE)
        for condition in CONDITIONS
    }

    assert contexts["weak"] == ""
    assert "0.021" not in contexts["placebo"]
    assert "0.021" in contexts["metrics"]
    assert "acceptance threshold" not in contexts["metrics"]
    assert "acceptance threshold" in contexts["interpreted"]


def test_severity_ratios_capture_worse_repair():
    repaired = {
        "metrics": {
            "max_state_relative_error": 1.97,
            "max_energy_relative_drift": 8.0,
        }
    }

    ratios = _severity_ratios(BASELINE, repaired)

    assert ratios["state_error_ratio"] == 1.97 / 0.021
    assert ratios["energy_drift_ratio"] == 8.0 / 0.046
    assert ratios["worst_scientific_ratio"] == 8.0 / 0.046
