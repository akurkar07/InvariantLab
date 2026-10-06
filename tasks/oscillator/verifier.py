import json
import math
import sys

CASES = [
    (1.0, 0.0, 1.0, 0.01, 800),
    (0.3, -0.4, 1.7, 0.005, 1200),
    (-0.8, 0.25, 0.7, 0.01, 900),
]


def analytic(t, x0, v0, omega):
    return (
        x0 * math.cos(omega * t) + (v0 / omega) * math.sin(omega * t),
        -x0 * omega * math.sin(omega * t) + v0 * math.cos(omega * t),
    )


def energy(x, v, omega):
    return 0.5 * v * v + 0.5 * omega * omega * x * x


def _is_real_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_cases(cases):
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise RuntimeError(f"case trajectories must be a list of length {len(CASES)}")
    for traj in cases:
        if (
            not isinstance(traj, list)
            or not traj
            or not all(
                isinstance(row, list)
                and len(row) == 3
                and all(_is_real_number(value) for value in row)
                for row in traj
            )
        ):
            raise RuntimeError(
                "each case trajectory must be a non-empty list of [t, x, v] rows of finite reals"
            )


with open(sys.argv[2], encoding="utf-8") as handle:
    THRESHOLDS = json.load(handle)
STATE_THRESHOLD = float(THRESHOLDS["max_state_relative_error"])
ENERGY_THRESHOLD = float(THRESHOLDS["max_energy_relative_drift"])

public = {}
scientific = {}

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("trajectory payload must be a JSON object")
    if "short" not in data:
        raise RuntimeError(data.get("error", "missing short trajectory"))
    if "cases" not in data:
        raise RuntimeError(data.get("error", "missing case trajectories"))

    short = data["short"]
    _validate_cases(data["cases"])

    public["shape"] = (
        isinstance(short, list)
        and len(short) == 3
        and all(len(row) == 3 for row in short)
    )
    public["initial_state"] = public["shape"] and all(
        abs(a - b) < 1e-12 for a, b in zip(short[0], (0.0, 1.0, 0.0), strict=True)
    )
    public["finite"] = public["shape"] and all(
        math.isfinite(float(value)) for row in short for value in row
    )
    public["one_step_sanity"] = (
        public["shape"] and abs(float(short[1][1]) - math.cos(0.05)) < 0.01
    )

    max_state_error = 0.0
    max_energy_drift = 0.0
    for traj, (x0, v0, omega, _dt, _n_steps) in zip(data["cases"], CASES, strict=True):
        t, x, v = map(float, traj[-1])
        x_ref, v_ref = analytic(t, x0, v0, omega)
        scale = max(1.0, abs(x_ref), abs(v_ref))
        max_state_error = max(
            max_state_error,
            math.hypot(x - x_ref, v - v_ref) / scale,
        )
        e0 = energy(float(traj[0][1]), float(traj[0][2]), omega)
        drift = max(
            abs(energy(float(row[1]), float(row[2]), omega) - e0)
            / max(abs(e0), 1e-15)
            for row in traj
        )
        max_energy_drift = max(max_energy_drift, drift)

    scientific["analytical_state"] = max_state_error < STATE_THRESHOLD
    scientific["energy_invariant"] = max_energy_drift < ENERGY_THRESHOLD
    metrics = {
        "max_state_relative_error": max_state_error,
        "max_energy_relative_drift": max_energy_drift,
    }
except Exception as exc:
    public.setdefault("execution", False)
    metrics = {"error": repr(exc)}

result = {
    "public": public,
    "scientific": scientific,
    "public_passed": bool(public) and all(public.values()),
    "scientific_passed": bool(scientific) and all(scientific.values()),
    "metrics": metrics,
}
print(json.dumps(result, sort_keys=True))
