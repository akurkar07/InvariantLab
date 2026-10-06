import importlib.util
import json
import sys

CASES = [
    (1.0, 0.0, 1.0, 0.01, 800),
    (0.3, -0.4, 1.7, 0.005, 1200),
    (-0.8, 0.25, 0.7, 0.01, 900),
]


def _serialise(traj):
    if not isinstance(traj, list):
        return None
    return [[float(v) for v in row] for row in traj]


def main():
    solver_path, output_path = sys.argv[1], sys.argv[2]
    payload = {}
    try:
        spec = importlib.util.spec_from_file_location("candidate", solver_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solve = module.solve_oscillator_verlet
        payload["short"] = _serialise(solve(1.0, 0.0, 1.0, 0.05, 2))
        payload["cases"] = [_serialise(solve(*case)) for case in CASES]
    except Exception as exc:
        payload["error"] = repr(exc)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))


if __name__ == "__main__":
    main()
