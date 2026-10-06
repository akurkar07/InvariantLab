Four repair records written by the real `run_repair_experiment` (conditions `weak`, `metrics`; 2 trials;
seed 1729; no order randomisation) with a scripted model `fixture/scripted-oscillator` and a
content-based `_evaluate_source` (see `_content_based_evaluator` in `tests/unit/test_repair_experiment.py`).
The weak trial-1 response is the unchanged mutation (failing repair); the other three are the reference solver.

`experiment.yaml` replays only the weak responses through the real Docker evaluator because metrics-condition
prompts embed baseline metrics that differ from those in this fixture. It is used by
`tests/integration/test_v1_replay_determinism.py`.
