# Research track: Studies 1-3 and the V1 verification-gap question

This page explains how the feedback-repair studies relate to the V1 benchmark question, what
each study can and cannot claim, the rules Study 3 must follow, and the order of the open
research issues. Studies are a sub-track of V1 that runs alongside the benchmark.

## Questions

**V1 headline question** ([README](../README.md)):

> Can a coding agent produce numerical physics software that is scientifically correct, not
> merely code that passes ordinary tests?

V1 answers it with the verification gap `G = P_public - P_science`: the public pass rate
minus the scientific pass rate of agent-written code, measured per task
([Methodology: Metrics](methodology.md#metrics)).

**Sub-question the studies answer:**

> When a model repairs an injected defect in the oscillator solver, does the feedback it
> receives (weak, placebo, raw metrics, interpreted metrics) change how often the repair is
> scientifically correct, and how often it makes the solver scientifically worse?

This is repair success under feedback conditions on one task. It is not the verification
gap of agent-written code across the four tasks.

## No completed study measures G

Neither Study 1 nor Study 2 reports G. Both report scientific pass rates of repairs to
injected oscillator defects. Their baseline mutants pass the public checks and fail the
scientific checks by construction, so the gap they exhibit is designed in, not measured.

A G measurement needs:

- public **and** scientific pass rates recorded for every attempt. The repair runner now
  reports `public_pass_rate`, `scientific_pass_rate` and `verification_gap` per condition
  ([#79](https://github.com/akurkar07/InvariantLab/issues/79)), but only for the oscillator;
- the same rates **per task** (oscillator, kepler, heat1d, wave1d), which needs the runner to
  grade candidates through the task packages for any registered mutant
  ([#120](https://github.com/akurkar07/InvariantLab/issues/120));
- a protocol in which the code under test is written by the agent, not a fixed mutant.

All of this is tracked in
[M5: Model Adapters & Evaluation Protocol](https://github.com/akurkar07/InvariantLab/milestone/4).

## Studies

| Study | Role | Attempts | Result | Reproducibility | Evidence |
|---|---|---:|---|---|---|
| Study 1 | Pilot: 3 oscillator mutation families, weak vs. hardened feedback, Cohere North Mini Code | 18 | 17/18 scientific passes; one update-order repair introduced a stale-acceleration bug | Not reproducible from the repo: no Study 1 experiment config or prompt is committed and the `hardened` condition does not exist in the runner. Pending [#100](https://github.com/akurkar07/InvariantLab/issues/100) | [Results](first-multi-condition-study-results.md) |
| Study 2 | Preregistered replication of the update-order defect, 4 conditions x 30 trials x 2 models (Qwen2.5-Coder-7B, DeepSeek-Coder-6.7B) | 240 | 240/240 scientific passes, 0 regressions: a saturated null, no condition effect is detectable | Configs committed; protocol deviations recorded ([#99](https://github.com/akurkar07/InvariantLab/issues/99)); run directories not yet published, pending [#100](https://github.com/akurkar07/InvariantLab/issues/100) | [Protocol](update-order-feedback-replication.md), [results](study-two-model-comparison.md) |
| Study 3 | Planned: preregistered feedback experiment on 3-5 calibrated defects | - | - | Must follow the rules below | [#51](https://github.com/akurkar07/InvariantLab/issues/51) |

What the completed studies can claim:

- Study 1 shows that an observable failure mode exists: a feedback-guided repair can fix the
  local defect and make the solver scientifically worse.
- Study 2 shows that the update-order defect is at ceiling for these two local models, so it
  cannot discriminate between feedback conditions.

What they cannot claim: a general feedback-condition effect, any effect on kepler, heat1d or
wave1d, or any value of G.

## Dependency order of the open research issues

```text
#49 verifiers + mutants  ->  #50 difficulty calibration  ->  #51 Study 3
```

1. [#49](https://github.com/akurkar07/InvariantLab/issues/49) "Finish oscillator verifier
   stack and mutation system". Its verifier half overlaps
   [M3: Verification Stack](https://github.com/akurkar07/InvariantLab/milestone/2) (layers 2,
   3 and 5 have since landed as gates; convergence and robustness remain) and the M3 issues
   #31-#37 that were closed `not_planned`. Its mutation half overlaps
   [M4: Controlled Defect Injection](https://github.com/akurkar07/InvariantLab/milestone/3)
   (the registry and mutant validation have landed; the oscillator has two mutants, not five).
2. [#50](https://github.com/akurkar07/InvariantLab/issues/50) "Difficulty-calibration
   harness": needs a pool of registered, validated mutants from #49 and selects 3-5 defects
   with 30-80% weak-condition success, to avoid another ceiling like Study 2.
3. [#51](https://github.com/akurkar07/InvariantLab/issues/51) "Study 3": needs the calibrated
   defects from #50 ("after calibration harness is built"). It also needs per-condition
   public and scientific pass rates ([#79](https://github.com/akurkar07/InvariantLab/issues/79))
   and, if any defect is outside the oscillator, the task-package runner
   ([#120](https://github.com/akurkar07/InvariantLab/issues/120)).

## Study 3 preregistration rules

Study 2 changed model, backend and temperature after preregistration and recorded the
deviations only later ([#99](https://github.com/akurkar07/InvariantLab/issues/99)). Study 3
must follow these rules:

1. **Commit before running.** The protocol document and every model and experiment config
   used by the study are committed to `main` before the first model call.
2. **Record the SHA.** The protocol document records the commit SHA of that preregistered
   state (added in a follow-up commit, since a commit cannot contain its own SHA), and this
   is merged before the first model call. Every run manifest must match that SHA or a later
   commit that changes only the "Deviations" section.
3. **Deviations only in a "Deviations" section.** After the SHA is recorded, the protocol's
   design, hypotheses, models, conditions, trial counts and analysis plan are not edited. Any
   change is appended to a "Deviations" section with the date, the commit, what changed and
   why, and whether it happened before or after the first model call.
4. **Publish the evidence bundle.** Every run directory is published in the
   [#100](https://github.com/akurkar07/InvariantLab/issues/100) evidence-bundle format
   (`events.jsonl`, `events.canonical.jsonl`, `artifact-integrity.json` and the summary),
   linked from the results document.
5. **Test the intervals.** Every published Wilson interval appears in a `docs/*.md` table
   that `tests/unit/test_study_reports.py`
   ([#98](https://github.com/akurkar07/InvariantLab/issues/98)) checks against
   `invariantlab.metrics.wilson_interval`.

## Tracking actions for the maintainer

Recommendations only; #49-#51 have not been edited.

- [#49](https://github.com/akurkar07/InvariantLab/issues/49): drop the "PR 2:" prefix; add the
  `P1` label (its body says P1); split it into an M3 issue (convergence and robustness gates
  for the oscillator) and an M4 issue (at least five registered and validated oscillator
  mutants), or close it in favour of the existing M3/M4 issues; refresh its "Current
  Problem", which still describes the stubs removed in #44.
- [#50](https://github.com/akurkar07/InvariantLab/issues/50): drop the "PR 3:" prefix; add the
  `P1` label; assign it to M4; add "blocked by #49".
- [#51](https://github.com/akurkar07/InvariantLab/issues/51): add the `P2` label; add "blocked
  by #50, #79, #100" (and #120 if any defect is outside the oscillator); add the
  preregistration rules above as acceptance criteria; replace "that's PR 1"/"that's PR 3"
  with issue links.
- Consider a "Research" milestone or label for #49-#51 and #100, so the study sub-track is
  visible next to M2-M7.
