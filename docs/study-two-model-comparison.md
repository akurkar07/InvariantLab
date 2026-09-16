# Study 2: Two-Model Comparison — Verifier Condition Effects on Update-Order Repair

**Date:** 2026-09-16  
**Models:** Qwen2.5-Coder-7B-Instruct, DeepSeek-Coder-6.7B-Instruct  
**Mutation:** update-order defect (stale acceleration)  
**Conditions:** weak, placebo, metrics, interpreted  
**Cells per model:** 120 (4 conditions × 30 trials)

## Baseline

- Public tests: **pass** (shape, initial state, finiteness, one-step sanity)
- Scientific tests: **fail** (state error 0.021, energy drift 0.046)

## Results

### Qwen2.5-Coder-7B-Instruct

| Condition | Pass | Regressions | Wilson 95% CI |
|---|---|---|---|
| weak | 30/30 | 0 | [0.904, 1.000] |
| placebo | 30/30 | 0 | [0.908, 1.000] |
| metrics | 30/30 | 0 | [0.904, 1.000] |
| interpreted | 30/30 | 0 | [0.893, 1.000] |
| **Total** | **120/120** | **0** | |

### DeepSeek-Coder-6.7B-Instruct

| Condition | Pass | Regressions | Wilson 95% CI |
|---|---|---|---|
| weak | 30/30 | 0 | [0.904, 1.000] |
| placebo | 30/30 | 0 | [0.908, 1.000] |
| metrics | 30/30 | 0 | [0.904, 1.000] |
| interpreted | 30/30 | 0 | [0.893, 1.000] |
| **Total** | **120/120** | **0** | |

## Artifact Integrity

| Model | Raw Records | Canonical Cells | Duplicates | Status |
|---|---|---|---|---|
| Qwen | 120 | 120 | 0 | ✓ Valid |
| DeepSeek | 120 | 120 | 0 | ✓ Valid |

## Interpretation

**Both models achieved 100% scientific repair success across all four verifier conditions.**

This is a **null result** — there was no observable verifier-feedback effect for either model on this defect.

### What this means

1. **No feedback effect observed.** Showing models concrete failure metrics did not change repair outcomes compared to the weak baseline.

2. **The task is at ceiling.** Both 7B models find and fix this particular update-order defect reliably regardless of prompt condition. This makes the defect a poor discriminator for detecting feedback effects.

3. **Study 1's regression remains unexplained.** The single regression observed in the exploratory Cohere study (1/18 in metrics) could be:
   - Sampling noise (n=3 per cell is very small)
   - Model-specific to weaker models
   - A genuine effect that requires a harder task to observe

### What we can claim

> Across 240 total scheduled repair attempts (120 per model) with two different 7B coding models, all four verifier-information conditions achieved 100% scientific repair success with no measured regressions. Verifier feedback had no observable effect on repair outcome for this model-defect configuration.

### What we cannot claim

- That "scientific feedback has no effect generally" — the task may be too easy
- That "Study 1 was definitely noise" — the models differ
- That feedback effects don't exist — only that they're not visible on this defect at this model scale

## Next Steps

The update-order mutation has saturated at 100% for both 7B models. Future studies should test **harder defects** where weak-condition repair success is 30-80%, creating room for feedback to help or hurt.

Candidate progression:
- Level 2: Two coupled stale-state/update-order defects
- Level 3: Defect that can violate a global invariant
- Level 4: Defect visible primarily through long-horizon behavior
- Level 5: Multiple locally plausible repairs, only one preserves the invariant

## Artifacts and provenance

The completed runs produced these local run directories:

- `runs/update-order-feedback-replication-ollama-qwen25-7b/`
- `runs/update-order-feedback-replication-ollama-deepseek/`

Each run directory contained:

- `events.jsonl` — immutable raw evidence
- `events.canonical.jsonl` — canonical scheduled cells
- `artifact-integrity.json` — integrity audit report
- `study-summary.json` — structured results table

These run directories are **not currently committed to the repository**. The tables in this
report were generated from the locally audited artifacts described above, but the paths
should not be interpreted as repository-hosted evidence. A future public release or dataset
should attach the canonical event records and integrity reports if independent reconstruction
of the Study 2 aggregates is required.
