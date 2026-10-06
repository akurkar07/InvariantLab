# Releasing InvariantLab

This page is the release procedure and the V1 packaging decision. What has changed since the
last release is recorded in [CHANGELOG.md](../CHANGELOG.md).

## Versions

Versions come from git tags via [hatch-vcs](https://github.com/ofek/hatch-vcs)
(`[tool.hatch.version] source = "vcs"` in `pyproject.toml`). An untagged checkout reports a
development version such as `0.1.devN+g<sha>` (`invariantlab version`); the tag `vX.Y.Z`
produces version `X.Y.Z`. **Never edit a version string by hand**: there is no version field
in `pyproject.toml` or `src/` to edit, and the tag is the single source of truth.
`CITATION.cff` `version` is the one copy kept in step with the tag (step 2).

## Packaging decision for V1

**V1 runs from a source checkout** (`git clone` + `uv sync`). The wheel provides the
`invariantlab` library and CLI only (`[tool.hatch.build.targets.wheel] packages =
["src/invariantlab"]`); benchmark tasks (`tasks/`) and configs (`configs/`) are read from the
checkout, relative to the current directory. `invariantlab run` started outside a checkout
exits non-zero with "run InvariantLab from a source checkout (tasks/ not found)".

Packaging tasks and configs as package data is explicitly deferred to post-V1. The task
packages contain hidden scientific tests that should not ship in a public wheel anyway.

## Procedure

Run these steps on a release branch cut from the candidate commit on `main`, and merge it
through a normal PR (all required checks green) before tagging.

1. **All V1 acceptance gates are green on the candidate SHA.** Run
   `uv run python scripts/check_v1_acceptance.py --report v1.json` (the full suite, including
   the Docker-backed V1-AC4 tests) and confirm every criterion V1-AC1..V1-AC8 is proved; see
   [V1 acceptance criteria](v1-acceptance.md). The CI checks on that SHA must also be green.
2. **Bump `CITATION.cff`.** Set `version: X.Y.Z` and `date-released: YYYY-MM-DD` (replacing
   the "added when the first release is tagged" comment), then validate with
   `cffconvert --validate -i CITATION.cff`.
3. **Move Unreleased in `CHANGELOG.md`.** Rename `## [Unreleased]` to
   `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty `## [Unreleased]` above it, and update the
   compare links at the bottom of the file.
4. **Set the classifier.** Update the `Development Status :: ...` classifier in
   `pyproject.toml` (currently `3 - Alpha`) to the maturity of this release.
5. **Tag.** After the release PR has merged, the maintainer pushes an annotated tag on `main`:
   `git tag -a vX.Y.Z -m "InvariantLab X.Y.Z" <merge-sha>` then `git push origin vX.Y.Z`.
   Only the maintainer creates tags; agents and contributors do not.
6. **Release workflow builds and publishes.** The tag-triggered release workflow builds the
   sdist and wheel, checks that the tag, `CITATION.cff` and `CHANGELOG.md` agree, and
   publishes the release. That workflow is tracked in
   [#71](https://github.com/akurkar07/InvariantLab/issues/71); until it lands, build locally
   with `uv build` and attach the artifacts to the GitHub release by hand.
