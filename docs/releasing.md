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
   `pyproject.toml` (`5 - Production/Stable` since 1.0.0) to the maturity of this release.
5. **Tag.** After the release PR has merged, the maintainer pushes an annotated tag on `main`:
   `git tag -a vX.Y.Z -m "InvariantLab X.Y.Z" <merge-sha>` then `git push origin vX.Y.Z`.
   Only the maintainer creates tags; agents and contributors do not.
6. **Release workflow builds and publishes.** Pushing a `v*` tag triggers
   [`.github/workflows/release.yml`](../.github/workflows/release.yml). It checks release
   metadata with `scripts/check_release_metadata.py --tag`, runs
   `scripts/check_v1_acceptance.py` for final tags, builds an sdist and wheel, checks the wheel
   version against the tag, and smoke-tests `invariantlab version` from the wheel in a fresh
   virtual environment. It then creates a GitHub Release with `dist/*` and the matching
   CHANGELOG section as its body; `-rc`, `-a` and `-b` tags are marked prereleases. After steps
   2–4, run `uv run python scripts/check_release_metadata.py --tag vX.Y.Z` locally to check the
   release metadata before tagging.

### Dry run

Before tagging, open **Actions → Release → Run workflow** on the release branch, enter the
intended tag, and leave `dry_run` enabled. Alternatively, dispatch it with
`POST /repos/akurkar07/InvariantLab/actions/workflows/release.yml/dispatches` and
`{"ref": "<branch>", "inputs": {"tag": "v1.0.0", "dry_run": "true"}}`. The workflow tags its
checkout locally (never pushed) so hatch-vcs builds the intended tag's version, runs every
build and validation step, and skips only the GitHub Release job.
