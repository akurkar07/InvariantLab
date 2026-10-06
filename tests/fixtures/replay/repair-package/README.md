# Package repair replay fixture

`events.jsonl` contains the trusted solver response for the oscillator
`non-conservative-damping` and wave1d `sign-error-startup` package mutants. Regenerate it
from the package prompt and committed reference solvers with:

```bash
uv run python tests/fixtures/replay/repair-package/regenerate.py
```
