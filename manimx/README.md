# manimx

Small parallel animation-engine prototype.

## Current scope

- Package scaffold parallel to `manim`.
- Isolated `uv` environment.
- Pytest with 90% coverage gate.
- Mutation testing with `mutmut`.
- State-test and visual-snapshot test scaffolds.

Production architecture starts only after this test harness is stable.

## Checks

```bash
uv sync
uv run pytest
uv run mutmut run
uv run mutmut results
```

Pytest enforces 90% branch coverage. `tests/test_state.py` tests the state
harness. `tests/test_visual.py` tests deterministic SVG snapshots with Syrupy.
