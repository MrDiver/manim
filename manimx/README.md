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
harness. `manimx/renderer/tests/test_visual.py` renders scenes with the WebGPU renderer and
compares them with reference PNGs in `manimx/renderer/tests/references`
(`uv run pytest --update-references` regenerates them; review before
committing).

## Renderer

The WebGPU renderer lives in `manimx/renderer`. See
[manimx/renderer/README.md](manimx/renderer/README.md) for usage, the
architecture diagrams, design decisions, known limitations, and how to run
its visual tests and benchmarks.
