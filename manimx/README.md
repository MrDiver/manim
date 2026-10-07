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
harness. `tests/test_visual.py` renders scenes with the WebGPU renderer and
compares them with reference PNGs in `tests/references`
(`uv run pytest --update-references` regenerates them; review before
committing).

## Benchmarks

Rendering benchmarks live in `benchmarks/` and are not part of the normal
test run. Each one measures complete frames (CPU, GPU and read-back) and
prints the GPU time per render pass, measured with timestamp queries.

```bash
uv run pytest benchmarks --no-cov                       # run and print
uv run pytest benchmarks --no-cov --benchmark-autosave  # store a baseline
uv run pytest benchmarks --no-cov --benchmark-compare --benchmark-compare-fail=mean:15%
```

Baselines are stored in `.benchmarks/` (ignored by git, since they depend
on the machine). `WGPURenderer(profile=True)` together with
`renderer.profiler.read()` gives the same per-pass GPU timings in any script.
