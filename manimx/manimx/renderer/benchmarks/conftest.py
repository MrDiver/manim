"""Rendering benchmarks (pytest-benchmark). Not part of the normal test run.

    uv run pytest manimx/renderer/benchmarks --no-cov                       # run and print
    uv run pytest manimx/renderer/benchmarks --no-cov --benchmark-autosave  # store a baseline
    uv run pytest manimx/renderer/benchmarks --no-cov --benchmark-compare --benchmark-compare-fail=mean:15%

Each benchmark measures wall time of a complete frame (CPU work, GPU work
and read-back), then renders once more with GPU timestamps and stores the
per-pass GPU times (ms) in the benchmark's ``extra_info``, which pytest-
benchmark saves with the results and prints in the summary below.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from manimx.renderer import WGPURenderer

_GPU_TIMINGS: dict[str, dict[str, float]] = {}


@pytest.fixture(scope="session")
def renderer() -> WGPURenderer:
    try:
        return WGPURenderer(profile=True)
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


@pytest.fixture
def measure(benchmark, renderer: WGPURenderer, request: pytest.FixtureRequest) -> Callable:
    """``measure(frame)``: benchmark ``frame()``, then record its GPU pass times."""

    def run(frame: Callable[[], object], rounds: int = 20) -> None:
        frame()  # warm up: pipelines, caches, buffers
        renderer.profiler.read()  # drop warm-up timings
        benchmark.pedantic(frame, rounds=rounds, warmup_rounds=2)
        renderer.profiler.read()
        frame()
        timings = {name: round(ms, 3) for name, ms in renderer.profiler.read().items()}
        benchmark.extra_info["gpu_ms"] = timings
        _GPU_TIMINGS[request.node.name] = timings

    return run


def pytest_terminal_summary(terminalreporter) -> None:
    if not _GPU_TIMINGS:
        return
    terminalreporter.section("GPU time per pass (ms, one frame)")
    passes = sorted({name for timings in _GPU_TIMINGS.values() for name in timings})
    width = max(len(name) for name in _GPU_TIMINGS)
    terminalreporter.write_line(" " * width + "".join(f"{p:>11}" for p in passes) + f"{'total':>11}")
    for test, timings in _GPU_TIMINGS.items():
        cells = "".join(f"{timings.get(p, 0.0):>11.3f}" for p in passes)
        terminalreporter.write_line(f"{test:<{width}}{cells}{sum(timings.values()):>11.3f}")
