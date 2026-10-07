from __future__ import annotations

import pytest


@pytest.fixture
def state() -> dict[str, object]:
    """Small state fixture for the future animation engine tests."""
    return {"objects": {"test-object": {"x": 0.0, "visible": True}}}


@pytest.fixture
def svg() -> str:
    """Deterministic visual fixture for Syrupy snapshot tests."""
    return '<svg height="16" viewBox="0 0 16 16" width="16"><rect height="8" width="8" x="4" y="4" /></svg>'


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-references",
        action="store_true",
        help="overwrite reference images in tests/references with the current output",
    )


@pytest.fixture
def update_references(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-references"))
