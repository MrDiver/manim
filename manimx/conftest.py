"""Project-wide pytest options, loaded for every test directory."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-references",
        action="store_true",
        help="overwrite the reference images of visual tests with the current output",
    )


@pytest.fixture
def update_references(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-references"))
