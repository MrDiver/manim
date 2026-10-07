from __future__ import annotations

import pytest


@pytest.fixture
def state() -> dict[str, object]:
    """Small state fixture for the future animation engine tests."""
    return {"objects": {"test-object": {"x": 0.0, "visible": True}}}

