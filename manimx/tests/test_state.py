import copy

import pytest


@pytest.mark.state
def test_state_fixture_is_independent(state: dict[str, object]) -> None:
    first = copy.deepcopy(state)
    first["objects"]["test-object"]["x"] = 10.0  # type: ignore[index]

    assert state["objects"]["test-object"]["x"] == 0.0  # type: ignore[index]


@pytest.mark.state
def test_state_contains_object_identity(state: dict[str, object]) -> None:
    objects = state["objects"]

    assert isinstance(objects, dict)
    assert "test-object" in objects
