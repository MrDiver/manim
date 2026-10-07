import numpy as np
import pytest
from PIL import Image

from manimx import main


def test_main_renders_and_shows_logo(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[Image.Image] = []
    monkeypatch.setattr(Image.Image, "show", lambda image: shown.append(image))
    try:
        main.main()
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")

    assert len(shown) == 1
    assert shown[0].size == (1920, 1080)
    assert np.asarray(shown[0])[0, 0, :3].tolist() == [236, 230, 226]  # background
