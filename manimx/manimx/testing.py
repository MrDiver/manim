from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, Signature, signature
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

TestFunction = TypeVar("TestFunction", bound=Callable[..., Any])


def assert_snapshot(value: object, snapshot: Any) -> None:
    """Compare rendered test content with Syrupy's stored snapshot."""
    assert value == snapshot


def visual_test(function: TestFunction) -> TestFunction:
    """Turn a rendered-content test into a Syrupy visual snapshot test.

    Decorated functions must accept the pytest/Syrupy ``snapshot`` fixture and
    return rendered content such as SVG text or PNG bytes.
    """
    import pytest

    @pytest.mark.visual
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> None:
        snapshot = kwargs.pop("snapshot")
        rendered_content = function(*args, **kwargs)
        assert_snapshot(rendered_content, snapshot)

    parameters = list(signature(function).parameters.values())
    if "snapshot" not in signature(function).parameters:
        parameters.append(
            Parameter("snapshot", kind=Parameter.KEYWORD_ONLY),
        )
    wrapped.__signature__ = Signature(parameters=parameters)

    return wrapped  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ImageTolerance:
    """How far a rendered image may deviate from its reference.

    Rasterizers differ in antialiasing, so an exact match is too strict. A
    pixel is an *outlier* if any channel differs by more than
    ``pixel_threshold``; the image matches if the mean channel difference and
    the fraction of outliers both stay within bounds.
    """

    mean_difference: float = 0.5
    pixel_threshold: int = 32
    outlier_fraction: float = 1e-3


def assert_image_matches(
    image: np.ndarray,
    reference: Path,
    *,
    tolerance: ImageTolerance = ImageTolerance(),
    update: bool = False,
) -> None:
    """Compare an ``(H, W, 4)`` uint8 image against a reference PNG.

    With ``update=True`` (pytest ``--update-references``) a missing or
    differing reference is overwritten instead; review the new PNG before
    committing it. On failure, ``<reference>.actual.png`` and
    ``<reference>.diff.png`` are written next to the reference.
    """
    from PIL import Image

    actual_path = reference.with_suffix(".actual.png")
    diff_path = reference.with_suffix(".diff.png")
    if update or not reference.exists():
        reference.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(reference)
        actual_path.unlink(missing_ok=True)
        diff_path.unlink(missing_ok=True)
        if not update:
            raise AssertionError(f"reference {reference.name} did not exist; created it, review and rerun")
        return

    expected = np.asarray(Image.open(reference).convert("RGBA"))
    if expected.shape != image.shape:
        raise AssertionError(f"size {image.shape} differs from reference {expected.shape}")

    difference = np.abs(image.astype(np.int16) - expected.astype(np.int16)).max(axis=2)
    mean = float(difference.mean())
    outliers = float(np.mean(difference > tolerance.pixel_threshold))
    if mean <= tolerance.mean_difference and outliers <= tolerance.outlier_fraction:
        actual_path.unlink(missing_ok=True)
        diff_path.unlink(missing_ok=True)
        return

    Image.fromarray(image).save(actual_path)
    Image.fromarray(np.clip(difference * 4, 0, 255).astype(np.uint8)).save(diff_path)
    raise AssertionError(
        f"{reference.name}: mean difference {mean:.3f} (max {tolerance.mean_difference}), "
        f"outliers {outliers:.2%} (max {tolerance.outlier_fraction:.2%}); "
        f"see {actual_path.name} and {diff_path.name}"
    )
