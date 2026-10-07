"""Type aliases ported from ``manim.typing`` as manimx needs them."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ManimFloat: TypeAlias = np.float64
ManimInt: TypeAlias = np.int64
ManimColorDType: TypeAlias = ManimFloat
PointDType: TypeAlias = ManimFloat

FloatRGB: TypeAlias = npt.NDArray[ManimColorDType]
FloatRGBLike: TypeAlias = FloatRGB | tuple[float, float, float]
FloatRGBA: TypeAlias = npt.NDArray[ManimColorDType]
FloatRGBALike: TypeAlias = FloatRGBA | tuple[float, float, float, float]

IntRGB: TypeAlias = npt.NDArray[ManimInt]
IntRGBLike: TypeAlias = IntRGB | tuple[int, int, int]
IntRGBA: TypeAlias = npt.NDArray[ManimInt]
IntRGBALike: TypeAlias = IntRGBA | tuple[int, int, int, int]

FloatHSV: TypeAlias = FloatRGB
FloatHSVLike: TypeAlias = FloatRGBLike
FloatHSVA: TypeAlias = FloatRGBA
FloatHSVALike: TypeAlias = FloatRGBALike
FloatHSL: TypeAlias = FloatRGB
FloatHSLLike: TypeAlias = FloatRGBLike

ManimColorInternal: TypeAlias = FloatRGBA

Point3D: TypeAlias = npt.NDArray[PointDType]
Point3DLike: TypeAlias = Point3D | tuple[float, float, float]
Vector3D: TypeAlias = npt.NDArray[PointDType]
Vector3DLike: TypeAlias = npt.NDArray[PointDType] | tuple[float, float, float]
