"""Direction and angle constants, as in Manim (world +z points out of the
screen in 2D scenes and up in 3D scenes)."""

from __future__ import annotations

import math

import numpy as np

ORIGIN = np.array([0.0, 0.0, 0.0])
RIGHT = np.array([1.0, 0.0, 0.0])
LEFT = -RIGHT
UP = np.array([0.0, 1.0, 0.0])
DOWN = -UP
OUT = np.array([0.0, 0.0, 1.0])
IN = -OUT
UL, UR, DL, DR = UP + LEFT, UP + RIGHT, DOWN + LEFT, DOWN + RIGHT

PI = math.pi
TAU = math.tau
DEGREES = TAU / 360
