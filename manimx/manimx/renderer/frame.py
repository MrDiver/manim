"""What Python hands the core's renderer per view: the camera's Frame
record and the configuration values the core's packer reads.

A "frame" (shaders/frame.wgsl) is everything a pass needs to know about
the point of view it renders from. The camera's frame drives the normal
passes; the core builds the frames of shadow maps itself, with the
camera at a light.
"""

from __future__ import annotations

import numpy as np

from .config import RenderConfig

# Mirrors `struct Frame` in shaders/frame.wgsl.
FRAME_DTYPE = np.dtype(
    [
        ("view", "<f4", (4, 4)),
        ("projection", "<f4", (4, 4)),
        ("view_projection", "<f4", (4, 4)),
        ("inverse_view", "<f4", (4, 4)),
        ("inverse_projection", "<f4", (4, 4)),
        ("camera_position", "<f4", 4),
        ("viewport", "<f4", 4),
        ("time", "<f4"),
        ("layer_depth_step", "<f4"),
        ("light_count", "<u4"),
        ("exposure", "<f4"),
        ("ambient", "<f4", 4),
        ("shadow", "<f4", 4),
        ("shadow_filter", "<f4", 4),
    ]
)
assert FRAME_DTYPE.itemsize == 416


def frame_data(
    view: np.ndarray,
    projection: np.ndarray,
    position: np.ndarray,
    size: tuple[int, int],
    config: RenderConfig,
    light_count: int = 0,
) -> np.ndarray:
    """One Frame record for a point of view (camera or light)."""
    width, height = size
    frame = np.zeros((), FRAME_DTYPE)
    # numpy matrices are row-major, WGSL's are column-major: transpose.
    frame["view"] = view.T
    frame["projection"] = projection.T
    frame["view_projection"] = (projection @ view).T
    frame["inverse_view"] = np.linalg.inv(view).T
    frame["inverse_projection"] = np.linalg.inv(projection).T
    frame["camera_position"] = (*position, 1.0)
    frame["viewport"] = (width, height, 1 / width, 1 / height)
    frame["time"] = config.time
    frame["layer_depth_step"] = config.layer_depth_step
    frame["light_count"] = light_count
    frame["exposure"] = config.exposure
    frame["ambient"][:3] = config.ambient_linear
    frame["shadow"] = (
        config.shadow_softness,
        config.shadow_samples,
        config.shadow_resolution,
        1.0 if config.shadows else 0.0,
    )
    frame["shadow_filter"] = config.shadow_filter
    return frame


def camera_frame(config: RenderConfig, light_count: int) -> np.ndarray:
    camera = config.camera
    return frame_data(
        camera.view_matrix(),
        camera.projection_matrix(config.aspect_ratio),
        camera.position,
        (config.pixel_width, config.pixel_height),
        config,
        light_count,
    )


def config_values(config: RenderConfig) -> tuple:
    """What the core's packer reads from ``config`` (see parse_settings in
    core/src/packing/packer.rs)."""
    camera = config.camera
    return (
        config.time,
        config.layer_depth_step,
        config.exposure,
        tuple(config.ambient_linear),
        config.shadow_softness,
        config.shadow_samples,
        config.shadow_resolution,
        config.shadows,
        config.default_lights,
        camera.transform.matrix().reshape(-1).tolist(),
        [(view.uid, 1.0 / view.width, 1.0 / view.height) for view in config.camera_views],
        config.hide_camera_views,
        camera.focus_distance,
        camera.aperture,
        config.shadow_filter,
    )
