"""Project settings: every value a project can change, with its docs.

The settings are plain dataclasses, nested by section::

    Settings
        output       OutputSettings        size, frame rate, background, files
        effects      EffectsSettings       motion blur, bloom, shadows, tone mapping
        lighting     LightingSettings      ambient light, default lights
        performance  PerformanceSettings   processes, frames in flight

Each value is declared with :func:`setting`: its default, a description
and its limits. That one declaration is the documentation everywhere: the
JSON schema editors use for manimx.json, ``manimx settings`` (the
interactive menu and its printout) and the validation of manimx.json. A
section with an ``enabled`` value can be switched off as a whole.

:meth:`Settings.render_config` and :meth:`Settings.effects_chain` turn the
settings into what the renderer and the frame composer take.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields
from typing import Any

from .renderer.config import RenderConfig
from .renderer.effects import Bloom, Effect, MotionBlur, ToneMap, ToneMapOperator
from .utils.color import ManimColor


class SettingsError(ValueError):
    """A value that does not fit its setting (the message names the setting)."""


def setting(
    default: Any,
    description: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    choices: tuple[Any, ...] | None = None,
    unit: str = "",
    color: bool = False,
    note: Callable[[Settings], str] | None = None,
) -> Any:
    """A setting: its default and what editors, the menu and errors say.

    ``note`` adds a computed remark in the menu, e.g. the shutter angle's
    exposure time in seconds at the current frame rate.
    """
    metadata = {"description": description, "minimum": minimum, "maximum": maximum, "choices": choices, "unit": unit, "color": color, "note": note}
    return field(default=default, metadata=metadata)


def section(kind: type, description: str) -> Any:
    """A nested section of settings."""
    return field(default_factory=kind, metadata={"description": description})


# --------------------------------------------------------------------------
# The sections
# --------------------------------------------------------------------------


@dataclass
class OutputSettings:
    """What the rendered video looks like."""

    width: int = setting(1920, "Video width in pixels.", minimum=1, unit="px")
    height: int = setting(1080, "Video height in pixels.", minimum=1, unit="px")
    fps: int = setting(60, "Frames per second.", minimum=1)
    background: str = setting("#0b1020", "Background color, e.g. #0b1020 or BLACK.", color=True)
    background_opacity: float = setting(
        1.0, "Background opacity; below 1 needs an output with alpha, e.g. --png.", minimum=0.0, maximum=1.0
    )
    antialiasing: int = setting(4, "Samples per pixel for smooth edges (1: off).", choices=(1, 4))
    audio: bool = setting(True, "Include the scenes' sounds.")
    sample_rate: int = setting(48_000, "Audio sample rate.", minimum=8000, unit="Hz")
    directory: str = setting("renders", "Where renders go, relative to the project.")


def _exposure(settings: Settings) -> str:
    blur = settings.effects.motion_blur
    seconds = blur.shutter_angle / 360 / settings.output.fps
    return f"{seconds:.4g} s exposure at {settings.output.fps} fps"


def _spacing(settings: Settings) -> str:
    blur = settings.effects.motion_blur
    seconds = blur.shutter_angle / 360 / settings.output.fps
    return f"one image every {seconds / blur.samples:.3g} s of the exposure"


@dataclass
class MotionBlurSettings:
    """Average several renders spread over the exposure, like a camera
    whose shutter stays open. Every sample is a full render."""

    enabled: bool = setting(True, "Blur moving things over the exposure time.")
    samples: int = setting(8, "Renders averaged per frame; long exposures need more, or they show separate images.", minimum=1, note=_spacing)
    shutter_angle: float = setting(
        180.0,
        "Exposure as a share of a frame: 360 is one frame, 180 half (film's default), 720 two frames.",
        minimum=0.0,
        unit="°",
        note=_exposure,
    )


@dataclass
class BloomSettings:
    """Glow around bright parts of lit 3D objects (2D shapes never bloom)."""

    enabled: bool = setting(False, "Add glow around bright lit pixels.")
    intensity: float = setting(0.1, "Strength of the glow.", minimum=0.0)
    threshold: float = setting(1.0, "Brightness above which pixels glow (1: white).", minimum=0.0)
    knee: float = setting(0.5, "Softness of the threshold: 0 is a hard cut.", minimum=0.0)
    radius: float = setting(1.0, "Width of the blur filter per level.", minimum=0.0)
    levels: int = setting(6, "Mip levels the glow spreads over; more give a wider glow.", minimum=1, maximum=12)


@dataclass
class ShadowSettings:
    """Shadows of 3D objects; lights and objects can also turn their own off."""

    enabled: bool = setting(True, "Cast shadows at all.")
    softness: float = setting(1.0, "Multiplies every light's size: 0 gives hard shadows, larger is softer.", minimum=0.0)
    samples: int = setting(16, "Lookups per shadowed pixel; more give smoother soft edges.", minimum=1, maximum=64)
    resolution: int = setting(2048, "Width and height of each shadow map; more give sharper detail.", choices=(512, 1024, 2048, 4096), unit="px")
    method: str = setting(
        "pcss",
        "pcss: edges soften with the distance to the caster and the light's size; pcf: a fixed narrow soft edge, much faster (previews use it).",
        choices=("pcss", "pcf"),
    )


TONE_MAPS = tuple(operator.name.lower() for operator in ToneMapOperator)


@dataclass
class ToneMapSettings:
    """How bright lit 3D colors are mapped to the screen (2D colors stay exact)."""

    operator: str = setting(
        "pbr_neutral",
        "pbr_neutral keeps base colors faithful, aces is filmic, agx is Blender's, reinhard is soft, none clamps.",
        choices=TONE_MAPS,
    )
    exposure: float = setting(1.0, "Multiplies lit 3D colors before tone mapping.", minimum=0.0)


@dataclass
class EffectsSettings:
    """Camera and image effects."""

    motion_blur: MotionBlurSettings = section(MotionBlurSettings, "Average renders over the exposure time.")
    bloom: BloomSettings = section(BloomSettings, "Glow around bright lit pixels.")
    shadows: ShadowSettings = section(ShadowSettings, "Shadows of 3D objects.")
    tone_map: ToneMapSettings = section(ToneMapSettings, "Mapping of lit 3D colors to the screen.")


@dataclass
class LightingSettings:
    """Light that 3D objects get besides the scene's own lights."""

    ambient_color: str = setting("#ffffff", "Color of the constant ambient light.", color=True)
    ambient_intensity: float = setting(0.15, "Strength of the ambient light.", minimum=0.0)
    default_lights: bool = setting(True, "Add a key light when a scene has 3D objects but no lights.")


@dataclass
class PerformanceSettings:
    """How rendering uses the machine; never changes the image."""

    workers: int = setting(1, "Render processes; more use more CPU cores.", minimum=1)
    frames_in_flight: int = setting(3, "Frames queued on the GPU per process.", minimum=1)
    core: bool = setting(True, "Evaluate and pack frames in the native core; off: the Python reference path (same pixels, slower).")


@dataclass
class Settings:
    """All settings of a project (everything in manimx.json except scenes and timeline)."""

    output: OutputSettings = section(OutputSettings, "Video size, frame rate, background and files.")
    effects: EffectsSettings = section(EffectsSettings, "Motion blur, bloom, shadows and tone mapping.")
    lighting: LightingSettings = section(LightingSettings, "Ambient and default light for 3D objects.")
    performance: PerformanceSettings = section(PerformanceSettings, "Processes and GPU queueing.")

    # -- for the renderer -----------------------------------------------------

    def render_config(self) -> RenderConfig:
        output, effects, lighting = self.output, self.effects, self.lighting
        return RenderConfig(
            pixel_width=output.width,
            pixel_height=output.height,
            background_color=output.background,
            background_opacity=output.background_opacity,
            msaa_samples=output.antialiasing,
            ambient_color=lighting.ambient_color,
            ambient_intensity=lighting.ambient_intensity,
            default_lights=lighting.default_lights,
            exposure=effects.tone_map.exposure,
            shadows=effects.shadows.enabled,
            shadow_softness=effects.shadows.softness,
            shadow_samples=effects.shadows.samples,
            shadow_resolution=effects.shadows.resolution,
            shadow_method=effects.shadows.method,
        )

    def effects_chain(self) -> tuple[Effect, ...]:
        """The frame composer's effects, in the order they apply."""
        chain: list[Effect] = []
        blur = self.effects.motion_blur
        if blur.enabled and blur.samples > 1:
            chain.append(MotionBlur(samples=blur.samples, shutter=blur.shutter_angle / 360 / self.output.fps))
        bloom = self.effects.bloom
        if bloom.enabled:
            chain.append(Bloom(threshold=bloom.threshold, knee=bloom.knee, intensity=bloom.intensity, radius=bloom.radius, levels=bloom.levels))
        chain.append(ToneMap(ToneMapOperator[self.effects.tone_map.operator.upper()]))
        return tuple(chain)

    @property
    def motion_blur_samples(self) -> int:
        """Renders per frame (1 when motion blur is off)."""
        blur = self.effects.motion_blur
        return blur.samples if blur.enabled else 1


# --------------------------------------------------------------------------
# Reading, writing, checking
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One value or section in the settings tree, for the menu and printouts."""

    path: tuple[str, ...]
    """Attribute names from :class:`Settings` down, e.g. ("effects", "bloom", "intensity")."""
    field: dataclasses.Field
    is_section: bool

    @property
    def name(self) -> str:
        return self.path[-1]

    @property
    def label(self) -> str:
        return self.name.replace("_", " ")

    @property
    def description(self) -> str:
        return self.field.metadata.get("description", "")


def entries(kind: type = Settings, path: tuple[str, ...] = ()) -> Iterator[Entry]:
    """Every section and value, depth first, in declaration order."""
    for item in fields(kind):
        nested = _is_section(item)
        here = (*path, item.name)
        yield Entry(here, item, nested)
        if nested:
            yield from entries(_section_type(kind, item), here)


def get(settings: Any, path: tuple[str, ...]) -> Any:
    for name in path:
        settings = getattr(settings, name)
    return settings


def set_value(settings: Settings, path: tuple[str, ...], value: Any) -> None:
    """Set the value at ``path`` after checking it (raises SettingsError)."""
    owner = get(settings, path[:-1])
    item = next(item for item in fields(owner) if item.name == path[-1])
    setattr(owner, path[-1], check(item, value, ".".join(path)))


def check(item: dataclasses.Field, value: Any, where: str) -> Any:
    """``value`` if it fits the setting ``item``, as the setting's type."""
    expected = type(item.default)
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if isinstance(value, bool) and expected is not bool or not isinstance(value, expected):
        raise SettingsError(f'"{where}" must be {_JSON_TYPES[expected]}, got {_shown(value)}')
    meta = item.metadata
    if meta.get("minimum") is not None and value < meta["minimum"]:
        raise SettingsError(f'"{where}" must be at least {meta["minimum"]:g}, got {value}')
    if meta.get("maximum") is not None and value > meta["maximum"]:
        raise SettingsError(f'"{where}" must be at most {meta["maximum"]:g}, got {value}')
    if meta.get("choices") is not None and value not in meta["choices"]:
        options = ", ".join(str(choice) for choice in meta["choices"])
        raise SettingsError(f'"{where}" must be one of {options}, got {_shown(value)}')
    if meta.get("color") and not _is_color(value):
        raise SettingsError(f'"{where}" is not a color: {value!r} (use #rrggbb or a name like BLUE)')
    return value


def parse(text: str, item: dataclasses.Field, where: str) -> Any:
    """A value typed by hand (in the menu) for the setting ``item``."""
    expected = type(item.default)
    text = text.strip()
    try:
        if expected is bool:
            value: Any = {"true": True, "on": True, "yes": True, "false": False, "off": False, "no": False}[text.lower()]
        elif expected is int:
            value = int(text)
        elif expected is float:
            value = float(text)
        else:
            value = text
    except (KeyError, ValueError):
        raise SettingsError(f'"{where}" must be {_JSON_TYPES[expected]}, got {text!r}') from None
    return check(item, value, where)


def from_json(data: Any, kind: type = Settings, where: str = "") -> Any:
    """Settings of ``kind`` from JSON data, every key and value checked."""
    if not isinstance(data, dict):
        raise SettingsError(f'"{where}" must be an object')
    items = {item.name: item for item in fields(kind)}
    for key in data:
        if key not in items:
            close = difflib.get_close_matches(key, items, n=1)
            raise UnknownSetting(f"{where}.{key}" if where else key, close[0] if close else None, list(items))
    values = {}
    for name, value in data.items():
        item = items[name]
        here = f"{where}.{name}" if where else name
        values[name] = from_json(value, _section_type(kind, item), here) if _is_section(item) else check(item, value, here)
    return kind(**values)


def to_json(settings: Any) -> dict[str, Any]:
    return dataclasses.asdict(settings)


def schema(kind: type = Settings) -> dict[str, Any]:
    """JSON schema of a settings dataclass (editors complete and check manimx.json)."""
    properties = {}
    for item in fields(kind):
        if _is_section(item):
            properties[item.name] = {**schema(_section_type(kind, item)), "description": item.metadata["description"]}
            continue
        meta = item.metadata
        entry: dict[str, Any] = {"type": _JSON_TYPES[type(item.default)], "default": item.default, "description": meta["description"]}
        if meta.get("minimum") is not None:
            entry["minimum"] = meta["minimum"]
        if meta.get("maximum") is not None:
            entry["maximum"] = meta["maximum"]
        if meta.get("choices") is not None:
            entry["enum"] = list(meta["choices"])
        properties[item.name] = entry
    return {"type": "object", "description": (kind.__doc__ or "").strip(), "properties": properties, "additionalProperties": False}


class UnknownSetting(SettingsError):
    """A key manimx.json does not know (likely a typo)."""

    def __init__(self, where: str, close: str | None, known: list[str]) -> None:
        super().__init__(f'unknown setting "{where}"')
        self.where, self.close, self.known = where, close, known


_JSON_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean"}


def _is_section(item: dataclasses.Field) -> bool:
    return item.default is dataclasses.MISSING and item.default_factory is not dataclasses.MISSING  # type: ignore[misc]


def _section_type(kind: type, item: dataclasses.Field) -> type:
    return item.default_factory  # type: ignore[return-value]  # sections are created by their class


_HEX = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})")


def _is_color(text: str) -> bool:
    """A hex color (exactly) or a color name ManimColor knows."""
    if text.startswith("#"):
        return _HEX.fullmatch(text) is not None
    try:
        ManimColor.parse(text)
    except Exception:  # noqa: BLE001 - any parse failure is a bad color
        return False
    return True


def _shown(value: Any) -> str:
    import json

    return json.dumps(value)
