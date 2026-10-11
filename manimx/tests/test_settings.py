"""Project settings: checking, JSON, migration and what the renderer gets."""

import json

import pytest

from manimx.project import Project, ProjectError
from manimx.renderer.effects import Bloom, RenderSamples, ToneMap, ToneMapOperator
from manimx.settings import Settings, SettingsError, entries, from_json, get, parse, set_value, to_json


def test_defaults_render_like_before() -> None:
    settings = Settings()
    config = settings.render_config()

    assert (config.pixel_width, config.pixel_height, config.msaa_samples) == (1920, 1080, 4)
    assert config.shadows and config.shadow_resolution == 2048 and config.default_lights
    assert settings.effects_chain() == (ToneMap(ToneMapOperator.PBR_NEUTRAL),)  # effects that cost renders are off
    assert settings.effects.shadows.method == "pcf"


def test_effects_follow_their_settings() -> None:
    settings = Settings()
    settings.effects.motion_blur.enabled = True
    settings.effects.motion_blur.shutter_angle = 720.0  # two frames of exposure
    settings.effects.bloom.enabled = True
    settings.effects.bloom.intensity = 0.4
    settings.effects.tone_map.operator = "agx"
    settings.effects.shadows.enabled = False
    settings.output.fps = 30
    settings.output.samples = 8

    samples, bloom, tone_map = settings.effects_chain()
    assert samples.count == 8
    assert samples.shutter == pytest.approx(2 / 30)
    assert bloom == Bloom(intensity=0.4)
    assert tone_map.operator is ToneMapOperator.AGX
    assert not settings.render_config().shadows

    settings.effects.motion_blur.enabled = False
    assert settings.effects_chain()[0] == RenderSamples(count=8)  # jittered renders, all at the frame's time
    settings.output.samples = 1
    assert not any(isinstance(effect, RenderSamples) for effect in settings.effects_chain())


def test_values_are_checked_against_their_setting() -> None:
    settings = Settings()
    set_value(settings, ("effects", "bloom", "intensity"), 2)  # an integer is a fine number
    assert settings.effects.bloom.intensity == 2.0

    with pytest.raises(SettingsError, match='"output.fps" must be at least 1'):
        set_value(settings, ("output", "fps"), 0)
    with pytest.raises(SettingsError, match="must be one of 512, 1024, 2048, 4096"):
        set_value(settings, ("effects", "shadows", "resolution"), 3000)
    with pytest.raises(SettingsError, match="at most 64"):
        set_value(settings, ("effects", "shadows", "samples"), 65)
    with pytest.raises(SettingsError, match="not a color"):
        set_value(settings, ("lighting", "ambient_color"), "no-such-color")
    with pytest.raises(SettingsError, match="must be boolean"):
        set_value(settings, ("output", "audio"), 1)
    assert settings.output.fps == 60  # rejected values change nothing


def test_typed_values_are_parsed_for_the_menu() -> None:
    item = next(entry for entry in entries() if entry.path == ("effects", "motion_blur", "shutter_angle")).field
    assert parse(" 720 ", item, "shutter_angle") == 720.0
    audio = next(entry for entry in entries() if entry.path == ("output", "audio")).field
    assert parse("off", audio, "audio") is False
    with pytest.raises(SettingsError, match="must be number"):
        parse("fast", item, "shutter_angle")


def test_json_round_trip_and_unknown_keys() -> None:
    settings = Settings()
    settings.effects.bloom.enabled = True
    data = json.loads(json.dumps(to_json(settings)))

    assert from_json(data) == settings
    assert from_json({}) == Settings()  # everything left out keeps its default
    with pytest.raises(SettingsError, match='unknown setting "effects.blom"') as error:
        from_json({"effects": {"blom": {}}})
    assert error.value.close == "bloom"


def test_every_entry_is_documented() -> None:
    paths = [entry.path for entry in entries()]
    assert ("effects", "motion_blur", "shutter_angle") in paths and ("lighting",) in paths
    for entry in entries():
        assert entry.description, entry.path
        assert get(Settings(), entry.path) is not None


def test_format_1_projects_are_migrated(tmp_path) -> None:
    (tmp_path / "manimx.json").write_text(
        json.dumps({"format": 1, "name": "old", "output": {"fps": 30, "motion_blur": 1}, "camera": {"frame_height": 8.0}})
    )
    project = Project.load(tmp_path)

    assert project.output.fps == 30
    assert project.settings.effects.motion_blur.enabled is False
    assert project.settings.output.samples == 1
    project.save()
    assert json.loads((tmp_path / "manimx.json").read_text())["format"] == 3


def test_format_2_motion_blur_samples_become_render_samples(tmp_path) -> None:
    blurred = {"enabled": True, "samples": 6, "shutter_angle": 90.0}
    (tmp_path / "manimx.json").write_text(json.dumps({"format": 2, "name": "old", "effects": {"motion_blur": blurred}}))
    settings = Project.load(tmp_path).settings
    assert settings.output.samples == 6
    assert settings.effects.motion_blur.shutter_angle == 90.0

    (tmp_path / "manimx.json").write_text(json.dumps({"format": 2, "name": "old", "effects": {"motion_blur": {**blurred, "enabled": False}}}))
    assert Project.load(tmp_path).settings.output.samples == 1  # the samples only counted with motion blur on

    (tmp_path / "manimx.json").write_text(json.dumps({"format": 2, "name": "x", "effects": {"shadows": {"samples": 0}}}))
    with pytest.raises(ProjectError, match='"effects.shadows.samples" must be at least 1'):
        Project.load(tmp_path)
