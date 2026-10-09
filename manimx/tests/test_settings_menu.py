"""The settings menu: the tree model, the terminal app and the command."""

import json

import pytest

from manimx.cli import main
from manimx.cli.settings_menu import SettingsMenu, lines, print_settings, run_menu
from manimx.settings import Settings


def labels(menu: SettingsMenu) -> list[str]:
    return [".".join(row.entry.path) for row in menu.rows()]


def go_to(menu: SettingsMenu, path: str) -> None:
    menu.cursor = labels(menu).index(path)


def press(menu: SettingsMenu, *keys: str) -> None:
    for key in keys:
        menu.key(key)


def text(menu: SettingsMenu) -> str:
    return "".join(piece for _, piece in lines(menu))


def test_sections_open_and_close() -> None:
    menu = SettingsMenu(Settings())
    assert labels(menu) == ["output", "effects", "lighting", "performance"]

    go_to(menu, "effects")
    press(menu, "right")
    assert labels(menu)[2:6] == ["effects.motion_blur", "effects.bloom", "effects.shadows", "effects.tone_map"]
    press(menu, "down", "enter")  # enter opens a section too
    assert "effects.motion_blur.shutter_angle" in labels(menu)
    assert "effects.motion_blur.enabled" not in labels(menu)  # shown on the section's line

    go_to(menu, "effects.motion_blur.samples")
    press(menu, "left")  # from a value: close its section, select it
    assert labels(menu)[menu.cursor] == "effects.motion_blur" and "effects.motion_blur.samples" not in labels(menu)
    press(menu, "up", "left")
    assert labels(menu) == ["output", "effects", "lighting", "performance"]
    press(menu, "up", "up", "up", "down")
    assert menu.cursor == 1


def test_switches_choices_and_edits() -> None:
    menu = SettingsMenu(Settings(), expanded={("effects",), ("effects", "motion_blur"), ("effects", "shadows")})

    go_to(menu, "effects.bloom")
    press(menu, "space")
    assert menu.settings.effects.bloom.enabled and "[on]" in text(menu)

    go_to(menu, "effects.shadows.resolution")
    press(menu, "space")  # steps through the choices
    assert menu.settings.effects.shadows.resolution == 4096

    go_to(menu, "effects.motion_blur.shutter_angle")
    press(menu, "enter", "backspace", "backspace", "backspace", "7", "2", "0")
    assert "720█" in text(menu)
    press(menu, "enter")
    assert menu.settings.effects.motion_blur.shutter_angle == 720.0
    assert "0.03333 s exposure at 60 fps" in text(menu)  # the note follows the value

    press(menu, "enter", "x", "enter")  # "720x" is no number: the edit line stays
    assert menu.editing == "720x" and "must be number" in menu.message
    press(menu, "escape")
    assert menu.editing is None and menu.settings.effects.motion_blur.shutter_angle == 720.0

    press(menu, "r")
    assert menu.settings.effects.motion_blur.shutter_angle == 180.0 and "default" in menu.message

    go_to(menu, "effects.shadows.samples")
    press(menu, "enter", "backspace", "backspace", "9", "9", "enter")
    assert "at most 64" in menu.message and menu.settings.effects.shadows.samples == 16
    press(menu, "escape")

    go_to(menu, "effects.motion_blur")
    press(menu, "space")
    assert not menu.settings.effects.motion_blur.enabled


def test_values_of_every_type_switch() -> None:
    menu = SettingsMenu(Settings(), expanded={("output",), ("effects",), ("effects", "tone_map")})

    go_to(menu, "output.audio")
    press(menu, "enter")
    assert menu.settings.output.audio is False and "no" in text(menu)
    go_to(menu, "effects.tone_map.operator")
    press(menu, "enter")
    assert menu.settings.effects.tone_map.operator == "aces"
    go_to(menu, "output.background")
    press(menu, "space", "space", "x", "enter")  # a space is typed while editing
    assert "not a color" in menu.message and menu.editing == "#0b1020 x"
    press(menu, "backspace", "backspace", "enter")
    assert menu.settings.output.background == "#0b1020"
    go_to(menu, "effects")
    press(menu, "space")  # sections without a switch just open and close
    assert ("effects",) not in menu.expanded
    press(menu, "r", "z")  # nothing to reset on a section; unknown keys do nothing
    assert not menu.message


def test_quitting_asks_once_about_unsaved_changes() -> None:
    menu = SettingsMenu(Settings(), expanded={("output",)})
    go_to(menu, "output.audio")
    press(menu, "space", "q")
    assert not menu.done and "unsaved changes" in menu.message
    press(menu, "q")
    assert menu.done and not menu.save

    saving = SettingsMenu(Settings())
    press(saving, "s")
    assert saving.done and saving.save


def test_the_terminal_app_forwards_keys() -> None:
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    menu = SettingsMenu(Settings(), expanded={("effects",)})
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[B\x1b[B\x1b[B ")  # down to bloom, switch it on
        pipe.send_text("s")
        run_menu(menu, input=pipe, output=DummyOutput())
    assert menu.settings.effects.bloom.enabled and menu.save

    menu = SettingsMenu(Settings())
    with create_pipe_input() as pipe:
        pipe.send_text("\x03")  # Ctrl+C leaves without saving
        run_menu(menu, input=pipe, output=DummyOutput())
    assert menu.done and not menu.save


def test_print_documents_every_setting() -> None:
    printed = print_settings(Settings())
    assert "shutter_angle = 180.0°  Exposure as a share of a frame" in printed
    assert "resolution = 2048px" in printed and "one of 512, 1024, 2048, 4096" in printed
    assert "  motion_blur: Average renders over the exposure time." in printed


def test_settings_command_prints_without_a_terminal(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "manimx.json").write_text(json.dumps({"format": 2, "name": "x", "effects": {"bloom": {"enabled": True}}}))
    assert main(["settings"]) == 0  # pytest has no terminal: the printout
    out = capsys.readouterr().out
    assert "bloom: Glow" in out and "enabled = true" in out


@pytest.mark.parametrize(("keys", "saved", "said"), [("s", True, "UPDATE"), ("q", False, "without saving"), ("", False, "Nothing changed")])
def test_settings_command_saves_what_the_menu_changed(tmp_path, monkeypatch, capsys, keys, saved, said) -> None:
    from manimx.cli import settings_menu, ui

    monkeypatch.chdir(tmp_path)
    (tmp_path / "manimx.json").write_text(json.dumps({"format": 2, "name": "x"}))
    monkeypatch.setattr(ui.Interaction, "possible", classmethod(lambda cls: True))

    def fake_run(menu):
        if keys:
            menu.settings.effects.bloom.enabled = True
            press(menu, *keys, *keys)  # q twice: quit without saving
        else:
            press(menu, "s")
        return menu

    monkeypatch.setattr(settings_menu, "run_menu", fake_run)
    assert main(["settings"]) == 0
    assert said in capsys.readouterr().out
    data = json.loads((tmp_path / "manimx.json").read_text())
    assert data.get("effects", {}).get("bloom", {}).get("enabled", False) is saved


def test_hints_say_what_the_keys_do_here() -> None:
    menu = SettingsMenu(Settings(), expanded={("effects",), ("effects", "motion_blur"), ("effects", "shadows"), ("output",)})

    go_to(menu, "effects.motion_blur")
    assert menu.hints() == [[("space", "switch off"), ("←", "close")], [("↑↓", "move"), ("q", "quit")]]
    go_to(menu, "lighting")
    assert menu.hints()[0] == [("→", "open")]
    go_to(menu, "effects.shadows.resolution")
    assert menu.hints()[0][0] == ("space", "next: 4096px")
    go_to(menu, "output.audio")
    assert menu.hints()[0][0] == ("space", "switch to no")

    go_to(menu, "effects.motion_blur.shutter_angle")
    menu.settings.effects.motion_blur.shutter_angle = 720.0
    assert menu.hints() == [
        [("enter", "edit"), ("r", "default: 180°"), ("←", "close section")],
        [("↑↓", "move"), ("s", "save and quit"), ("q", "quit without saving")],
    ]
    press(menu, "enter")
    assert menu.hints() == [[("enter", "set"), ("esc", "cancel"), ("⌫", "delete")], [("", "a number, at least 0")]]
    assert " enter  set" in text(menu)
    press(menu, "escape")

    go_to(menu, "effects.shadows.samples")
    press(menu, "enter")
    assert menu.hints()[1] == [("", "a whole number, at least 1 and at most 64")]
    press(menu, "escape")
    go_to(menu, "output.background")
    press(menu, "enter")
    assert "#rrggbb" in menu.hints()[1][0][1]
