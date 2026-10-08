"""The manimx CLI end to end: new project, generate scenes, list, render."""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from manimx.cli import main
from manimx.cli.scaffold import SceneName
from manimx.project import Project, ProjectError, ProjectSource

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
TINY = ["--width", "64", "--height", "36", "--fps", "5", "--motion-blur", "1"]


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A fresh example project, with the working directory inside it."""
    monkeypatch.chdir(tmp_path)
    assert main(["new", "demo", "--example", "--resolution", "720p", "--manimx-source", "/opt/manimx"]) == 0
    monkeypatch.chdir(tmp_path / "demo")
    return tmp_path / "demo"


def config(root: Path) -> dict:
    return json.loads((root / "manimx.json").read_text())


def test_help_and_version(capsys) -> None:
    assert main(["--help"]) == 0
    assert "render" in capsys.readouterr().out
    assert main(["--version"]) == 0
    assert "manimx 0.1.0" in capsys.readouterr().out


def test_new_creates_a_project_with_example_scenes(capsys, project) -> None:
    data = config(project)
    assert data["format"] == 1 and data["$schema"] == "./.manimx/project.schema.json"
    assert data["output"]["width"] == 1280 and data["output"]["fps"] == 30 and data["camera"]["frame_height"] == 8.0
    assert data["performance"] == {"workers": 1, "frames_in_flight": 3}
    assert list(data["scenes"]) == ["Intro", "Bounce"] and [c["scene"] for c in data["timeline"]["clips"]] == ["Intro", "Bounce"]
    schema = json.loads((project / ".manimx" / "project.schema.json").read_text())
    assert schema["properties"]["output"]["properties"]["fps"] == {
        "type": "integer", "default": 60, "description": "Frames per second.", "minimum": 1,
    }  # fmt: skip
    assert (project / "scenes" / "intro.py").is_file() and (project / "assets").is_dir()
    assert 'manimx = { path = "/opt/manimx", editable = true }' in (project / "pyproject.toml").read_text()
    assert "uv run manimx render --preview" in capsys.readouterr().out


def test_new_refuses_non_empty_folders_and_needs_a_name(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file").write_text("x")

    assert main(["new", "taken", "--empty"]) == 1
    assert "not empty" in capsys.readouterr().err
    assert main(["new"]) == 1  # no terminal: nothing to ask
    assert "missing project name" in capsys.readouterr().err
    assert main(["new", "fine", "--empty", "--resolution", "8k"]) == 1
    assert "unknown resolution" in capsys.readouterr().err
    assert main(["-y", "new", "empty", "--empty"]) == 0
    assert config(tmp_path / "empty")["scenes"] == {}


def test_generate_registers_scenes_everywhere(project, capsys) -> None:
    assert main(["g", "scene", "title-card", "--kind", "simulation"]) == 0
    out = capsys.readouterr().out
    assert "CREATE scenes/title_card.py" in out and "UPDATE manimx.json" in out
    assert config(project)["scenes"]["TitleCard"] == {"module": "scenes.title_card", "class": "TitleCard"}
    assert config(project)["timeline"]["clips"][-1] == {"scene": "TitleCard"}
    assert "Simulation(step" in (project / "scenes" / "title_card.py").read_text()

    assert main(["generate", "scene", "Outro", "--no-timeline"]) == 0
    assert "Outro" in config(project)["scenes"] and config(project)["timeline"]["clips"][-1]["scene"] == "TitleCard"

    for arguments, message in (
        (["g", "scene", "intro"], "already exists"),
        (["g", "scene", "3d"], "not a usable scene name"),
        (["g", "scene", "class"], "keyword"),
        (["g", "scene", "x", "--kind", "opera"], "unknown scene kind"),
    ):
        assert main(arguments) == 1
        assert message in capsys.readouterr().err


def test_scene_names() -> None:
    assert SceneName.parse("title-card") == SceneName("TitleCard", "title_card")
    assert SceneName.parse("TitleCard") == SceneName("TitleCard", "title_card")
    assert SceneName.parse("intro 2") == SceneName("Intro2", "intro_2")


def test_remove_keeps_or_deletes_the_module(project, capsys) -> None:
    assert main(["remove", "scene", "Intro", "--keep"]) == 0
    assert "Intro" not in config(project)["scenes"] and (project / "scenes" / "intro.py").exists()
    assert main(["remove", "scene", "Bounce"]) == 0  # no terminal: keep by default
    assert (project / "scenes" / "bounce.py").exists()

    main(["g", "scene", "outro"])
    assert main(["remove", "scene", "Outro", "--delete"]) == 0
    assert not (project / "scenes" / "outro.py").exists()
    assert main(["remove", "scene", "Outor"]) == 1
    assert "no scene named" in capsys.readouterr().err


def test_list_shows_scenes_and_resolved_clip_times(project, capsys) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "scenes.bounce" in out and "4.00 s" in out  # Bounce starts when Intro ends
    assert main(["list", "--no-build"]) == 0
    assert "after previous" in capsys.readouterr().out


@needs_ffmpeg
def test_render_the_timeline_a_scene_and_stills(project, capsys) -> None:
    assert main(["render", *TINY, "--end", "1"]) == 0
    assert (project / "renders" / "demo.mp4").stat().st_size > 0
    assert main(["render", "Bounce", *TINY, "--end", "1", "--png", "-o", "frames"]) == 0
    assert len(list((project / "frames").glob("*.png"))) == 6
    assert main(["render", "--preview", "--end", "0.2", "-o", "preview.mp4"]) == 0
    assert (project / "preview.mp4").exists()

    assert main(["still", "Intro", "-t", "1.5", "--width", "64", "--height", "36"]) == 0
    assert Image.open(project / "renders" / "Intro-1.5s.png").size == (64, 36)
    assert main(["still", "-t", "999"]) == 1
    assert "nothing to show" in capsys.readouterr().err


@needs_ffmpeg
def test_render_with_worker_processes(project) -> None:
    assert main(["render", *TINY, "--start", "4", "--end", "5", "-w", "2", "-o", "workers.mp4"]) == 0
    assert (project / "workers.mp4").stat().st_size > 0


def test_errors_point_at_the_problem(project, tmp_path, monkeypatch, capsys) -> None:
    (project / "scenes" / "intro.py").write_text(
        "from manimx.animation import Scene, Square\n\n"
        "class Intro(Scene):\n"
        "    def construct(self):\n"
        "        Square().animate.shfit((1, 0, 0))\n"
    )
    assert main(["render", "Intro", *TINY]) == 1
    err = capsys.readouterr().err
    assert "Intro failed while it was being prepared" in err and "intro.py" in err and "shfit" in err

    main(["g", "scene", "broken"])
    (project / "scenes" / "broken.py").write_text("def broken(:\n")
    capsys.readouterr()
    assert main(["render", "Bounce", *TINY, "--end", "0"]) == 0  # other scenes still work
    assert main(["render", "Broken", *TINY]) == 1
    assert "Broken failed while it was imported" in capsys.readouterr().err

    assert main(["render", "Intor"]) == 1
    assert "did you mean 'Intro'" in capsys.readouterr().err

    monkeypatch.chdir(tmp_path)
    assert main(["list"]) == 1
    assert "no manimx.json" in capsys.readouterr().err
    assert main(["--traceback", "render", "--end", "x"]) == 2  # usage errors come from Click


@pytest.mark.parametrize(
    ("content", "message", "hint"),
    [
        ("{not json", "cannot read", None),
        ({"format": 99, "name": "x"}, "has format 99", None),
        ({"format": 1}, '"name" must be a string', None),
        ({"format": 1, "name": "x", "ouput": {}}, 'unknown setting "ouput"', 'did you mean "output"?'),
        ({"format": 1, "name": "x", "output": {"fsp": 30}}, 'unknown setting "output.fsp"', 'did you mean "fps"?'),
        ({"format": 1, "name": "x", "output": {"fps": "60"}}, '"output.fps" must be integer, got "60"', None),
        ({"format": 1, "name": "x", "output": {"audio": 1}}, '"output.audio" must be boolean', None),
        ({"format": 1, "name": "x", "output": {"width": True}}, '"output.width" must be integer', None),
        ({"format": 1, "name": "x", "camera": {"frame_height": -1}}, "must be at least 0", None),
        ({"format": 1, "name": "x", "camera": 5}, '"camera" must be an object', None),
        ({"format": 1, "name": "x", "scenes": {"A": {"modul": "a"}}}, 'unknown setting "scenes.A.modul"', None),
        ({"format": 1, "name": "x", "scenes": {"A": {}}}, "invalid", None),
        ({"format": 1, "name": "x", "timeline": {"clips": [{"sceen": "A"}]}}, 'unknown setting "timeline.clips[0].sceen"', None),
    ],
)
def test_project_files_are_validated(tmp_path, content, message, hint) -> None:
    (tmp_path / "manimx.json").write_text(content if isinstance(content, str) else json.dumps(content))
    with pytest.raises(ProjectError, match=message.replace("[", r"\[").replace("]", r"\]")) as error:
        Project.load(tmp_path)
    if hint is not None:
        assert error.value.hint == hint


def test_timeline_entries_round_trip_and_never_ending_clips(project) -> None:
    (project / "scenes" / "waiting.py").write_text(
        "from manimx.animation import Scene\n\n"
        "class Waiting(Scene):\n"
        "    def construct(self):\n"
        "        self.wait(1)\n"
        "        self.wait_for('click')\n"
        "        self.wait(1)\n"
    )
    loaded = Project.find()
    loaded.add_scene("Waiting", "scenes.waiting")
    loaded.timeline[0].layer, loaded.timeline[0].cut, loaded.timeline[1].start = 2, (0.5, 2.0), 1.0
    loaded.add_scene("Late", "scenes.intro", "Intro")
    loaded.save()

    again = Project.find()
    assert again.timeline[0].cut == (0.5, 2.0) and again.timeline[0].layer == 2 and again.timeline[1].start == 1.0
    with pytest.raises(ProjectError, match="never fires"):
        again.build_timeline()
    again.events["click"] = 3.0
    timeline = again.build_timeline()
    assert [clip.name for clip in timeline.clips] == ["Intro", "Bounce", "Waiting", "Late"]
    # Bounce starts at 1 s and ends at 6 s; Waiting plays 1 s, its click (3 s)
    # came early, so it continues at 7 s and ends at 8 s; Late follows.
    assert timeline.clips[3].start == pytest.approx(8.0)
    assert ProjectSource(str(project), "Intro")().duration > 0


def test_missing_arguments_are_asked_for_in_a_terminal(tmp_path, monkeypatch, capsys) -> None:
    from manimx.cli import ui

    answers = iter(["asked", "empty", "4k", "#000000", "scene-two", "simulation", True])

    class Question:
        def __init__(self, *args, **kwargs):
            pass

        def ask(self):
            return next(answers)

    monkeypatch.setattr(ui.Interaction, "possible", classmethod(lambda cls: True))
    for kind in ("text", "select", "confirm"):
        monkeypatch.setattr(ui.questionary, kind, Question)
    monkeypatch.chdir(tmp_path)

    assert main(["new"]) == 0
    data = config(tmp_path / "asked")
    assert data["output"]["width"] == 3840 and data["output"]["background"] == "#000000" and data["scenes"] == {}
    monkeypatch.chdir(tmp_path / "asked")
    assert main(["g", "scene"]) == 0
    assert config(tmp_path / "asked")["scenes"] == {"SceneTwo": {"module": "scenes.scene_two", "class": "SceneTwo"}}
    assert main(["remove", "scene", "SceneTwo"]) == 0  # confirmed: the module is deleted
    assert not (tmp_path / "asked" / "scenes" / "scene_two.py").exists()

    monkeypatch.setattr(Question, "ask", lambda self: None)  # Ctrl+C in a prompt
    assert main(["g", "scene"]) == 130
    assert "Cancelled" in capsys.readouterr().err
