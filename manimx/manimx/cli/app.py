"""The ``manimx`` commands."""

from __future__ import annotations

import copy
import dataclasses
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, ProgressColumn, SpinnerColumn, Task, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree

import manimx

from ..animation.timing import NEVER, Timeline
from ..project import PROJECT_FILE, Project, ProjectError
from ..settings import OutputSettings, Settings, SettingsError, set_value
from . import scaffold, templates, ui

app = typer.Typer(
    name="manimx",
    help="Create, extend and render manimx projects.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
generate_app = typer.Typer(help="Add something to the project (alias: [bold]g[/]).", no_args_is_help=True)
remove_app = typer.Typer(help="Remove something from the project.", no_args_is_help=True)
app.add_typer(generate_app, name="generate")
app.add_typer(generate_app, name="g", hidden=True)
app.add_typer(remove_app, name="remove")


class Options:
    traceback = False


def _version(value: bool) -> None:
    if value:
        ui.console.print(f"manimx {manimx.__version__}")
        raise typer.Exit


@app.callback()
def root(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")] = False,
    no_input: Annotated[bool, typer.Option("--no-input", "-y", help="Never ask; use defaults.")] = False,
    traceback: Annotated[bool, typer.Option("--traceback", help="Show full Python tracebacks on errors.")] = False,
) -> None:
    ui.Interaction.allowed = not no_input
    Options.traceback = traceback


# --------------------------------------------------------------------------
# new
# --------------------------------------------------------------------------


@app.command()
def new(
    name: Annotated[str | None, typer.Argument(help="Project name (also its folder).", show_default=False)] = None,
    example: Annotated[bool | None, typer.Option("--example/--empty", help="Start with example scenes, or empty.", show_default=False)] = None,
    resolution: Annotated[str | None, typer.Option(help="1080p, 720p or 4k.", show_default=False)] = None,
    background: Annotated[str | None, typer.Option(help="Background color.", show_default=False)] = None,
    directory: Annotated[Path | None, typer.Option(help="Parent folder.", show_default=False)] = None,
    manimx_source: Annotated[str | None, typer.Option(help="Path of a manimx checkout to depend on.", show_default=False)] = None,
) -> None:
    """Create a new project."""
    name = name or ui.ask_text("Project name:", missing="project name")
    if example is None:
        example = ui.ask_choice("Start with:", [("example", "an animation and a simulation scene"), ("empty", "no scenes")], "example") == "example"
    if resolution is None:
        resolution = ui.ask_choice("Resolution:", [("1080p", "1920×1080 · 60 fps"), ("720p", "1280×720 · 30 fps"), ("4k", "3840×2160 · 60 fps")], "1080p")
    if resolution not in scaffold.RESOLUTIONS:
        raise ProjectError(f"unknown resolution {resolution!r}", hint="use 1080p, 720p or 4k")
    background = background or ui.ask_text("Background:", default=OutputSettings().background)
    width, height, fps = scaffold.RESOLUTIONS[resolution]
    options = OutputSettings(width=width, height=height, fps=fps, background=background)

    folder = (directory or Path.cwd()) / name
    project, changes = scaffold.create_project(folder, name, example, options, manimx_source)
    ui.changes(changes, folder)
    ui.console.print()
    ui.success(f"Created [bold]{name}[/]")
    ui.console.print(_tree(project))
    ui.console.print("\n  [bold]Next steps[/]")
    for step in (f"cd {_shown(folder)}", "uv sync", "uv run manimx render --preview"):
        ui.console.print(f"    [{ui.ACCENT}]{step}[/]")


def _tree(project: Project) -> Tree:
    tree = Tree(f"[bold]{project.root.name}[/]", guide_style="dim")
    for path in sorted(project.root.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            branch = tree.add(f"[bold]{path.name}/[/]")
            for child in sorted(path.iterdir()):
                if not child.name.startswith(".") and child.name != "__pycache__":
                    scene = next((n for n, e in project.scenes.items() if e.module == f"{path.name}.{child.stem}"), None)
                    branch.add(child.name + (f"  [dim]{scene}[/]" if scene else ""))
        else:
            tree.add(path.name)
    return tree


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# generate / remove
# --------------------------------------------------------------------------


@generate_app.command("scene")
def generate_scene(
    name: Annotated[str | None, typer.Argument(help="Scene name, e.g. title-card or TitleCard.", show_default=False)] = None,
    kind: Annotated[str | None, typer.Option(help="animation or simulation.", show_default=False)] = None,
    timeline: Annotated[bool, typer.Option("--timeline/--no-timeline", help="Append the scene to the timeline.")] = True,
) -> None:
    """Create a scene: its module, its import and its place on the timeline."""
    project = Project.find()
    name = name or ui.ask_text("Scene name:", missing="scene name")
    if kind is None:
        kind = ui.ask_choice(
            "Kind:",
            [("animation", "keyframed scene with play() and wait()"), ("simulation", "step function baked into keyframes")],
            "animation",
        )
    ui.changes(scaffold.generate_scene(project, name, kind, timeline), project.root)


@remove_app.command("scene")
def remove_scene(
    name: Annotated[str, typer.Argument(help="Scene name, as registered.")],
    delete: Annotated[bool | None, typer.Option("--delete/--keep", help="Also delete the scene's module.", show_default=False)] = None,
) -> None:
    """Unregister a scene and take it off the timeline."""
    project = Project.find()
    project._check_scene(name)
    if delete is None:
        delete = ui.confirm(f"Also delete the module of {name}?", default=False)
    ui.changes(scaffold.remove_scene(project, name, delete), project.root)


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


@app.command("settings")
def settings_(
    print_: Annotated[bool, typer.Option("--print", help="Print every setting with its docs instead of the menu.")] = False,
) -> None:
    """Change the project's settings in a menu: output, effects (motion blur,
    bloom, shadows, tone mapping), lighting and performance."""
    from .settings_menu import SettingsMenu, print_settings, run_menu

    project = Project.find()
    if print_ or not ui.Interaction.possible():
        ui.console.print(print_settings(project.settings), markup=False)
        return
    menu = SettingsMenu(copy.deepcopy(project.settings), expanded={("effects",)})
    run_menu(menu)
    if menu.save and menu.dirty:
        project.settings = menu.settings
        project.save()
        ui.changes([scaffold.Change("update", project.root / PROJECT_FILE, "settings")], project.root)
    elif menu.save:
        ui.console.print("[dim]Nothing changed.[/]")
    else:
        ui.console.print("[dim]Left without saving.[/]")


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------


@app.command("list")
def list_(
    build: Annotated[bool, typer.Option("--build/--no-build", help="Prepare scenes to show clip times.")] = True,
) -> None:
    """Show the project's scenes and timeline."""
    project = Project.find()
    ui.header(project.name, f"{project.output.width}×{project.output.height}", f"{project.output.fps} fps")
    scenes = Table(title="scenes", title_justify="left", title_style="bold", box=None, padding=(0, 2))
    scenes.add_column("scene", style="bold")
    scenes.add_column("module", style="dim")
    for name, entry in project.scenes.items():
        scenes.add_row(name, entry.module)
    ui.console.print(scenes)
    ui.console.print()
    ui.console.print(_timeline_table(project, _build(project) if build else None))


def _build(project: Project) -> Timeline:
    with ui.console.status("Preparing scenes…", spinner="dots") as status:
        timeline = project.build_timeline(lambda entry: status.update(f"Preparing [bold]{entry.scene}[/]…"))
    return timeline


def _timeline_table(project: Project, timeline: Timeline | None) -> Table:
    table = Table(title="timeline", title_justify="left", title_style="bold", padding=(0, 2))
    table.add_column("clip", style="bold")
    table.add_column("scene")
    table.add_column("start", justify="right")
    table.add_column("end", justify="right")
    table.add_column("layer", justify="right", style="dim")
    clips = timeline.clips if timeline is not None else [None] * len(project.timeline)
    for entry, clip in zip(project.timeline, clips, strict=True):
        start = "after previous" if entry.start is None else f"{entry.start:.2f} s"
        end = "–"
        if clip is not None:
            start = f"{clip.start:.2f} s"
            end = "waits" if clip.end == NEVER else f"{clip.end:.2f} s"
        table.add_row(clip.name if clip is not None else entry.name or entry.scene, entry.scene, start, end, str(entry.layer))
    if not project.timeline:
        table.add_row("[dim]empty: add a scene with `manimx g scene`[/]", "", "", "", "")
    return table


# --------------------------------------------------------------------------
# render / still
# --------------------------------------------------------------------------


class _RateColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        return Text("" if speed is None else f"{speed:.0f} fps", style="dim")


@app.command()
def render(
    scene: Annotated[str | None, typer.Argument(help="Render one scene instead of the timeline.", show_default=False)] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file (or folder with --png).", show_default=False)] = None,
    preview: Annotated[bool, typer.Option("--preview", "-p", help="Fast: 960×540, 30 fps, no motion blur, PCF shadows.")] = False,
    png: Annotated[bool, typer.Option("--png", help="Write a PNG sequence instead of a video.")] = False,
    window: Annotated[bool, typer.Option("--window", help="Show the frames in a window as they render.")] = False,
    write: Annotated[
        bool | None, typer.Option("--write/--no-write", help="Write the video or PNGs (default: unless --window).", show_default=False)
    ] = None,
    width: Annotated[int | None, typer.Option(show_default=False)] = None,
    height: Annotated[int | None, typer.Option(show_default=False)] = None,
    fps: Annotated[int | None, typer.Option(show_default=False)] = None,
    motion_blur: Annotated[int | None, typer.Option(help="Renders averaged per frame (1: off).", show_default=False)] = None,
    shutter_angle: Annotated[
        float | None, typer.Option(help="Exposure in degrees of a frame: 180 half a frame, 720 two frames.", show_default=False)
    ] = None,
    workers: Annotated[int | None, typer.Option("--workers", "-w", help="Render processes.", show_default=False)] = None,
    start: Annotated[float, typer.Option(help="First second to render.")] = 0.0,
    end: Annotated[float | None, typer.Option(help="Last second to render.", show_default=False)] = None,
    audio: Annotated[bool | None, typer.Option("--audio/--no-audio", help="Include sounds (default: output.audio).", show_default=False)] = None,
    open_: Annotated[bool, typer.Option("--open", "-O", help="Open the result when done.")] = False,
) -> None:
    """Render the timeline (or one scene) to a video or PNG sequence, a
    window, or both."""
    from ..drivers import ImageSequence, Tee, VideoFile, Window
    from ..drivers import render as render_frames

    write = not window if write is None else write
    if not write and not window:
        raise ProjectError("--no-write without --window renders nothing", hint="add --window")
    if not write and (png or output is not None or open_):
        raise ProjectError("--png, --output and --open need a written file", hint="add --write")
    project = Project.find()
    chosen = copy.deepcopy(project.settings)
    if preview:
        chosen.output.width, chosen.output.height, chosen.output.fps = 960, 540, 30
        chosen.effects.motion_blur.enabled = False
        chosen.effects.shadows.method = "pcf"
    _override(chosen, ("output", "width"), width)
    _override(chosen, ("output", "height"), height)
    _override(chosen, ("output", "fps"), fps)
    _override(chosen, ("output", "audio"), audio)
    _override(chosen, ("effects", "motion_blur", "shutter_angle"), shutter_angle)
    if motion_blur is not None:
        _override(chosen, ("effects", "motion_blur", "samples"), motion_blur)
        chosen.effects.motion_blur.enabled = motion_blur > 1
    options = chosen.output
    workers = workers or project.performance.workers
    settings = dataclasses.replace(project.render_settings(chosen), start=start, end=end)

    what = scene or "timeline"
    samples = chosen.motion_blur_samples
    blur = f"motion blur ×{samples} · {chosen.effects.motion_blur.shutter_angle:g}°" if samples > 1 else "no motion blur"
    ui.header(project.name, what, f"{options.width}×{options.height}", f"{options.fps} fps", blur, f"{workers} worker{'s' * (workers != 1)}")
    if scene is None:
        source: Any = _build(project)
        ui.console.print(_timeline_table(project, source))
        if not source.clips:
            raise ProjectError("the timeline is empty", hint="add a scene with `manimx g scene`")
    else:
        with ui.console.status(f"Preparing [bold]{scene}[/]…", spinner="dots"):
            source = project.prepare(scene)
    if workers > 1:
        source = project.source(scene)  # workers load the project themselves

    sinks: list[Any] = []
    if write:
        folder = project.root / options.directory
        target = output or folder / (f"{scene or project.name}" + ("" if png else ".mp4"))
        target.parent.mkdir(parents=True, exist_ok=True)
        sinks.append(ImageSequence(target) if png else VideoFile(target))
    if window:
        sinks.append(Window(title=f"{project.name} · {what}"))
    sink = sinks[0] if len(sinks) == 1 else Tee(tuple(sinks))

    columns = (SpinnerColumn(), TextColumn("Rendering"), BarColumn(), MofNCompleteColumn(), _RateColumn(), TimeRemainingColumn())
    began = time.perf_counter()
    with Progress(*columns, console=ui.console, transient=True) as progress:
        task = progress.add_task("render", total=None)
        stats = render_frames(
            source, sink, settings, workers=workers, progress=lambda done, total: progress.update(task, completed=done, total=total)
        )
    elapsed = time.perf_counter() - began
    seconds = stats.frames / options.fps
    if not write:
        pace = "real time" if elapsed <= seconds * 1.05 else f"{seconds / elapsed:.2f}× real time"
        ui.success(f"[bold]{stats.frames} frames[/]  [dim]{seconds:.1f} s shown in {elapsed:.1f} s ({pace})[/]")
        return
    size = target.stat().st_size if target.is_file() else sum(p.stat().st_size for p in target.iterdir())
    ui.success(
        f"[bold]{_shown(target)}[/]  [dim]{stats.frames} frames · {seconds:.1f} s · "
        f"{size / 1e6:.1f} MB · {elapsed:.1f} s ({stats.frames / elapsed:.0f} fps)[/]"
    )
    if open_:
        _open(target)


@app.command()
def still(
    scene: Annotated[str | None, typer.Argument(help="One scene instead of the timeline.", show_default=False)] = None,
    at: Annotated[float, typer.Option("--at", "-t", help="Time in seconds.")] = 0.0,
    output: Annotated[Path | None, typer.Option("--output", "-o", show_default=False)] = None,
    width: Annotated[int | None, typer.Option(show_default=False)] = None,
    height: Annotated[int | None, typer.Option(show_default=False)] = None,
    open_: Annotated[bool, typer.Option("--open", "-O", help="Open the result when done.")] = False,
) -> None:
    """Render a single frame as a PNG."""
    from PIL import Image

    from ..drivers import FrameList
    from ..drivers import render as render_frames

    project = Project.find()
    sizes = {key: value for key, value in {"width": width, "height": height}.items() if value is not None}
    chosen = copy.deepcopy(project.settings)
    chosen.effects.motion_blur.enabled = False
    chosen.output.audio = False
    for key, value in sizes.items():
        _override(chosen, ("output", key), value)
    settings = dataclasses.replace(project.render_settings(chosen), start=at, end=at)
    with ui.console.status("Preparing…", spinner="dots"):
        source = project.build_timeline() if scene is None else project.prepare(scene)
    if not 0.0 <= at <= source.duration:
        raise ProjectError(f"nothing to show at {at:g} s", hint=f"pick a time between 0 and {source.duration:.2f} s")
    frames = FrameList()
    render_frames(source, frames, settings)
    if not frames.frames:
        raise ProjectError(f"nothing to show at {at} s", hint="pick a time within the timeline")
    target = output or project.root / project.output.directory / f"{scene or project.name}-{at:g}s.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames.frames[0], "RGBA").save(target)
    ui.success(f"[bold]{_shown(target)}[/]  [dim]{scene or 'timeline'} at {at:g} s[/]")
    if open_:
        _open(target)


# --------------------------------------------------------------------------
# ui
# --------------------------------------------------------------------------


@app.command("ui")
def ui_command(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to serve on (0: any free port).")] = 0,
    open_: Annotated[bool, typer.Option("--open/--no-open", help="Open the page in the browser.")] = True,
    reload: Annotated[bool, typer.Option("--reload/--no-reload", help="Rebuild when a scene file or manimx.json is saved.")] = True,
) -> None:
    """Preview the timeline in the browser and scrub through it."""
    import webbrowser

    from ..web import PreviewServer, ProjectReloader

    project = Project.find()
    # With reload, a scene that fails to build is shown in the page (and
    # fixed by saving) instead of stopping the command.
    timeline = Timeline() if reload else _build(project)
    try:
        server = PreviewServer(timeline, _preview_settings(project), title=project.name, port=port)
    except OSError as error:
        raise ProjectError(f"cannot serve on port {port}: {error.strerror}", hint="pick another one with --port, or 0 for any free port") from None
    reloader = ProjectReloader(server, project.root, _preview_settings) if reload else None
    if reloader is not None:
        with ui.console.status("Preparing scenes…", spinner="dots"):
            reloader.reload()
        if server.error is not None:
            ui.console.print(f"[yellow]{escape(server.error.splitlines()[0])}[/]  [dim]shown in the page until a save fixes it[/]")
        reloader.start()
    watching = "  reloads on save" if reload else ""
    ui.success(f"[bold]{server.url}[/]  [dim]{project.name}{watching}; Ctrl+C stops[/]")
    if open_:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if reloader is not None:
            reloader.stop()
        server.close()


def _preview_settings(project: Project) -> Any:
    """The project's render settings for previews: one render per frame,
    PCF shadows."""
    chosen = copy.deepcopy(project.settings)
    chosen.effects.motion_blur.enabled = False
    chosen.effects.shadows.method = "pcf"
    chosen.output.audio = False
    return project.render_settings(chosen)


def _override(settings: Settings, path: tuple[str, ...], value: Any) -> None:
    """Set a setting from a command line option (if given), checked like manimx.json."""
    if value is not None:
        try:
            set_value(settings, path, value)
        except SettingsError as error:
            raise ProjectError(str(error).replace(f'"{".".join(path)}"', f"--{path[-1].replace('_', '-')}")) from None


def _open(path: Path) -> None:
    """Open a file (or a PNG sequence's folder) with the system's default app."""
    if typer.launch(str(path)) != 0:
        ui.console.print(f"[dim]could not open {_shown(path)}[/]")


__all__ = ["app", "templates"]
