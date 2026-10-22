"""Benchmark rendering on fixed scenes, on this tree or another manim tree.

Each scene is rendered in a process of its own through the public API
(``Scene.render()`` with the output format ``"none"``: every frame is
drawn, nothing is written), so the same scenes run on any manim tree
that has that format, for example trunk before the native renderer:

    python scripts/benchmark_render.py
    python scripts/benchmark_render.py --scene "1000 squares, still"
    python scripts/benchmark_render.py --manim ../manim-old --python ../manim-old/.venv/bin/python
    python scripts/benchmark_render.py --profile py-spy     # sampled time (--native: Rust too)
    python scripts/benchmark_render.py --profile memray     # Python and native allocations

    python scripts/benchmark_render.py --gpu low-power      # the integrated GPU

Reported per scene: the wall time of the render, the frames drawn, the
time per frame, the part of it the renderer itself draws ("draw ms", see
``time_drawing``) and the peak resident memory of the process. The header names the tree's
commit, the GPU and the laptop's power mode (switch it before a run);
the numbers are saved to ``benchmarks/results/`` as JSON. Profiles go to
``benchmarks/<profile>/``: py-spy writes speedscope files (open them at
https://www.speedscope.app), memray writes flame graphs as HTML.

py-spy is a command line tool (``uv tool install py-spy``); memray must
be installed in the benchmarked tree's Python (``uv pip install memray``)
so it can import that tree's manim. On a CPU with fast and slow cores, pin the
runs (``taskset -c 2``) and compare only runs made the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Scene name: (construct, what it measures). Each construct receives the
# scene and uses only API that every benchmarked tree has.
SCENES: dict[str, str] = {}


def scene(name: str, measures: str):
    def register(construct):
        SCENES[name] = measures
        CONSTRUCTS[name] = construct
        return construct

    return register


CONSTRUCTS: dict = {}


def squares(manim, count: int):
    columns = int(count**0.5)
    side = 7.0 / columns
    return manim.VGroup(
        *(
            manim.Square(
                side * 0.8, fill_color=manim.BLUE, fill_opacity=0.8, stroke_width=1
            )
            for _ in range(count)
        )
    ).arrange_in_grid(cols=columns, buff=side * 0.2)


for count in (100, 1000, 10000):

    @scene(f"{count} squares, still", f"{count} mobjects, nothing moves")
    def still(self, manim, count=count):
        self.add(squares(manim, count))
        self.wait(0.5)

    @scene(f"{count} squares, moving", f"{count} mobjects, all turn")
    def moving(self, manim, count=count):
        group = squares(manim, count)
        self.add(group)
        self.play(manim.Rotate(group, manim.PI / 4), run_time=0.5)


@scene("plot", "many strokes; one dot moves")
def plot(self, manim):
    axes = manim.Axes()
    self.add(manim.NumberPlane(), axes)
    for k in range(1, 4):
        self.add(axes.plot(lambda x, k=k: manim.np.sin(k * x)))
    dot = manim.Dot()
    self.add(dot)
    self.play(dot.animate.shift(manim.RIGHT * 2), run_time=0.5)


@scene("text", "many small fills (Text, no LaTeX); written in")
def text(self, manim):
    lines = manim.VGroup(
        *(
            manim.Text(f"Line {k}: the quick brown fox", font_size=24)
            for k in range(1, 13)
        )
    ).arrange(manim.DOWN, buff=0.15)
    self.play(manim.Write(lines), run_time=0.5)


@scene("3D", "meshes and lights; the camera turns")
def three_d(self, manim):
    self.set_camera_orientation(phi=70 * manim.DEGREES, theta=30 * manim.DEGREES)
    self.add(manim.Torus(), manim.Sphere(radius=0.7).shift(manim.UP * 1.5))
    self.move_camera(theta=60 * manim.DEGREES, run_time=0.5)


def run_scene(name: str, size: str, gpu: str | None) -> None:
    """Render one scene in this process and print its numbers as JSON."""
    import manim

    manim.np = __import__("numpy")
    if gpu:
        try:
            from manim.renderer import insanity_adapter
        except ImportError:
            pass  # a tree without the native renderer
        else:
            # The adapter makes its RenderConfig itself; give it the GPU.
            config_class = insanity_adapter.RenderConfig
            insanity_adapter.RenderConfig = lambda **values: config_class(
                gpu=gpu, **values
            )
    construct = CONSTRUCTS[name]
    base = manim.ThreeDScene if name == "3D" else manim.Scene
    width, height = (int(value) for value in size.split("x"))

    class Benchmark(base):
        def construct(self):
            construct(self, manim)

    draw_seconds = time_drawing(manim)
    frames = []
    with (
        tempfile.TemporaryDirectory() as media,
        manim.tempconfig(
            {
                "media_dir": media,
                "format": "none",
                "pixel_width": width,
                "pixel_height": height,
                "frame_rate": 30,
                "disable_caching": True,
                "verbosity": "ERROR",
                "progress_bar": "none",
            }
        ),
    ):
        benchmark = Benchmark()
        renderer = benchmark.renderer
        get_frame = renderer.get_frame

        def counted_get_frame(*args, **kwargs):
            frames.append(None)
            return get_frame(*args, **kwargs)

        renderer.get_frame = counted_get_frame
        start = time.perf_counter()
        benchmark.render()
        seconds = time.perf_counter() - start
    print(
        json.dumps(
            {
                "renderer": type(renderer).__name__,
                "seconds": seconds,
                "draw_seconds": draw_seconds[0],
                "frames": len(frames),
                "peak_mib": peak_mib(),
            }
        )
    )


def time_drawing(manim) -> list[float]:
    """Sum the time the renderer itself draws, without turning mobjects
    into what it draws, into the returned one-element list.

    - native renderer: ``WgpuRenderer.render`` (resolving the tree, packing
      the GPU records, the GPU passes, the read-back);
    - Cairo: the ``_CairoDrawingContext._display_*`` calls (Cairo filling
      and stroking the paths), not ``extract`` before them.
    """
    total = [0.0]

    def timed(function):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                total[0] += time.perf_counter() - start

        return wrapper

    try:
        from manim.renderer import insanity_adapter
    except ImportError:
        from manim.renderer.cairo import rendering

        context = rendering._CairoDrawingContext
        for name in ("_display_paths", "_display_point_clouds", "_display_images"):
            setattr(context, name, timed(getattr(context, name)))
        return total

    class TimedRenderer:
        """A WgpuRenderer whose render calls are timed (the class itself
        cannot be patched: it is a native class).
        """

        def __init__(self):
            self.renderer = native_class()
            self.render = timed(self.renderer.render)

    native_class = insanity_adapter.native.WgpuRenderer
    native_module = insanity_adapter.native

    class Native:
        def __getattr__(self, name):
            return (
                TimedRenderer
                if name == "WgpuRenderer"
                else getattr(native_module, name)
            )

    insanity_adapter.native = Native()
    return total


def peak_mib() -> float | None:
    """The peak resident memory of this process, where the system says."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux counts KiB, macOS bytes.
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", choices=list(SCENES), action="append")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument(
        "--manim", type=Path, default=ROOT, help="the manim tree to run"
    )
    parser.add_argument("--python", default=sys.executable, help="its Python")
    parser.add_argument("--profile", choices=["py-spy", "memray"])
    parser.add_argument(
        "--native",
        action="store_true",
        help="py-spy: sample the Rust frames too (much slower; use small scenes)",
    )
    parser.add_argument(
        "--gpu",
        choices=["high-performance", "low-power"],
        help="RenderConfig.gpu of the native renderer: the dedicated or the"
        " integrated GPU (default: the RenderConfig default; Cairo ignores it)",
    )
    parser.add_argument("--run", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.run:
        run_scene(arguments.run, arguments.size, arguments.gpu)
        return

    names = arguments.scene or list(SCENES)
    tree = arguments.manim.resolve()
    environment = {**os.environ, "PYTHONPATH": str(tree)}
    environment.pop("VIRTUAL_ENV", None)
    output = ROOT / "benchmarks" / (arguments.profile or "")
    if arguments.profile == "py-spy" and shutil.which("py-spy") is None:
        sys.exit("py-spy is not installed (uv tool install py-spy)")
    if arguments.profile:
        output.mkdir(parents=True, exist_ok=True)
    run = {
        "tree": str(tree),
        "commit": git_commit(tree),
        "size": arguments.size,
        "gpu": arguments.gpu or "default",
        "power": power_profile(),
        "profile": arguments.profile,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenes": {},
    }
    print(
        f"{tree} ({run['commit']}), {arguments.size}, GPU {run['gpu']},"
        f" power {run['power']}"
    )
    print(
        f"{'scene':<24} {'seconds':>8} {'frames':>7} {'ms/frame':>9}"
        f" {'draw ms':>8} {'peak MiB':>9}"
    )
    for name in names:
        command = [arguments.python, __file__, "--run", name, "--size", arguments.size]
        if arguments.gpu:
            command += ["--gpu", arguments.gpu]
        stem = name.replace(" ", "_").replace(",", "")
        if arguments.profile == "py-spy":
            native = ["--native"] if arguments.native else []
            command = [
                "py-spy",
                "record",
                *native,
                "--rate",
                "250",
                "--format",
                "speedscope",
                "--output",
                str(output / f"{stem}.json"),
                "--",
                *command,
            ]
        elif arguments.profile == "memray":
            capture = output / f"{stem}.bin"
            capture.unlink(missing_ok=True)
            # memray must run in the tree's own Python to import its manim.
            command = [
                arguments.python,
                "-m",
                "memray",
                "run",
                "--native",
                "--output",
                str(capture),
                *command[1:],
            ]
        result = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            cwd=arguments.manim,
        )
        lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
        if result.returncode or not lines:
            print(f"{name:<24} failed: {result.stderr.strip().splitlines()[-1:]}")
            continue
        numbers = json.loads(lines[-1])
        run["scenes"][name] = numbers
        frames = max(numbers["frames"], 1)
        per_frame = 1000 * numbers["seconds"] / frames
        draw_per_frame = 1000 * numbers["draw_seconds"] / frames
        peak = "-" if numbers["peak_mib"] is None else f"{numbers['peak_mib']:.0f}"
        print(
            f"{name:<24} {numbers['seconds']:>8.2f} {numbers['frames']:>7}"
            f" {per_frame:>9.1f} {draw_per_frame:>8.1f} {peak:>9}"
        )
        if arguments.profile == "memray":
            subprocess.run(
                [
                    "memray",
                    "flamegraph",
                    "--force",
                    "--output",
                    str(output / f"{stem}.html"),
                    str(capture),
                ],
                capture_output=True,
            )
    if not arguments.profile:
        results = ROOT / "benchmarks" / "results"
        results.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = (
            results
            / f"{stamp}_{run['commit']}_gpu-{run['gpu']}_power-{run['power']}.json"
        )
        path.write_text(json.dumps(run, indent=2))
        print(f"saved {path.relative_to(ROOT)}")


def git_commit(tree: Path) -> str:
    """The short commit of `tree`, with "+" when it has changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=tree,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=tree,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit + ("+" if changed else "") if commit else "unknown"


def power_profile() -> str:
    """The laptop's power mode, where the system says (Linux: the ACPI
    platform profile); "unknown" elsewhere.
    """
    try:
        return Path("/sys/firmware/acpi/platform_profile").read_text().strip()
    except OSError:
        return "unknown"


if __name__ == "__main__":
    main()
