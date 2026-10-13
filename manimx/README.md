# manimx

manimx is an animation engine in the spirit of
[Manim](https://www.manim.community/): you write scenes in Python, and
manimx renders them on the GPU, with 2D shapes, lit 3D meshes, shadows,
motion blur and sound. Scenes are put together into a project with a
timeline, rendered to video from the command line, and previewed in the
browser.

```python
from manimx.animation import Scene
from manimx.userland import Circle, FadeIn, Square, RIGHT

class Hello(Scene):
    def construct(self) -> None:
        square = Square(2.0, color="#58C4DD", fill_opacity=0.5)
        self.play(FadeIn(square))
        self.play(square.animate.shift(2 * RIGHT))
        self.add(Circle(1.0, color="#FC6255"))
        self.wait()
```

Start with [GETTING_STARTED.md](GETTING_STARTED.md): install, create a
project, write scenes, render.

## How it works, in one paragraph

`Scene.construct()` runs once and records what every animation does to
every property (a *signal*) and when. That record is handed to the
native core (`manimx._core`, written in Rust), which can then set every
signal to its value at any time, in any order, turn the scene at that
time into GPU records, and draw them. Because any time can be asked
for, scrubbing, retiming, cuts, motion blur and rendering on several
processes all come for free.

## Where things are

```text
manimx/            the Python package
  animation/       what scenes are made of
    signals/         the animatable values and how animations change them
    nodes/           the scene nodes: shapes, meshes, lights, groups, cameras
    scene.py         Scene: construct() once, evaluate at any time
    timing.py        Timeline and Clip: scenes placed on a global timeline
    plan.py          a prepared scene handed to the core as data
  userland/        what a user could have written: Circle, FadeIn, graphs,
                   SVG, Typst and Code, tables, matrices and charts,
                   glTF, rate functions, geometry
  robjects/        render state (what a scene shows at one time) and the
                   value types nodes take: materials, meshes, styles
  renderer/        the settings Python gives the renderer, and its shaders
  drivers/         rendering videos and stills: frames in flight, workers
  project.py       a project: manimx.json, its scenes and timeline
  cli/             the manimx command
  web/             the browser preview (manimx ui)
core/              the native core in Rust (evaluation, packing, rendering)
tests/             the tests, and the reference images in tests/references/
scripts/           benchmarks, rendering scripts, the renderer's pictures
example-project/   a finished example project
showcase/          a project showing every feature
```

Every package and module starts with a description of what it holds;
every Rust folder's `mod.rs` starts with a table of its modules.

## What to read

1. [GETTING_STARTED.md](GETTING_STARTED.md): using manimx.
2. [ARCHITECTURE.md](ARCHITECTURE.md): how the whole system fits
   together, which class owns what, and why.
3. [core/README.md](core/README.md) and [CORE.md](CORE.md): the Rust
   core, and the renderer: its stages and how a frame is drawn, pass by
   pass, with pictures.
4. [manimx/renderer/README.md](manimx/renderer/README.md): what the
   renderer draws and how to use it.
5. [REFERENCES.md](REFERENCES.md): every algorithm and where it comes from.
6. [PERFORMANCE.md](PERFORMANCE.md): how fast it is and how it got there.
7. [FUTURE_IDEAS.md](FUTURE_IDEAS.md): what is planned, and what was
   built and taken out again.

## Checks

```bash
uv sync
uv run pytest                          # every test; at least 90% branch coverage
uv run pytest --update-references      # write the reference images again (review them!)
uv run ruff check --no-fix             # lint (without --no-fix it changes files)
cd core && cargo clippy                # lint the Rust core
```

`uv run` builds the Rust core again whenever its sources changed.
Rendered frames are compared with the reference PNGs in
`tests/references/`; on a mismatch, `*.actual.png` and `*.diff.png` are
written next to the reference.
