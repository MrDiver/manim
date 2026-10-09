# Getting started with manimx

manimx is a GPU (WebGPU) animation engine in the spirit of Manim. You write
scenes in Python, and the `manimx` command line puts them into a project,
plays them one after another on a timeline, and renders them to video, with
sound.

This guide walks through the whole journey: create a project, add scenes,
write animations, add physics-like simulations and sound, and render.
A finished example lives in [`example-project/`](example-project).

## 1. Install

You need [uv](https://docs.astral.sh/uv/), a GPU with Vulkan, Metal or
DirectX 12 support, and `ffmpeg` on your `PATH` (for video files).

manimx is not published yet, so the CLI runs from this checkout:

```sh
cd manimx
uv sync
uv run manimx --help
```

## 2. Create a project

```sh
uv run manimx new my-video
```

Anything you leave out is asked for interactively: example scenes or an
empty project, the resolution and the background color. In scripts, pass
the options instead (`manimx new my-video --example --resolution 1080p`)
or use `-y` to take the defaults.

```
my-video/
├── manimx.json                 the project: output settings, scenes, timeline
├── .manimx/project.schema.json lets your editor complete and check manimx.json
├── pyproject.toml              depends on manimx (this checkout)
├── scenes/
│   ├── intro.py                Intro, an animation scene
│   └── bounce.py               Bounce, a simulation scene
├── assets/                     your sounds and other files
└── renders/                    output
```

Then install the project's environment and render a quick preview:

```sh
cd my-video
uv sync
uv run manimx render --preview
```

The video lands in `renders/my-video.mp4`. From here on, every command runs
inside the project folder, or any folder below it.

## 3. Add scenes

```sh
uv run manimx generate scene title-card            # or: manimx g scene title-card
uv run manimx g scene falling-leaves --kind simulation
```

```
CREATE scenes/title_card.py  TitleCard (animation)
UPDATE manimx.json  registered TitleCard, appended to the timeline
```

`generate` writes the scene module from a template, registers the scene in
`manimx.json`, and appends it to the timeline. Names are normalized:
`title-card`, `TitleCard` and `"title card"` all give the class `TitleCard`
in `scenes/title_card.py`.

```sh
uv run manimx list                       # scenes, and when each clip plays
uv run manimx remove scene TitleCard     # unregister; asks whether to delete the module
```

## 4. Write an animation

A scene describes what happens in `construct()`:

```python
from manimx.animation import DOWN, PI, RIGHT, UP, Circle, FadeIn, FadeOut, Indicate, Scene, Square


class TitleCard(Scene):
    def construct(self) -> None:
        square = Square(2.0, color="#5ea3d0", fill_opacity=0.8)
        circle = Circle(1.0, color="#e07a5f", fill_opacity=0.8).shift(DOWN * 2)

        self.play(FadeIn(square, shift=UP * 0.5))                 # 1 s by default
        self.play(square.animate.rotate(PI / 4).shift(RIGHT), FadeIn(circle), run_time=1.5)
        self.play(Indicate(circle))
        self.wait(0.5)
        self.play(FadeOut(square), FadeOut(circle))
```

The building blocks:

| | |
|---|---|
| Shapes | `Circle`, `Dot`, `Square`, `Rectangle`, `Triangle`, `RegularPolygon`, `Polyline`, `Shape2D` (any Bezier path) |
| Containers | `Group(a, b, ...)` moves and fades its members together; `ClippingGroup(mask, content)` shows content only inside the mask |
| Style | `color=`, `fill_color=`, `fill_opacity=`, `stroke_color=`, `stroke_width=`, `stroke_opacity=`, or `set_fill`, `set_stroke`, `set_color` |
| Changes | `shift`, `move_to`, `rotate`, `scale_by`, `set_scale`, `set_alpha` |
| Animations | `x.animate.<changes>` (chain several), `FadeIn`, `FadeOut` (both take `shift=`), `Rotate(x, angle)`, `Indicate`, `GrowFromCenter`, `ShrinkToCenter` |
| Timing | `self.play(..., run_time=2, rate=..., lag_ratio=0.3)`, `self.wait(seconds)` |
| Rate functions | `from manimx.animation.rate_functions import linear, smooth, there_and_back, ease_in_out_cubic, ...` |
| Directions | `UP`, `DOWN`, `LEFT`, `RIGHT`, `IN`, `OUT`, `ORIGIN`, `UL`, `UR`, `DL`, `DR`, and `PI`, `TAU`, `DEGREES` |

Things that work differently from Manim:

- **Animations reference their objects.** There are no target copies.
  `square.animate.shift(RIGHT)` animates the square's own position, and
  several animations of different properties (or different axes) run at the
  same time without interfering.
- **Changes between plays happen at that moment.** `square.shift(RIGHT)`
  outside of `play` moves the square at that point of the scene, not
  retroactively from the start.
- **`play` adds objects automatically**, when their animation starts.
- **No updaters: derived values instead.** Keep one object at another's
  position with `label.follow(ball, UP)`. Or derive any value, here from
  the scene's time:

  ```python
  import math
  from manimx.animation import derive

  dot.position.y = derive(lambda t: 0.3 * math.sin(4 * t), self.time.scene_time)
  ```

## 5. Simulations

manimx has no built-in physics engine. A `Simulation` is your own step
function: it moves objects by `dt`, however you like. It runs once while the
scene is prepared and is stored as keyframes, so rendering never simulates
again, and every frame can be rendered independently and in parallel.

```python
import numpy as np
from manimx.animation import RIGHT, UP, Dot, Rectangle, Scene, Simulation


class Drop(Scene):
    def construct(self) -> None:
        wall = Rectangle(0.2, 3, fill_opacity=1).shift(RIGHT * -5)
        ball = Dot(0.25).shift(UP * 2)
        velocity = np.array([1.0, 0.0, 0.0])

        def step(dt: float) -> None:
            velocity[1] -= 9.81 * dt
            ball.shift(velocity * dt)
            if ball.position.y.value < -3 and velocity[1] < 0:   # bounce on the floor
                velocity[1] *= -0.8
                simulation.emit("bounce")                          # an event at this moment

        simulation = Simulation(step, dt=1 / 480, duration=4)    # or: until=lambda: ...
        self.play(simulation, wall.animate.shift(RIGHT * 2))       # the wall is an input
```

- Animations in the same `play` call are inputs: the step function sees the
  wall wherever its animation has put it.
- Values derived from simulated objects follow every keyframe.
- After `play`, the scene continues from the simulation's final state.

## 6. Sound

```python
sound = self.play_sound("assets/whoosh.wav", volume=0.5)  # WAV directly; MP3, FLAC, OGG through ffmpeg
self.play_wave(samples, sample_rate=48000)                # a numpy array: (count,) mono or (count, 2) stereo
self.wait(sound.duration)                                 # playing does not wait by itself
```

Inside a simulation, `simulation.play_wave(...)` starts the sound at the
simulated moment. That's how the example's balls make a plop on every
bounce, panned to where they land, with a reverb. All of it is a few lines
of numpy in
[`example-project/scenes/bounce.py`](example-project/scenes/bounce.py).

Sounds keep their pitch: slowing a scene down moves its sounds apart, but
never stretches them. Rendered videos include the mix automatically, and
`--no-audio` leaves it out.

## 7. Events: waiting for a moment

A scene can hold until an event happens, and can announce events:

```python
self.play(FadeIn(logo))
self.wait_for("click")     # hold here until "click"
self.play(FadeOut(logo))
self.emit("logo-gone")     # other scenes can wait_for this one
```

When rendering a video, the event's time comes from `manimx.json`:

```json
"timeline": {
  "clips": [ { "scene": "Intro" }, { "scene": "Bounce" } ],
  "events": { "click": 3.0 }
}
```

Later, the presentation mode (web UI) will fire such events on a presenter's
click.

## 8. The project file

`manimx.json` holds the project's settings in named sections. Your editor
completes and checks it through the schema, and manimx rejects typos with a
"did you mean" hint.

```json
{
  "$schema": "./.manimx/project.schema.json",
  "format": 1,
  "name": "my-video",
  "output": {
    "width": 1920, "height": 1080, "fps": 60,
    "background": "#0b1020",
    "motion_blur": 8,
    "audio": true, "sample_rate": 48000,
    "directory": "renders"
  },
  "camera": { "frame_height": 8.0 },
  "performance": { "workers": 1, "frames_in_flight": 3 },
  "scenes": {
    "Intro": { "module": "scenes.intro", "class": "Intro" }
  },
  "timeline": {
    "clips": [
      { "scene": "Intro" },
      { "scene": "Bounce", "start": 2.5, "layer": 1, "cut": [0.5, 4.0] }
    ],
    "events": {}
  }
}
```

| Setting | Meaning |
|---|---|
| `output` | how the video looks: resolution, frame rate, background, motion blur (sub-frames per frame; 1 is off), audio |
| `camera.frame_height` | world units visible vertically (Manim's default: 8) |
| `performance` | speed only, never the image: render processes, frames queued on the GPU |
| `timeline.clips` | clips in order. Without `start` a clip begins when the previous one ends. `layer` draws on top; `cut` keeps only that part of the scene, in seconds |
| `timeline.events` | times of the events scenes `wait_for` |

## 9. Render

```sh
uv run manimx render --preview          # 960×540, 30 fps, no motion blur: fast
uv run manimx render                    # the full timeline with the project's settings
uv run manimx render Bounce             # one scene
uv run manimx render -w 4               # four render processes (more CPU cores)
uv run manimx render --png -o frames    # a PNG sequence (plus audio.wav)
uv run manimx render --start 2 --end 5  # a part of the timeline
uv run manimx still -t 2.5              # one frame as a PNG
```

Options on the command line override `manimx.json` for that run. Rendering
keeps several frames queued on the GPU, and with `-w` it spreads over
processes. Each process prepares the scenes itself, which costs about a
second at startup and pays off for longer videos.

## 10. When something goes wrong

Errors name the problem and how to fix it:

```
╭─ scene error ──────────────────────────────╮
│ ✖ Intro failed while it was being prepared │
╰────────────────────────────────────────────╯
╭──────────── Traceback (most recent call last) ────────────╮
│ scenes/intro.py:18 in construct                           │
│ ❱ 18 │   │   self.play(square.animate.shfit(UP))          │
╰───────────────────────────────────────────────────────────╯
AttributeError: Square.shfit cannot be animated
```

- A broken scene never breaks the others: `manimx render OtherScene` still
  works.
- `--traceback` shows manimx's own frames too.
- `manimx list` is a quick check that every scene still imports and prepares.

## What's next

The web UI (scrubbing, editing the timeline, presenting with clicks) builds
on the same `manimx.json`: what you do in the UI and what you write in code
end up in the same project file.
