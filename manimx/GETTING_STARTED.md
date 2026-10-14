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
├── manimx.json                 the project: settings, scenes, timeline
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
from manimx import DOWN, PI, RIGHT, UP, Circle, FadeIn, FadeOut, Indicate, Scene, Square


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
| Style | `color=`, `fill_color=`, `fill_opacity=`, `stroke_color=`, `stroke_width=`, `stroke_opacity=`, `stroke_joint=`, `stroke_cap=`, or `set_fill`, `set_stroke`, `set_color`. Colors, opacities and widths take one value or a list: `fill_color=[RED, BLUE]` is a gradient, `stroke_width=[2, 8, 2]` tapers |
| Changes | `shift`, `move_to`, `rotate`, `scale_by`, `set_scale`, `set_alpha` |
| Animations | `x.animate.<changes>` (chain several), `FadeIn`, `FadeOut` (both take `shift=`), `Rotate(x, angle)`, `Indicate`, `GrowFromCenter`, `ShrinkToCenter` |
| Timing | `self.play(..., run_time=2, rate=..., lag_ratio=0.3)`, `self.wait(seconds)`, `self.wait_until(scene_time)` (e.g. the next section starts on a beat) |
| Rate functions | `from manimx.userland.rate_functions import linear, smooth, there_and_back, ease_in_out_cubic, ...` |
| Directions | `UP`, `DOWN`, `LEFT`, `RIGHT`, `IN`, `OUT`, `ORIGIN`, `UL`, `UR`, `DL`, `DR`, and `PI`, `TAU`, `DEGREES` |

Everything above is importable from `manimx`. Behind it are two packages:
`manimx.animation` is the core (scene nodes, signals, `Scene`, the
`Animation` contract), and `manimx.userland` is what is built on top of
it (the shapes, the animations, rate functions, constants, the glTF
loader). Your own shapes and animations belong to the same layer as
userland: they use exactly the same public API.

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
  from manimx import derive

  dot.position.y = derive(lambda t: 0.3 * math.sin(4 * t), self.time.scene_time)
  ```

## 5. Simulations

manimx has no built-in physics engine. A `Simulation` is your own step
function: it moves objects by `dt`, however you like. It runs once while the
scene is prepared and is stored as keyframes, so rendering never simulates
again, and every frame can be rendered independently and in parallel.

```python
import numpy as np
from manimx import RIGHT, UP, Dot, Rectangle, Scene, Simulation


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

### Timed by a voice

What is said is written in a script, `scripts/narration.txt` (in
`manimx ui`: **+ New** under Scripts, then the Script tab), before
anything is recorded. A line
`---` ends a part, what is recorded in one go; a line `[middle]` names a
section:

```text
Every flash, every circle, every square.
---
[middle]
Time has run out for the square.
```

Every word is an event named after the script: `narration.square` the
first "square", `narration.square.2` the second, `narration.middle.square`
the first after `[middle]`. Put a part on a script track and its words
are timed as if read out at once (each word as long as its letters); a scene puts a `cue` where a
word should land, and `start`/`stop` run animations from one word to
another while the scene goes on:

```python
self.cue("narration.circle")                     # here, "circle" is said
flight = self.start(MoveAlongPath(dot, path))    # flies while the scene goes on
self.wait(2)
self.cue("narration.circle.2")
self.stop(flight)                                # lands on the second "circle"
```

The clip's time bends smoothly so every cue lands on its word. Record a
part (in `manimx ui`, see [the preview page](#10-the-preview-page); or
use a recording: `uv run manimx take narration take.m4a --part 1`) and
Whisper times its words: the scene follows the real voice without
changes. Whisper comes with `manimx[voice]`, which projects made by
`manimx new` depend on (in an older project: `uv add 'manimx[voice]'`);
its model, downloaded on first use, is the setting `voice.model`. A
clip's `"cues": {"square": "narration.middle.square"}` in `manimx.json`
syncs a cue to another event. Renders put the parts' subtitles into the
video and an `.srt` file next to it. The example project's Voice scene
is reanimate's voice example remade this way.

## 8. The project file

`manimx.json` holds the project's settings in named sections. Your editor
completes and checks it through the schema, and manimx rejects typos with a
"did you mean" hint. You rarely need to edit it by hand: `manimx settings`
opens a menu with every setting (see below).

```json
{
  "$schema": "./.manimx/project.schema.json",
  "format": 3,
  "name": "my-video",
  "output": {
    "width": 1920, "height": 1080, "fps": 60,
    "background": "#0b1020", "background_opacity": 1.0,
    "antialiasing": 4, "samples": 1,
    "audio": true, "sample_rate": 48000,
    "directory": "renders"
  },
  "effects": {
    "motion_blur": { "enabled": false, "shutter_angle": 180.0 },
    "bloom": { "enabled": false, "intensity": 0.1, "threshold": 1.0, "knee": 0.5, "radius": 1.0, "levels": 6 },
    "shadows": { "enabled": true, "softness": 1.0, "samples": 16, "resolution": 2048, "method": "pcf" },
    "tone_map": { "operator": "pbr_neutral", "exposure": 1.0 }
  },
  "lighting": { "ambient_color": "#ffffff", "ambient_intensity": 0.15, "default_lights": true },
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
| `output` | how the video looks: resolution, frame rate, background, antialiasing, audio |
| `output.samples` | renders averaged per frame (1 by default, since every frame then costs `samples` renders), each with the camera moved by a fraction of a pixel and the shadow taps turned: smoother edges and soft shadows, like Blender Eevee's render samples |
| `effects.motion_blur` | spreads the `output.samples` renders over the exposure. `shutter_angle` is the exposure as a share of a frame: 360 is one frame, 180 half (the film default), 720 two frames, so the smear grows with it. Long exposures need more samples, or they show separate images |
| `effects.bloom` | glow around bright parts of lit 3D objects |
| `effects.shadows` | shadows of 3D objects: `softness` (0 is hard), `samples` (smoother edges), `resolution` (sharper detail), `method`: `pcf` (default, a fixed narrow soft edge) or `pcss` (edges soften with the distance to the caster, slower) |
| `effects.tone_map` | how lit 3D colors map to the screen (`pbr_neutral`, `aces`, `agx`, `reinhard`, `none`) and their `exposure`. 2D colors always stay exact |
| `lighting` | ambient light for 3D objects, and a default key light for scenes without lights |
| `performance` | speed only, never the image: render processes, frames queued on the GPU |
| `voice` | timing recorded takes: the Whisper `model` (`small`; `large-v3-turbo` hears better) and the `language` (empty: Whisper finds it) |
| `timeline.clips` | clips in order. Without `start` a clip begins when the previous one ends. `layer` draws on top; `cut` keeps only that part of the scene, in seconds; `speed` plays it faster (2) or in slow motion (0.5) |
| `timeline.events` | times of the events scenes `wait_for` |

The camera is not a project setting: every scene has its own `self.camera`.

### The settings menu

```sh
uv run manimx settings          # the menu
uv run manimx settings --print  # every setting with its value and docs, as text
```

```
  ▸ Output                                Video size, frame rate, background and files.
  ▾ Effects                               Motion blur, bloom, shadows and tone mapping.
      ▾ Motion blur          [on]
            samples          8            one image every 0.00417 s of the exposure
›           shutter angle    720°         0.03333 s exposure at 60 fps
      ▸ Bloom                [off]
      ▸ Shadows              [on]
      ▸ Tone map                          Mapping of lit 3D colors to the screen.
  ▸ Lighting                              Ambient and default light for 3D objects.
  ▸ Performance                           Processes and GPU queueing.

  Exposure as a share of a frame: 360 is one frame, 180 half (film's default), 720 two frames.

   enter  edit    r  default: 180°    ←  close section
   ↑↓  move    s  save and quit    q  quit without saving
```

Below the tree: the description of the selected setting, then the keys
that work on it right now (while typing a value, what it accepts). In
short: ↑/↓ move, →/← open and close sections, space switches an effect
(or a yes/no value) on and off or steps through choices, enter edits a
value, r puts it back to its default, s saves to `manimx.json`, q quits.

## 9. Render

```sh
uv run manimx render --preview          # 960×540, 30 fps, one render per frame: fast
uv run manimx render                    # the full timeline with the project's settings
uv run manimx render Bounce             # one scene
uv run manimx render -w 4               # four render processes (more CPU cores)
uv run manimx render --png -o frames    # a PNG sequence (plus audio.wav)
uv run manimx render --start 2 --end 5  # a part of the timeline
uv run manimx still -t 2.5              # one frame as a PNG
uv run manimx render --window           # watch it render instead of writing a file
uv run manimx render --window --write   # both
uv run manimx render --open             # open the video when done
uv run manimx render --samples 16       # renders averaged per frame
uv run manimx render --motion-blur --shutter-angle 720
uv run manimx render --stems            # the sound for editing elsewhere (below)
```

`--stems` renders the video without sound and writes next to it a WAV
per track (`renders/<name>-stems/`) and two sessions with the video and
all of them in place: `<name>.rpp` for Reaper and `<name>.otio`
(OpenTimelineIO) for Kdenlive, DaVinci Resolve and others. Clean,
compress and master the sound there. Each WAV has its clips' volumes
and fades; the tracks' volumes, mutes and solos are the session's
faders and buttons.

Options on the command line override `manimx.json` for that run. Rendering
keeps several frames queued on the GPU, and with `-w` it spreads over
processes. Each process prepares the scenes itself, which costs about a
second at startup and pays off for longer videos.

Videos are H.264, encoded on the GPU's video encoder when one works
(NVIDIA NVENC, AMD AMF, Intel Quick Sync, VA-API on Linux, VideoToolbox
on macOS), else with x264 on the CPU; the render's header names it.
`output.encoder` (or `--encoder`) picks one: `auto`, `nvenc`, `amf`,
`qsv`, `vaapi`, `videotoolbox` or `x264`.

## 10. The preview page

```sh
uv run manimx ui                        # opens the preview page; Ctrl+C stops it
uv run manimx ui --port 8000 --no-open  # a fixed address, without opening a browser
```

The page is a video editor for the project. Everything you do on it is
saved to `manimx.json` (and `scripts/`) a moment later, and shows at
once; Ctrl+Z undoes any of it. **?** (or the ? key) lists every key and
mouse action.

- **Left, the bin:** the project's scenes, scripts and sounds. Drag one
  onto the timeline to use it; **+ Import** copies sound files in.
- **Middle, the frame** at the playhead, with the playback buttons
  (Space plays). The **Script** tab beside **Preview** is where scripts
  are written.
- **Right, the inspector:** what is selected, to type exact values.
- **Below, the timeline:** the ruler in minutes and seconds (drag on it
  to scrub, Ctrl+wheel zooms), and three kinds of tracks:
  - **Video** tracks hold scenes; higher tracks are drawn on top. A
    clip shows thumbnails, its cues as small diamonds (hover to name
    them), and its arrow opens the scene's nodes and animations. A
    scene that plays sounds gets a *sound* row under its track, with a
    volume slider.
  - **Script** tracks hold parts of scripts, each word a block.
  - **Audio** tracks hold music, effects and free recordings.

Every track's header names it (double-click to rename) and has its
switches: hide a video track, mute (M) or solo (S) a sound track, lock
any track against slips, and a volume slider. Drag a clip to move it,
its edges to cut it, the corners of a sound to fade it in or out; S
splits at the playhead.

### Narrating

1. Write a script in the Script tab: what is said, a `---` line between
   parts recorded in one go.
2. Drag its parts from the bin onto a script track (or **Place at the
   playhead**). Until recorded, they are timed as if read out; **Fit to
   scenes** moves the sentences so the words the scenes' cues wait for
   are said there.
3. Select a part and press ● (or R). The microphone opens, playback
   starts a few seconds before the part with 3, 2, 1 over the frame,
   and the **Teleprompter** shows each sentence a moment before it is
   due. Press ● again to stop, Escape to throw the take away. Wear
   headphones, so the timeline's sound stays out of the microphone.
4. Whisper times the take's words, and the scenes synced to them follow
   your voice. A word dragged on the track moves it (Shift: its
   sentence).

The teleprompter (T) and the subtitles (C) can also be shown while just
playing. The settings `voice.countdown` and `voice.lead` set how long
the countdown is and how early sentences show.

### Rendering from the page

**Render** saves `manimx.json` and runs `manimx render` in a process of
its own (so editing goes on meanwhile), for the whole timeline or a
range, shows its progress, can cancel it, and plays the finished video.
**Sound for editing** is `--stems`. **Settings** changes everything
`manimx settings` changes, grouped as Video, Look, Sound, Voice and
Advanced.

The preview keeps the frames it fetched and fetches the ones after the
playhead ahead (the "Ahead" box; a yellow line on the ruler marks the
frames kept); the menu beside it renders the preview at 1/2, 1/4 or 1/8
of the size. Frames are rendered once each (no motion blur or render
samples). Moving sounds or changing their volume keeps the frames;
only edits that move a scene fetch them again. Saving a scene file
updates the preview, preparing only that file's scenes again; an error
shows above the frame until the next save fixes it (`--no-reload` turns
this off). The page's address keeps the time (`#t=2.5`).

## 11. When something goes wrong

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

Presenting (firing events with clicks) and editing scenes in the UI
(keyframes, gizmos) come next. They build on the same `manimx.json`:
what you do in the UI and what you write in code end up in the same
project file.
