# light-show

An introduction to manimx in 45 seconds: a camera flies through a club
while moving heads play a light show to a synthesized 120 BPM track, and
the ManimX logo arrives on the beat with fireworks as the finale.

```sh
uv sync
uv run manimx render --preview    # 960×540, 30 fps, no motion blur (about 30 s)
uv run manimx render              # 1080p60, motion blur ×8, bloom (about 15 min)
uv run manimx still Finale -t 9.4 # one frame
```

## The timeline

| clip | seconds | what happens |
|---|---|---|
| `LightShow` | 0–32 | the camera flies in through glowing arches, circles the sculpture of moving shapes, climbs above the rig and ends facing the stage; the lights go intro → build → drop → ballyhoo → hit → blackout |
| `Finale` | 32–45 | the logo's parts slam in on four hits, each with its own spot; the X grows in glowing on the fifth; fireworks burst behind the stage |

## What it shows

- **glTF models with animated parts** (`assets/*.glb`): a moving head, a
  speaker and a mirror ball, loaded with `load_gltf`. A scene finds their
  parts by node name (`find("yoke")`, `find("head")`) and material tag
  (`find_all(tags="material:Lens")`), and pans, tilts and pumps them.
- **SpotLights inside a model**: each moving head's `SpotLight` is a child
  of its `head` node, so panning the yoke and tilting the head aim the
  light. `MovingHead.pan_tilt()` (`scenes/rig.py`) works out pan and tilt
  for any world point. The spots have no visible cone in the air yet; you
  see them where they land.
- **Animations on the beat grid**: `scenes/lights.py` has what a lighting
  console plays (`Aim`, `Flash`, `Pulse`, `Strobe`, `Chase`, `Fade`,
  `Recolor`, `Rainbow`). Each one is a function of its progress only. The
  scenes start every section with `wait_until(bar)` and play them for the
  section's length, so 32 strobe flashes over four bars are eighth notes.
- **Derived signals instead of updaters**: the camera path, the spinning
  mirror ball, the sculpture, the arches, the floor grid and the speaker
  cones are functions of `scene_time`. Each lens glows with its light's
  color and intensity.
- **A Simulation**: the fireworks in `scenes/finale.py` are a step function
  with gravity, drag and fading sparks. It is baked once when the scene is
  prepared, then rendered like any animation. Bangs and whooshes are played
  at the simulated times.
- **Sound**: `scenes/music.py` synthesizes the track with numpy. Its beat
  grid (`kick_envelope`) also drives the lights.
- **Effects**: bloom on everything emissive, motion blur, shadows from nine
  spots (`manimx.json`).

## Files

- `scenes/light_show.py`, `scenes/finale.py`: the two scenes
- `scenes/rig.py`: the stage, the moving heads, `fly`, the ManimX logo
  (not a scene)
- `scenes/lights.py`: the light animations and their paths
- `scenes/music.py`: the instruments, the arrangement, the beat grid
- `tools/build_assets.py`: generates `assets/*.glb` from manimx's geometry
  helpers (`uv run python tools/build_assets.py`)
