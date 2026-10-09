# Rendering performance

Measurements of the renderer and animation engine, one row per optimization
step, so every change is justified by numbers and regressions show up.

## How we measure

```sh
uv run python scripts/benchmark_project.py example-project --check-reference /tmp/ref.npz
```

- **Workload:** the example project (Intro: the Manim logo, Bounce: five
  simulated balls, Outro: a template animation): 1114 frames, 1920×1080, 60 fps,
  motion blur with 8 sub-frames, all rendered in one process with 3 frames in
  flight. No encoding: frames go nowhere, because H.264 encoding runs in
  parallel in ffmpeg and was measured to add nothing.
- **Stage times** come from wrapping the stage functions with timers (no
  profiler, which roughly doubles Python's cost). Stages nest:
  `render_layers` contains the vector batch, the mesh batch and the shadow
  views.
- **GPU times** come from timestamp queries on a sample of frames, rendered one
  at a time.
- **Correctness:** `--check-reference` renders 48 frames across the timeline
  and compares them with frames saved before the first optimization. Every step
  below is bit-identical.
- **Machine:** Intel Core Ultra 9 285H (16 threads), NVIDIA GeForce RTX 5070
  Laptop GPU (Vulkan), Linux, power profile `performance`. Numbers are
  comparable within this table only. Run-to-run noise is about ±0.5 fps.
- **Power:** from step E on, the power state is recorded: the USB-C adapter
  was connected but the battery was discharging (about 5 W at idle), so the
  adapter delivered less than the laptop drew. Earlier steps were measured in
  the same session, but their power state was not checked. Absolute numbers on
  the full-power charger may be higher.

## Results

| Step | fps (blur 8) | ms/frame | fps (no blur) | state_at | vector batch | render_layers | GPU ms/frame |
|---|---|---|---|---|---|---|---|
| 0. Baseline | 74.5 | 13.43 | 407.6 | 2.12 | 3.91 | 9.24 | 3.68 |
| A. Cached stroke records | 87.5 | 11.43 | 452.2 | 2.09 | 1.97 | 7.29 | 4.06* |
| B. No 3D stages without meshes | 119.6 | 8.36 | 556.3 | 2.02 | 1.88 | 4.32 | 1.72 |
| C. Renderer as a service | 134.9 | 7.41 | 603.9 | 1.99 | 1.87 | 3.51 | 1.75 |
| D. Snapshot fast paths | 149.2 | 6.70 | 644.9 | 1.20 | 1.88 | 3.55 | 1.72 |
| E. Retained vector scene | 169.8 | 5.89 | 674.1 | 1.18 | 1.18 | 2.71 | 1.74 |

\* GPU time is sampled on 60 frames with a full sync per frame and varies by
about ±10% between runs. Step A does not change any GPU work.

## Baseline analysis

What a frame cost before any optimization (13.4 ms):

- **CPU-bound:** the GPU needs 3.7 ms per frame, so it idled about 75% of
  the time.
- **Motion blur multiplies everything:** without it a frame takes 2.5 ms.
  Each of the 8 sub-frames evaluates the scene and runs the renderer's whole
  CPU side again.
- **Within a sub-frame, from the largest cost down:**
  - **Vector batch (29%):** every shape's stroke records (curve links,
    joints, per-point width and color) were rebuilt in Python loops for
    every sub-frame, even for shapes that only move.
  - **WebGPU call overhead (about 28%):** views, render pass descriptors
    and structs were recreated for every sub-frame, about 330k binding
    calls for 557 frames.
  - **`state_at` (16%):** evaluating signals and building render objects,
    including re-parsing colors that are already floats.
  - **3D stages with no 3D content (about 12% CPU, 1.9 ms GPU):** mesh
    batch, default lights, shadow views, and the geometry and lighting
    passes ran although the project has no meshes.
- **Everything else is small:** preparing all scenes takes 0.6 s once,
  mixing the audio 5 ms, and importing manimx 0.34 s.

## Steps

### A. Cached stroke records

Stroke records depend on a shape's points and stroke style, never on its
transform, but were rebuilt for every sub-frame. Now:

- **Topology is cached** per shape, keyed by points version, joints and cap:
  curve records linked inside each subpath, and the pairs of anchors shared
  by consecutive curves.
- **Per-point records are cached** (position, width, color with shared anchors
  matched), keyed by points and stroke version. A fade, which changes the
  stroke's opacity, rebuilds only these, vectorized. The old per-curve loop
  that averaged shared anchors is a single numpy operation, and gives the same
  float32 results, because every anchor is in at most one pair.
- **Each shape's local center** (for sorting transparent draws) is cached by
  points version.
- **Curve instances** are added as one array per shape instead of a Python
  loop per curve.

Moving shapes now only offset the cached records by their position in the
frame's buffers.

### B. No 3D stages without meshes

Lights, shadows and the deferred passes (geometry, lighting) only affect
meshes (RShape3D). The example project has none, yet every sub-frame built
a mesh batch with the default key light, built shadow views for 2D shapes
that cast shadows onto nothing, and ran a full-screen lighting pass over
an empty G-buffer: 1.6 ms of CPU and 1.9 ms of GPU per frame.

Now a frame without meshes skips all of that. Two things still have to be
right:

- **The G-buffer's depth must be empty.** Vector shapes are depth-tested
  against the meshes' depth, so stale depth from an earlier frame with
  meshes would hide them.
- **The lit layer must be empty**, because the composite reads it.

The renderer therefore clears both once, with load-op clears and no draws,
and remembers it for the current targets. Following mesh-free frames skip
those passes completely; a frame with meshes, or new targets (a size
change), resets the state.
`test_frames_without_meshes_after_meshes_start_from_empty_3d_layers`
renders flat, then flat with a sphere, then flat again, and checks that the
last frame equals a fresh renderer's. Breaking the clear makes it fail.

Result: +37% (87.5 → 119.6 fps). GPU time drops from about 3.7 to 1.7 ms per
frame, and `render_layers` from 7.3 to 4.3 ms.

### C. The renderer as a service

The renderer is meant to be a long-lived service: built once per process,
keeping everything it creates across all calls. Within one render call it
already lived for all frames, but every call (each driver `render()`,
`manimx still`, each worker process) created its own renderer with its own
GPU device. Even within a call, these were recreated for every sub-frame:

- **render target views and attachments** for the lit, vector and mask
  scratch targets and the vector depth;
- **in the composer:** a bind group and texture views for every fullscreen
  pass (composite, finalize, bloom, shader effects);
- **command encoders and submissions:** two per sub-frame.

Now:

- **`WGPURenderer.shared()`** is the process's renderer service: one device,
  created on first use. Drivers render through it, so the CLI, tests and
  later the web UI keep shaders, pipelines, buffers, targets and per-object
  data across calls. Rendering stays single-threaded per renderer.
- **`renderer.composer(effects)`** keeps one composer per effect chain, with
  its pipelines and texture pool.
- **`ResourceCache`**, owned by the renderer, keeps views per texture (weakly,
  so they go with their texture) and bind groups keyed by their layout and
  the identity of everything they bind (LRU-bounded, 512 entries).
  Fullscreen pipelines keep their bind group layout, because each
  `get_bind_group_layout()` call returns a new object, which would defeat
  the cache.
- **Render targets** build their attachments once.
- **One command encoder and one submission per sub-frame:** the renderer
  records into the composer's encoder. All buffer writes of a sub-frame land
  before its submission, and the previous sub-frame is already submitted, so
  sharing the same buffers stays correct.
- **The per-object cache is bounded:** a long-lived cache would otherwise
  keep data for objects that never come back (a rebuilt timeline has new
  uids). Every render is a generation; every 1024 renders, entries unused
  for 1024 renders are dropped.

New tests cover the shared instance and composers per effect chain, reuse
and LRU eviction in the resource cache, renders after size changes matching
a fresh renderer, and cache sweeping. The test suite runs faster too (11.2 →
9.6 s), since driver and CLI tests now share one renderer.

Result: +13% (119.6 → 134.9 fps). The composite drops from 0.72 to 0.37 ms
per frame, and `render_layers` from 4.3 to 3.5 ms.

### D. Snapshot fast paths

Snapshots rebuild every render object for every sub-frame, and the
render objects validate their inputs as if they came from users. Three of
those validations were most of `state_at`:

- **Colors:** each `Fill` and `Stroke` sent the color signal's RGB float
  triple through `ManimColor.parse`. For a float array of shape (3,) the
  parser returns the values unchanged (checked on 20,000 random colors,
  including values outside [0, 1]), so that case now skips it. Integer
  arrays (0–255), lists and strings still go through the parser.
- **Scalars:** opacity and width were validated with two numpy reductions
  each, 21% of `state_at` after the color fix. Plain Python numbers are
  now range-checked in Python, with the same error and NaN behaviour (only
  values below or above the range are rejected). Bools and sequences take
  the general path.
- **Points:** every snapshot copied the path's points. A float64 array of
  the right shape that owns its data and is read-only (what path signals
  hold) is now shared, since nobody can change it.

A new test pins down that the fast paths match the general parsers and keep
their errors.

Result: +11% (134.9 → 149.2 fps). `state_at` drops from 1.99 to 1.20 ms
per frame (−40%).

### Tried and dropped

- **Reusing layers for identical sub-frames.** The rendered layers were
  reused when a sub-frame's render tree, compared by fingerprint, equaled the
  previous sub-frame's. Result: 148.6 against 149.2 fps, no gain. It was also
  the wrong idea: a sub-frame is just another time, and it only helped when
  nothing changed at all. The waste is rebuilding and re-uploading every
  object although usually only a few transforms change, which E addresses.
- **A snapshot cache on the animation side.** The evaluator remembered each
  node's last render object (by version tokens), so `state_at` returned
  the same instances, or copies sharing all content, when only the transform
  changed. Alone it measured 148.9 fps (no change). Together with E it
  measured 171.3 fps against 169.8 without it, about 1%, near the noise, for
  110 lines of code. The renderer patches by uid, versions and world matrix,
  not by instance identity, so it does not need the cache. Dropped.

### E. The retained vector scene

The renderer kept derived data per object (tessellations, stroke records), but
every render still packed all objects' records into fresh arrays and uploaded
all of them. `RetainedVectors` (retained.py) keeps the packed records in
persistent arrays (*arenas*) mirroring the GPU buffers, with one slot per shape
(its uid and whether it is drawn as a mask). Each render diffs the flat scene
against the slots, using the identity and version tokens of the render
contract:

| Change | Work |
|---|---|
| none | nothing written |
| world matrix, layer or clip | the shape's 80-byte object entry |
| fill colors or stroke style (versions) | its color or point records, in place |
| new points | its records, in place if the size fits, else in a new range; curves and instances that point at them are rewritten |
| shape left the scene | its ranges become garbage, its object slot is reused |

Only the written ranges are uploaded (`VectorBatch.dirty`,
`GrowableBuffer.write`). The draw lists are still rebuilt every frame: they
are small, and transparent draws must be re-sorted by camera distance anyway.
When garbage outweighs the live records, everything is repacked from scratch.
The records are the same as before, only at other positions in the buffers, so
the pixels are identical.

Tests: a sequence of frames covering every kind of patch (move, recolor,
opacity, new points with a different count, a new stroke width, removal, an
arrival that reuses a slot, a clipping mask, reordering), each compared with a
fresh renderer; moving one shape rewrites exactly one object entry and nothing
else; an unchanged frame writes nothing; garbage is repacked.

Result: +14% (149.2 → 169.8 fps). The vector batch drops from 1.88 to
1.18 ms per frame, and `render_layers` from 3.55 to 2.71 ms. What remains per
shape and render is Python bookkeeping: the matrix comparison, the camera
distance and the draw record.

### F. Reused fill and stroke values

`Shape2D` now carries every field of `Fill` and `Stroke` as a signal:
lists of colors, opacities and widths (`COLORS`, `SCALARS`), joints and
caps. Building `Fill` and `Stroke` from those values in every snapshot
made `state_at` 7% slower (1.28 → 1.37 ms per frame; a `Shape2D` snapshot
took 10.4 µs).

`Fill` and `Stroke` are immutable values of the signals in their version
token. A `Shape2D` keeps the last snapshot's token with each value and
reuses the value while the token holds. Only a change of a style signal or
an `alpha` above builds a new one; moving shapes never do.

Measured in one session with the battery discharging on an underpowered
adapter (so absolute numbers are below the table's; stages this step does
not touch stay put: `render_layers` 2.83 → 2.86 ms, readback 0.82 →
0.84 ms):

| | fps (blur 8) | state_at ms/frame | Shape2D snapshot |
|---|---|---|---|
| all fields, values rebuilt every snapshot | 157.2 | 1.37 | 10.4 µs |
| values reused while their token holds | 162.0 | 1.13 | 6.1 µs |

Result: `state_at` −18%, also below the 1.28 ms before the new fields.
Frames are bit-identical. `Curve3D` builds its stroke every snapshot: the
benchmark has no `Curve3D`, so a reuse there would be unmeasured.

### G. Shadow bounds once per scope

`build_shadow_views()` called `scene_bounds()` for every light: a Python
loop over every drawn object, transforming its cached box. All lights in
one clip scope get the same bounds, and lights without shadows did not
need them at all. Now they are computed once per scope, and only for
lights that cast shadows.

Measured on `light-show` (nine shadow-casting spots, about 300 meshes in
the fireworks), 60 renders between 38 and 44 s, no motion blur:

| | ms/render | shadow views |
|---|---|---|
| bounds per light | 46 | 12.6 ms |
| bounds per scope | 36 | 2.1 ms |

`example-project` (one light): 245.5 → 255.2 fps without motion blur,
frames bit-identical.

### H. Simulation keyframes: one lookup per progress

A baked `Simulation` answered every signal on its own: a `bisect` in the
keyframe times and an interpolation, for each of its signals in every
evaluation. The light-show fireworks record about 6800 signals (every
spark's position, rotation, scale and glow), most of them unchanged at
any moment (sparks not yet born or already gone).

All signals of one simulation are asked for the same progress in turn,
so the keyframe index and fraction are now kept for the last progress,
and a signal whose two keyframes hold the same value object (it did not
change between them) returns it without interpolating.

| light-show, 38–44 s, no motion blur | ms/render | frame_at |
|---|---|---|
| lookup per signal | 36 | 19.7 ms |
| one lookup per progress | 32 | 14.6 ms |

`example-project` (one small simulation) is unchanged within noise
(alternating runs: 246.9 / 257.1 fps before, 257.2 / 255.7 after), and
its frames are bit-identical.

## Where the time goes now

5.89 ms per frame against 13.43 at the start (2.3× faster): `render_layers`
2.71 ms (1.18 of it the retained vector batch), `state_at` 1.18 ms, readback
0.69 ms, compositing 0.37 ms. The GPU needs 1.74 ms and runs in parallel, so
rendering is still CPU-bound.
