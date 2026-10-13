# Rendering performance

Measurements of the renderer and animation engine, one row per optimization
step, so every change is justified by numbers and regressions show up.

This is a log: older entries name the Python renderer (`RenderService`,
`SceneFlattener`, `FramePacker`, `WGPURenderer`, `FrameComposer`), which
the core's renderer replaced and which is removed now (FUTURE_IDEAS.md).

## How we measure

```sh
uv run python scripts/benchmark_project.py tests/projects/benchmark --check-reference /tmp/ref.npz
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

### I. Mesh batch: cached material entries, matrices in one go

`MeshBatch.build()` converted every material's colors (`ManimColor.parse`
and sRGB to linear) and inverted every normal matrix one mesh at a time,
in every render. Snapshots build a new `PbrMaterial` each time, so
nothing was reused. The material entry is now cached per shape and
material version (equal versions mean equal values), and the model and
normal matrices of all meshes are built with one batched inverse.

| light-show, no motion blur | ms/render | mesh batch |
|---|---|---|
| show (2–30 s), before | 13 | 4.8 ms |
| show (2–30 s), after | 12 | 2.0 ms |
| fireworks (38–44 s), before | 31 | 10.9 ms |
| fireworks (38–44 s), after | 28 | 7.1 ms |

`example-project` frames are bit-identical.

### J. Whole snapshots reused while nothing changed

Step F reused a `Shape2D`'s `Fill` and `Stroke`, but every snapshot
still built a new render node, and `Shape3D`, `Curve3D`, lights and
groups reused nothing. Now `Node.snapshot()` keeps the last render node
with a token of the versions of all the node's signals and of the alphas
above it, and returns the same object while the token holds (render
nodes are immutable). Groups and clipping groups also need every child
(mask, content) to have returned the same object. Leaf classes build
through `_build_snapshot()`.

| | fps (blur 8) | fps (no blur) | frame_at |
|---|---|---|---|
| benchmark project, before | 44.5 | 266.0 | 5.14 ms |
| benchmark project, after | 49.2 | 285.2 | 3.10 ms |
| light-show 2–30 s, before | | 97.9 | 3.9 ms |
| light-show 2–30 s, after | | 110.4 | 3.0 ms |

Frames are bit-identical. The light-show fireworks (38–44 s) are
unchanged within noise: every spark changes every frame, so there is
nothing to reuse and the token is extra work.

Tried and dropped in the same round: skipping the fold of a
`SignalTrack` before its first and after its last placement (returning
the base or the final value). `frame_at` did not change measurably
(3.9 / 14.3 ms before, 4.0 / 16.6 ms after on light-show), and two
reference frames changed, so it was reverted.

### K. Particles: one node for many copies of a mesh

The light-show fireworks were 832 spark and 12 rocket `Shape3D` nodes:
832 snapshots, 832 objects in `MeshBatch.build()`, and a simulation
recording about 6800 signals. `Particles` (a `Shape3D` with four array
signals: positions, rotations, scales, colors) holds them in two nodes;
the simulation records eight arrays, and the renderer draws each node
with one instanced draw (the mesh shader already reads its object by
instance index), building all its object and material rows with numpy.

Measured with the machine in low-power mode (so absolute numbers are
below the other rows), light-show 38–44 s, no motion blur, alternating
the old and the new `finale.py`:

| | ms/render | frame_at | mesh batch |
|---|---|---|---|
| 844 Shape3D nodes | 73 | 39.1 ms | 21.6 ms |
| 2 Particles nodes | 28 | 7.2 ms | 7.6 ms |

Preparing light-show (baking the fireworks) drops from 6.5 to 2.0 s.
The benchmark project has no particles; its frames are bit-identical,
and `MeshBatch` there measured 2.68 ms before and 1.98 ms after in the
same power state.

### L. The render chain (structure, no speedup)

The renderer is split into stages with one input and one output each:
`SceneFlattener` (tree to `FlatScene`), `FramePacker` (`FlatScene` to
`FramePacket`, owning every per-object cache), `WGPURenderer` (packet to
layers, GPU work only) and `FrameComposer`. `RenderService` holds them.
The benchmark now times the stages of the chain, with the packer's parts
indented.

Measured on the benchmark project (with `PortalOrbit`), power profile
`performance`, charging:

| | fps (blur 8) | fps (no blur) |
|---|---|---|
| before | 48.6 | 287.6 |
| after | 49.0 | 287.0 |

Frames are bit-identical. Where the time goes, per frame with motion
blur: pack 9.25 ms (shadow views 5.03, vector records 2.28, mesh batch
1.54), `frame_at` 3.09 ms, render (GPU encoding) 2.92 ms, flatten 1.74
ms. The GPU needs 8.15 ms. The next steps cache inside the packer and
the flattener.

### M. Shadow views cached, unchanged shadow maps skipped

`PortalOrbit` has six shadow-casting lights (three point lights with six
cube faces each), so every sub-frame built 21 shadow views with small
numpy operations and rendered 21 depth passes, eight times per frame.
The lights never move, and shadow maps do not depend on the camera, which
is what moves.

- **A light's views are cached** in `RNodeDataCache` (by light uid),
  keyed by its world matrix, its settings (kind, range, cone), the bounds
  of its scope and the config values a frame record holds. Only the
  frames' time is written per call.
- **`scene_bounds()` transforms all boxes in one `einsum`** instead of a
  matrix product per object.
- **`ShadowLayers` marks unchanged layers.** A layer's depth depends on
  its frame record and on its scope's shadow casters. Render nodes are
  immutable, so an object's identity, its version tokens and its world
  matrix stand for everything the depth pass reads. The packer holds the
  previous call's objects so their ids cannot be reused. `ShadowPass`
  skips unchanged layers, unless the maps were recreated or another
  packer fed the renderer.

| | fps (blur 8) | fps (no blur) | shadow views | render (GPU encoding) | GPU shadows |
|---|---|---|---|---|---|
| before | 49.0 | 287.0 | 5.03 ms | 2.92 ms | 1.06 ms |
| after | 67.1 | 357.8 | 0.92 ms | 2.22 ms | 0.22 ms |

+37%, frames bit-identical. Dropped in the same round: caching the
linear ambient color (`RenderConfig.ambient_linear` parses a color on
every render): 66.2 / 67.1 fps with it against 66.5 / 65.1 without, so
within noise.

### N. Transform matrices computed once

`SceneFlattener` called `Transform.matrix()` for every node in every
sub-frame: a quaternion to matrix conversion and a few small arrays,
although a `Transform` never changes and snapshots of unchanged nodes
share theirs (step J). The matrix is now computed on first use and kept
on the `Transform`, read-only.

| | fps (blur 8) | fps (no blur) | flatten |
|---|---|---|---|
| before | 67.1 | 357.8 | 1.73 ms |
| after (three runs) | 70.3 / 70.9 / 70.6 | 377.3 | 0.69 ms |

About +5%, frames bit-identical. Dropped in the same round: reusing a
node's world matrix object while the node and its parent's world were
unchanged (and comparing world matrices by identity in
`RetainedVectors`). On top of the matrix cache it measured 71.1 / 71.6
fps against 70.6 / 70.9 without, and flatten 0.64 against 0.70 ms:
about 1%, within noise, for a cache of every node's world matrix.

### O. Mesh vertices and indices retained

`MeshBatch.build()` concatenated every mesh's vertices and indices into
new arrays on every pack, and `MeshPass` uploaded them in full: in
`PortalOrbit` about 0.96 MB per sub-frame (13,674 vertices, 75,252
indices), 7.7 MB per frame with motion blur, although the meshes only
move. `RetainedMeshes` keeps them in arenas mirroring the GPU buffers
(`Arena`, now shared with `RetainedVectors` in `arena.py`), one range
per mesh uid, rewritten only for a new mesh version. The batch reports
the written ranges, and `MeshPass` uploads only those. Objects,
materials and draws are small and still rebuilt.

| | fps (blur 8) | fps (no blur) | mesh batch | render (GPU encoding) |
|---|---|---|---|---|
| before | 70.6 | 377.3 | 1.54 ms | 2.26 ms |
| after (two runs) | 74.4 / 73.9 | 379.9 / 390.7 | 1.16 ms | 2.00 ms |

+5%, frames bit-identical.

### Tried and dropped after step O

Each measured on the benchmark project, frames bit-identical, and dropped
because the fps change stayed within noise:

- **Scope bounds in one pass.** `scene_bounds()` ran once per clip scope
  with shadow-casting lights, each time looping over every drawn object.
  Transforming every box once and selecting per scope measured 74.6 fps
  against 74.2, and shadow views 0.91 against 0.92 ms per frame.
- **Light colors cached.** `light_array()` parses every light's color on
  every pack. Caching the linear color per light (keyed by the color
  object's identity and the intensity) cut the mesh batch from 1.16 to
  1.00 ms per frame, but measured 73.5 / 74.8 fps against 74.4 / 73.9.
- **Fewer wgpu calls.** After steps M to O the remaining binding overhead
  is spread thin. The largest part, `finish()` and `submit()` once per
  sub-frame, is required: buffer writes of a sub-frame must land after
  the previous sub-frame's commands ran, and the buffers are shared.
  Rendering masks straight into their layer instead of a scratch target
  plus a copy is not possible while the mask pass samples the same array
  texture. Shadow passes only run for changed layers now (0.47 ms GPU per
  frame in `PortalOrbit`). Nothing left promised more than 1 to 2%.

### P. Alpha per object, not baked into styles

`Node.alpha` (fades, `FadeIn`, `FadeOut`, `set_alpha`) was multiplied into
every `Fill` and `Stroke` opacity at snapshot time. So during a fade every
shape below got a new style version, and `RetainedVectors` rebuilt and
uploaded its per-point color records in every sub-frame; a group's fade
also rebuilt every snapshot below it, since the alphas above were part of
each snapshot's token.

Now `RNode.alpha` carries the node's `alpha`, `SceneFlattener` multiplies
it down the tree like the world matrix, and the vector shader multiplies
every color's alpha by the object's (one more float in the object entry,
80 to 96 bytes). Styles and their tokens no longer contain any alpha. A
fade rewrites one object entry; a group's fade reuses every snapshot
below it. Meshes multiply the material's opacity and lights their
intensity by the item's alpha, in `MeshBatch`.

Measured with the power profile `balanced` (so absolute numbers are below
the other rows), committed code and the change alternating:

| | fps (blur 8) | frame_at | vector records |
|---|---|---|---|
| before | 62.0 / 60.9 | 3.49 ms | 2.60 ms |
| after | 63.8 / 63.9 | 3.07 ms | 2.42 ms |

About +4% on the benchmark, which fades little; fade-heavy scenes gain
more. Frames are bit-identical to the reference: the GPU's float32
product equals the float32 of the former float64 product on every
reference frame. `test_alpha_renders_like_the_opacity_it_multiplies`
allows a difference of 1 in 8 bits for other values.

### Q. Frames like the one before are held

A frame whose sub-frames all show what the frame before showed (waits,
pauses between plays) was rendered again in full: eight sub-frames, their
composites and a readback. ManimGX holds such frames instead. Now the
composer keys every sub-frame's views (`_TreeKeys`: leaves by identity,
since render nodes are immutable and snapshots of unchanged nodes are the
same objects; groups by placement and children; nodes of the frame before
by their remembered key, so a reused group costs a lookup). A frame equal
to the frame before, with the same config apart from its time, is not
rendered (`FrameComposer.held`), and `FramePipeline` resolves its future
with the previous image, with no copy and no readback. Frames with a
`ShaderEffect` are never held: it may read `frame.time`.

Power profile `balanced`, like step P:

| | fps (blur 8) |
|---|---|
| before | 63.8 / 63.9 |
| after (three runs) | 69.0 / 68.2 / 68.5 |

+7%, frames bit-identical. Dropped in the same round: rendering a frame
whose sub-frames are all equal once, with weight one, instead of eight
times at one eighth. Holding already covers every such frame but the
first of a still stretch, so it measured 70.4 against 69.0 fps (one run
each, within noise), and it changed four reference frames by 1 of 255
(one render against eight summed in a 16-bit float accumulator).

### R. The core evaluates and packs (CORE.md, phases 1 to 4)

Frames come from `PlannedTimeline` (`--core`): the core (Rust,
`manimx._core`) evaluates every signal of a scene, flattens its scene
graph and packs the vector and mesh records, lights and shadow views
itself. No `RNode` tree, `FlatScene` or per-object Python per sub-frame;
Python is only called for user functions (`derive()`, unknown rates,
animations without a data form) and, once per version, for records built
from geometry (triangulated fills, stroke curves, interleaved meshes,
material entries). `WGPURenderer` renders the same packets.

Power profile `balanced`, both paths measured in the same session:

| | fps (blur 8) | fps (no blur) | evaluate | flatten + pack |
|---|---|---|---|---|
| RNode path | 74.3 | 322.2 | 2.34 ms | 6.03 ms |
| core | 105.8 | 415.3 | 1.10 ms | 1.40 ms |

+42% with motion blur, +29% without. The 48 reference frames are
identical, and so is every frame compared across the benchmark,
example-project and light-show (tests/test_core_render.py compares
pixels of every example scene, a scene with every node type, motion blur
and the projects). light-show, seconds 10 to 12 at 1920×1080 with motion
blur 8: 59.7 ms per frame on the RNode path, 25.0 ms on the core, which
is near its 19 ms of GPU work.

What remains per frame on the benchmark is mostly outside the core:
`render` (the wgpu-py calls, 2.16 ms), `composite` and the readback.

### S. The core renders (CORE.md, phase 6)

The passes, the composer (composite, bloom, shader effects, tone map,
finalize) and the frame pipeline's readback run in Rust on wgpu
(`manimx._core.FrameRenderer`, driven by `CoreFramePipeline`), with the
same WGSL. Python only asks the timeline for each sub-frame's views and
builds each camera's Frame record. Power profile `balanced`:

| | fps (blur 8) | fps (no blur) | GPU ms/frame |
|---|---|---|---|
| core packs, Python renders (step R) | 105.8 | 415.3 | 6.13 |
| core packs and renders | 123.5 | 604.0 | 6.83 |

+17% with motion blur, +45% without; the benchmark is now within 1.2
times its GPU time. light-show, seconds 10 to 12: 25.0 to 23.1 ms per
frame (GPU about 19). Every frame compared across the three projects and
the test scenes (motion blur, bloom, both shader stages, tone mapping,
several views, empty frames) is bit-identical to the Python renderer
(tests/test_core_pipeline.py).

Two things mattered for the readback: waiting for each frame's own
submission (waiting for the latest one serialized CPU and GPU: 94 fps),
and copying the mapped image once, straight into the Python `bytes`
(two copies out of mapped memory: 272 fps without blur).

### T. Depth of field reads each depth sample once

The depth of field gathered 64 taps per pixel, and every tap read all
MSAA samples of both depth buffers (eight loads at 4×MSAA) to find its
distance. Now a pass before it (view_distance.wgsl) resolves the
distance of every pixel into an `r32float` texture, and the blur reads
one value per tap. Showcase `Cameras` (four views, depth of field, motion
blur 8), power profile `balanced`:

| | fps | GPU ms/frame | depth of field |
|---|---|---|---|
| before | 6.9 | 105.1 | 82.1 |
| after | 19.6 | 34.7 | 11.9 |

Frames are bit-identical, in the Python renderer and in the core.

### U. Lighting once per pixel, per sample only on edges

The lighting pass shaded every MSAA sample (4 per pixel), although the
geometry pass writes the same G-buffer values to all samples of a pixel
inside a surface. Now three fullscreen triangles light the frame: one
marks the pixels whose samples hold different surfaces in a stencil
(depth, normal, colors and clip scope compared with small tolerances, so
a smooth mesh's triangle seams count as one surface), one shades the
other pixels once, at their first sample, and one shades the marked
pixels per sample. Showcase, motion blur 8 and bloom forced on, power
profile `balanced`, GPU lighting ms/frame and fps:

| scene | lighting before | after | fps before | after |
|---|---|---|---|---|
| Solids | 23.8 | 9.9 | 27.3 | 50.4 |
| Cornell | 7.7 | 5.7 | 65.3 | 84.0 |
| Cameras | 13.2 | 10.2 | 19.5 | 22.2 |

Frames are not bit-identical: an interior pixel now gets one turn of
the soft shadow taps instead of four. Against the previous renderer,
Cornell differs by 0.07 levels on average (0.06% of pixels by more
than 8), Solids by 0.06; the `shadows` reference image was updated.
Python renderer and core still match.

### V. Stroke records in the core

The core asked Python (`stroke_records`) for a path's stroke records
whenever its points or stroke style changed: for every morphing,
plotted or otherwise derived path, in every sub-frame. The core now
builds them in Rust (core/src/strokes.rs), byte for byte as the Python
reference does (tests/test_core_strokes.py), and calls Python only for
style values it does not read. Showcase, motion blur 8 and bloom
forced on, power profile `balanced`, fps:

| scene | before | after |
|---|---|---|
| Graphs | 88.8 | 109.9 |
| Morphing | 155.4 | 203.3 |
| Measures | 166.5 | 201.2 |
| Lines | 148.4 | 184.5 |

### W. The render object tree as the interface

The renderer no longer reads the evaluator's slots: the evaluator builds
the render object tree in Rust (`SceneGraph::snapshot`, every value kept
while its slots keep their versions), and the renderer packs the tree
(CORE.md, "The interface, corrected"). The Python callbacks for meshes,
materials, light colors, the default light and texture mips are Rust
now. Power profile `balanced`, against the commit before on the same
machine:

| | fps (blur 8) | fps (no blur) | frame_at |
|---|---|---|---|
| before (packing from slots) | 121.8 | 606.9 | 1.43 ms |
| after (packing a tree) | 130.2 | 628.5 | 1.49 ms |

light-show, seconds 10 to 12: 17.7 to 15.8 ms per frame. Building the
tree costs about 0.01 ms per sub-frame over evaluating alone; the gain
comes from the callbacks into Python that are gone.

The read-back first measured 453 fps without blur: the image now goes
to Python as a numpy array that takes over a Rust `Vec`, and copying the
mapped rows into a fresh `Vec` faulted its pages in one by one between
the slow reads from mapped memory (1.6 ms per 1080p frame). Zeroing the
`Vec` first (`vec![0; n]`), as `PyBytes::new_with` did before, takes the
copy back to 1.1 ms.

### X. NV12 for video, converted on the GPU

A video sink used to get RGBA frames (4 bytes per pixel, 8.3 MB per 1080p
frame), read back from the GPU, piped to ffmpeg and converted there to
the Y'CbCr 4:2:0 an H.264 encoder takes; for a hardware encoder the
frame then went back to the GPU. Now the last pass writes the two NV12
planes on the GPU (finalize.wgsl, BT.709 limited range), so a third of
the bytes is read back and piped, and ffmpeg converts nothing (VideoFile
asks for NV12 through `FrameSink.pixel_format`).

Benchmark project, 2014 frames at 1080p without motion blur, one
process, into an .mp4:

| encoder | before | NV12 |
|---|---|---|
| NVENC | 128.3 fps (7.79 ms) | 190.3 fps (5.26 ms) |
| x264 | 107.2 fps (9.33 ms) | 127.9 fps (7.82 ms) |

The colors also became right: ffmpeg's own conversion used BT.601 and
left the file untagged, which players show as BT.709 for HD sizes; the
file is now BT.709 and tagged so.

## Where the time goes now

Benchmark project with `PortalOrbit`, after steps L to O: 48.6 to about
74 fps with motion blur (+53%), 287.6 to about 385 without. Per part,
milliseconds per frame with motion blur 8:

| part | before L | now | GPU now |
|---|---|---|---|
| `PortalOrbit` (`RShape3D`, six lights) | 38.8 | 22.1 | 14.5 |
| `Intro`, `Bounce`, `Outro` (`RShape2D`) | 6.0 | 6.1 | 1.6 |

`PortalOrbit` is now within 1.5 times its GPU time; the lighting pass
alone needs 9.7 ms of GPU per frame (every MSAA sample, every light, eight
sub-frames). The 2D scenes did not change: their cost is the animation
side (`frame_at`, about 20%), the wgpu-py bindings (about 27%) and the
per-shape Python bookkeeping in `RetainedVectors.build()` (about 18%),
so they stay CPU-bound at nearly four times their GPU time.
