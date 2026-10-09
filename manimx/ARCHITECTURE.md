# manimx architecture

How manimx works as a whole: which classes exist, who creates and owns their
instances, how long they live, how data flows from `Scene.construct()` to
pixels and sound, and why it is built this way. Everything is named as in
the Python code. Where two classes share a name, the module comes first
(`animation.Group` is a `Node`). Render classes carry an R prefix
(`RGroup`, `RShape2D`, ...): an `RGroup` is the snapshot of an
`animation.Group`.

More detail: the renderer's internals (passes, shaders, buffers) in
[manimx/renderer/README.md](manimx/renderer/README.md), the animation
engine's design rationale in `docs/animation_engine_design.md` (repository
root), performance work in [PERFORMANCE.md](PERFORMANCE.md), usage in
[GETTING_STARTED.md](GETTING_STARTED.md).

Contents:

1. [The big picture](#1-the-big-picture)
2. [Packages](#2-packages)
3. [From `Scene.construct()` to a video](#3-from-sceneconstruct-to-a-video)
4. [The animation package](#4-the-animation-package)
5. [Timing: `Timeline`, `Clip`, `Segment`, `MonotoneMap`](#5-timing-timeline-clip-segment-monotonemap)
6. [`World2D`, `World3D`, `animation.Camera`, `View`](#6-world2d-world3d-animationcamera-view)
7. [`RNode`: the contract between both sides](#7-rnode-the-contract-between-both-sides)
8. [The render chain and `RenderService`](#8-the-render-chain-and-renderservice)
9. [Drivers: `FramePipeline`, processes, `FrameSink`, audio](#9-drivers-framepipeline-processes-framesink-audio)
10. [`Project` and the CLI](#10-project-and-the-cli)
11. [Lifetimes at a glance](#11-lifetimes-at-a-glance)
12. [Design decisions](#12-design-decisions)
13. [State of the work and next steps](#13-state-of-the-work-and-next-steps)

---

## 1. The big picture

manimx has two passes:

1. **Compile once.** `Scene.prepare()` runs `Scene.construct()` once and
   records what happens when:
   - every `Animation` gets a `Placement` and one `Entry` on the
     `SignalTrack` of each `RootSignal` it changes;
   - every `Simulation` is baked into recorded values;
   - every `Emit`, `Cue` and `SoundEvent` is stored in its `Segment`.

   The result is a `PreparedScene`.
2. **Evaluate anywhere.** `PreparedScene.state_at(t)` and
   `Timeline.state_at(t)` return a tree of `RNode`s for any time, in
   any order and in any process, without replaying anything.
   The render chain (`SceneFlattener`, `FramePacker`, `WGPURenderer`,
   `FrameComposer`) turns such a tree into pixels.

Everything else follows from this split: random access makes scrubbing,
retiming, cutting and parallel rendering possible. `RenderService` is a
long-lived service that keeps GPU data per `RNode.uid` across all
calls and only patches what changed.

```mermaid
flowchart LR
    Code["Scene.construct()"] -->|"Scene.prepare() — once"| PS["PreparedScene<br/>SignalTrack, Segment"]
    PS --> TL["Timeline<br/>Clip, World2D/World3D, events"]
    Proj["manimx.json"] -->|"Project.build_timeline()"| TL
    TL -->|"Timeline.state_at(t) — any t, any order"| RO["RNode tree<br/>(RGroup, RShape2D, RShape3D, ...)"]
    RO --> R["RenderService.shared()<br/>SceneFlattener, FramePacker,<br/>WGPURenderer"]
    R --> C["FrameComposer<br/>MotionBlur, Bloom, ToneMap"]
    C --> D["drivers.render()<br/>FramePipeline, worker processes"]
    D --> Out["VideoFile / ImageSequence"]
    TL -->|"Timeline.audio(start, end)"| D
```

## 2. Packages

| Package | Contents | Depends on |
|---|---|---|
| `manimx.robjects` | `RNode` and its subclasses `RGroup`, `RClippingGroup`, `RBezierPath` (→ `RShape2D`, `RCurve3D`), `RShape3D` (→ `RParticles`), `RLight` (→ `RDirectionalLight`, `RPointLight`, `RSpotLight`); `Transform`, `Fill`, `Stroke`, `MeshData`, `PbrMaterial`, `Texture`; `new_uid()`, `versioned()` | numpy |
| `manimx.animation` (core) | `RootSignal`, `DerivedSignal`, `Vector3Signal`, `TimeSignals`; `Node` and its subclasses (`Group`, `ClippingGroup`, `Shape2D`, `Curve3D`, `Shape3D` (→ `Particles`), `Light` → `DirectionalLight`, `PointLight`, `SpotLight`; `animation.Camera`), `PositionMixin`, `RotationMixin`, `ScalableMixin`; the `Animation` contract (`Animation`, `SetValue`, `ChangeSignals`/`AnimationBuilder`) and the default rates `linear`, `smooth`; `Scene`, `SceneCompiler`, `PreparedScene`; `Timeline`, `Clip`; `World2D`, `World3D`, `animation.Camera`, `View`; `Simulation`; `Sound` | robjects; from `manimx.renderer` only the dataclasses `renderer.Camera` and `ViewState` |
| `manimx.userland` | everything a project could write on top of the core: shapes (`Circle`, `Dot`, `Square`, `Rectangle`, `RegularPolygon`, `Triangle`, `Polyline`, `ParametricCurve3D`, `Box`, `Sphere`, `Torus`, `ParametricSurface`), animations (`FadeIn`, `FadeOut`, `Rotate`, `Indicate`, `GrowFromCenter`, `ShrinkToCenter`), `rate_functions`, `constants` (`UP`, `PI`, ...), `geometry` (every point and mesh generator), `load_gltf()` | animation, robjects |
| `manimx.renderer` | `SceneFlattener` → `FlatScene`; `tessellate_fill()`; `RNodeDataCache`; `RenderConfig`, `renderer.Camera`, `ViewState`; `MotionBlur`, `Bloom`, `ToneMap`, `ShaderEffect`; `RemoveObjects`, `ClearScene` | robjects |
| `manimx.renderer.webgpu` | `RenderService`, `FramePacker` → `FramePacket`, `WGPURenderer`, `FrameComposer`, `RetainedVectors`, `VectorBatch`, `MeshBatch`, `VectorPass`, `MeshPass`, `LightingPass`, `ShadowPass`, `RenderTargets`, `RenderLayers`, `ResourceCache`, `GPUContext` | wgpu |
| `manimx.drivers` | `render()`, `FramePipeline`, `RenderSettings`, `FrameSource`, `load_source()`; `FrameSink` (→ `VideoFile`, `ImageSequence`, `FrameList`) | renderer, animation |
| `manimx.project` | `Project`, `SceneEntry`, `ClipEntry`, `ProjectSource` | animation, drivers, settings |
| `manimx.settings` | `Settings` and its sections (`OutputSettings`, `EffectsSettings`, `LightingSettings`, `PerformanceSettings`), `setting()` | renderer (config, effects) |
| `manimx.cli` | the Typer `app` (`manimx` command), `scaffold`, `ui` | project, drivers |

`manimx.animation` never touches the GPU, and `manimx.renderer` never sees a
`RootSignal` or a time: `RNode` trees are the only interface between
them.

## 3. From `Scene.construct()` to a video

```mermaid
sequenceDiagram
    autonumber
    participant CLI as cli.app.render
    participant P as Project
    participant S as Scene
    participant TL as Timeline
    participant D as drivers.render
    participant FP as FramePipeline
    participant FC as FrameComposer
    participant R as RenderService.shared()
    participant W as _Writer thread
    participant FF as ffmpeg (VideoFile)

    CLI->>P: Project.find() reads manimx.json
    P->>S: import scenes.intro, Intro().prepare()
    S-->>P: PreparedScene
    P->>TL: Timeline.add() per ClipEntry, Clip.cut(), events
    CLI->>D: render(timeline, VideoFile, RenderSettings)
    D->>TL: Timeline.audio(first frame, last frame)
    D->>FF: VideoFile.open(FrameFormat with the audio)
    loop every time t of RenderSettings.frame_times()
        D->>FP: FramePipeline.submit(index, t, timeline.state_at)
        FP->>FC: FrameComposer.compose_texture(state_at, RenderConfig at t)
        loop every time s of MotionBlur.sample_times(t)
            FC->>TL: Timeline.state_at(s) returns an RNode tree
            FC->>R: render_layers(tree, RenderConfig, encoder): flatten, pack, render; returns RenderLayers
            FC->>FC: composite RenderLayers into the HDR accumulator
        end
        FC-->>FP: output texture
        FP->>FP: copy into a staging buffer, map_async
        Note over FP: up to RenderSettings.in_flight frames in flight
        FP-->>D: Future with the (H, W, 4) uint8 image
        D->>W: put(index, image)
        W->>FF: VideoFile.write in index order
    end
```

## 4. The animation package

### 4.1 Signals

Every animatable property of a `Node` is a signal:

- **`RootSignal`**: holds a value set from outside, by `authoring.change()`,
  by `SignalTrack.value_at()` during evaluation, or by a `Simulation` step.
- **`DerivedSignal`**: a pure function of other signals, created with
  `derive()` (and `PositionMixin.follow()`).
- **`SignalKind`** instances say how values interpolate and combine:
  `SCALAR`, `FACTOR`, `ROTATION` (slerp), `COLOR`, `PATH` (point arrays),
  `DISCRETE`.
- **`Vector3Signal`**: three `RootSignal`s `x`, `y`, `z`, so animations of
  different axes never interfere.
- **`TimeSignals`**: `scene_time` (stands still while a `Segment` waits) and
  `global_time` (always runs).

Setting a `RootSignal` invalidates its dependents; a `DerivedSignal`
recomputes when it is read. Every signal has a process-unique `id` and a
`version` that increases exactly when its value changes. Versions drive all
caching downstream.

### 4.2 `Node`, `Animation`, `SignalTrack`

```mermaid
classDiagram
    direction LR
    class Node {
        uid
        name
        tags
        alpha: RootSignal
        present: RootSignal
        snapshot() RNode
    }
    class PositionMixin {
        position: Vector3Signal
        shift(), move_to(), follow()
    }
    class RotationMixin {
        rotation: RootSignal
        rotate()
    }
    class ScalableMixin {
        scale: Vector3Signal
        scale_by(), set_scale()
    }
    class Shape2D {
        points: RootSignal
        fill_color, fill_opacity
        stroke_color, stroke_width, stroke_opacity
        stroke_joint, stroke_cap
        cast_shadows
    }
    class Group {
        children
        find(name), find_all(name, tags)
    }
    class ClippingGroup {
        mask, content
        one_sided: RootSignal
        find(name), find_all(name, tags)
    }
    class Curve3D {
        points: RootSignal
        stroke_color, stroke_width, stroke_opacity
        stroke_joint, stroke_cap
        cast_shadows
    }
    class Shape3D {
        mesh: RootSignal
        base_color, opacity, metallic, roughness
        emissive, emissive_strength
        normal_scale, occlusion_strength
        five texture signals
        cast_shadows
    }
    class Light {
        color, intensity, size
        cast_shadows
    }
    class Camera {
        frame_height
        field_of_view
        perspective
        render_camera()
    }
    Node <|-- Shape2D
    Node <|-- Group
    Node <|-- ClippingGroup
    Node <|-- Camera
    PositionMixin <|-- Shape2D
    RotationMixin <|-- Shape2D
    ScalableMixin <|-- Shape2D
    Shape2D <|-- Circle
    Shape2D <|-- Square
    Shape2D <|-- Polyline
    Node <|-- Curve3D
    Node <|-- Shape3D
    Node <|-- Light
    Shape3D <|-- Sphere
    Shape3D <|-- Box
    Shape3D <|-- ParametricSurface
    Light <|-- DirectionalLight
    Light <|-- PointLight
    Light <|-- SpotLight

    class Animation {
        run_time
        rate
        targets() Target[]
        contribution(signal, alpha, start_value)
    }
    Animation <|-- ChangeSignals
    Animation <|-- FadeIn
    Animation <|-- Simulation
    Animation <|-- SetValue

    class Placement {
        animation
        start
        duration
        order
        rate
    }
    class Entry {
        placement
        mode: Change
        start_value
    }
    class SignalTrack {
        signal: RootSignal
        base
        entries
        value_at(t)
    }

    Animation "1" <-- "1" Placement : references
    Placement "1" <-- "*" Entry : one per RootSignal
    SignalTrack "1" o-- "*" Entry
    SignalTrack --> RootSignal : sets during evaluation
    Animation --> Node : references (no copies)
```

`Group`, `ClippingGroup`, `Camera` and the lights in this diagram are the
`manimx.animation` classes. `Circle`, `Square`, `Polyline`, `Sphere`, `Box`,
`ParametricSurface` and `FadeIn` are userland (`manimx.userland`), built on
the core classes they derive from. The mixins are left out of the diagram: every
`Node` subclass takes all three, except `Camera` (position and rotation),
`DirectionalLight` (rotation), `PointLight` (position) and `SpotLight`
(position and rotation).

- **The scene tree is parallel to the render tree.** Every `Node` class
  snapshots to exactly one R class (`Shape2D` → `RShape2D`, `Curve3D` →
  `RCurve3D`, `Shape3D` → `RShape3D`, `DirectionalLight` →
  `RDirectionalLight` and so on, `Group` → `RGroup`, `ClippingGroup` →
  `RClippingGroup`); `Camera` snapshots to nothing. Only `Group` and `ClippingGroup` have children, so `family()`
  only descends through them.
- **The transform comes in parts.** `PositionMixin`, `RotationMixin` and
  `ScalableMixin` each add one signal and the changes that use it, and a
  class takes the parts that mean something for it: `Group`,
  `ClippingGroup` and `Shape2D` take all three, `Camera` takes position and
  rotation. A missing part is the identity in the snapshot's `Transform`.
  Using a missing part raises an error that names the mixin, for example
  `Indicate(camera)` or `camera.scale_by(2)`.
- **Names and tags are for authoring only.** `Node.name` (meant to be
  unique, not checked) and `Node.tags` (a set of strings) work like an
  HTML id and classes: `group.find(name)` returns the one match below the
  group or raises a `LookupError` with close names, and
  `group.find_all(name=, tags=)` returns every match in tree order. Neither
  crosses to the render side; snapshots carry only the `uid`.

- **An `Animation` references its `Node`s** and never copies them.
  `Animation.targets()` names the `RootSignal`s it changes, each with a
  `Change` (`RELATIVE` or `ABSOLUTE`). `Animation.contribution()` is a pure
  function of the progress.
- **`Scene.play()`** creates one `Placement` per `Animation`, with its start,
  duration, authoring order and rate function.
- **Every `RootSignal` an `Animation` changes** gets a `SignalTrack` with
  one `Entry` per `Placement`. `SignalTrack.value_at(t)` folds the entries in
  authoring order: `RELATIVE` contributions combine, `ABSOLUTE` ones
  override, starting from `Entry.start_value`.
- **A direct call such as `square.shift(RIGHT)` inside `construct()`**
  becomes a `SetValue` placement at the current cursor
  (`authoring.change()`), so it does not rewrite the square's past.

### 4.3 `Scene.prepare()` and `PreparedScene.state_at()`

```mermaid
flowchart TD
    subgraph Compile["Scene.prepare() — once"]
        A["Scene.construct()"] -->|"add, play, wait, wait_for, emit, cue, play_sound"| SC["SceneCompiler<br/>(active through a ContextVar)"]
        SC --> T["SignalTrack per RootSignal"]
        SC --> SG["Segment: offset, duration, waits_for,<br/>emits (Emit), cues (Cue), sounds (SoundEvent)"]
        SC -->|"play(Simulation)"| BK["Simulation.bake():<br/>run step(dt), record values"]
        BK --> T
    end
    T & SG --> PS["PreparedScene<br/>duration, tracks, roots, segments, time"]
    subgraph Eval["PreparedScene.state_at(t)"]
        PS --> E1["TimeSignals.set(scene_time, global_time)"]
        E1 --> E2["every SignalTrack: signal.set(value_at(t))"]
        E2 --> E3["DerivedSignals recompute when read"]
        E3 --> E4["Node.snapshot() for every present root"]
        E4 --> RO["RGroup of RNodes"]
    end
```

- **Evaluation sets the `RootSignal`s and reads them back**, so a
  `PreparedScene`'s `Node`s are its evaluation state. That is why one
  `PreparedScene` backs only one `Clip`.
- **Purity:** the result of `state_at(t)` depends only on `t`.
- **`Node.snapshot()`** builds new `RNode`s on every call
  (one R class per class, see section 4.2; `animation.Camera` →
  nothing). Each keeps the `Node.uid` and carries version tokens from
  `version_token()` (section 7).

### 4.4 `Simulation`

A `Simulation` wraps the user's `step(dt)` function, which changes
`Node`s however it likes. `SceneCompiler.play()` bakes it with
`Simulation.bake()`:

- `step` runs at the fixed step `dt` from start to end;
- before every step, each `RootSignal` with a `SignalTrack` from the same
  `play()` call (or earlier) is set to its value at that time;
- every `RootSignal` the step changes is recorded at `keyframe_rate`, in
  `Simulation.times` and the recorded values.

Afterwards the `Simulation` is an `ABSOLUTE` `Animation` that interpolates
the recorded values. `Simulation.emit()` and `Simulation.play_wave()` add an
`Emit` and a `SoundEvent` at the simulated time. There is no built-in physics
engine; this is deliberate.

### 4.5 `Sound`

A `Sound` holds float32 samples (`(count, 1)` or `(count, 2)`) and a
`sample_rate`. It comes from `Sound.load()` (WAV directly, anything else
through ffmpeg) or from an array. `Scene.play_sound()` and
`Scene.play_wave()` record a `SoundEvent` in the current `Segment` without
moving the cursor. `Timeline.audio()` (via `Clip.placed_sounds()` and
`mix()`) starts each `Sound` at the global time its point in the scene is
shown, then plays it at its own speed, so a `MonotoneMap` never changes its
pitch.

## 5. Timing: `Timeline`, `Clip`, `Segment`, `MonotoneMap`

```mermaid
classDiagram
    direction LR
    class Timeline {
        clips
        worlds
        events: dict name to time
        live
        add()
        fire()
        state_at(t, world)
        view_at(view, t)
        audio(start, end)
        resolve()
    }
    class Clip {
        scene: PreparedScene
        start
        world, transform, layer
        scales, ramps, cue_events
        cut_in, cut_out
        timings: SegmentTiming[]
    }
    class SegmentTiming {
        start
        end
        compiled_start
        compiled_end
        map: MonotoneMap
    }
    class Segment {
        offset
        duration
        waits_for
        emits: Emit[]
        cues: Cue[]
        sounds: SoundEvent[]
        play_offset
    }
    Timeline "1" *-- "*" Clip
    Timeline "1" *-- "*" World
    Clip --> PreparedScene : backs exactly one
    Clip --> World
    Clip "1" *-- "*" SegmentTiming : resolved
    PreparedScene "1" *-- "*" Segment
```

- **`Segment`:** `Scene.wait_for(name)` closes a `Segment`. Segments lie on
  one compiled time axis, separated by `SEGMENT_GAP`, which is never shown.
  `Segment.play_offset` counts the time without those gaps.
- **Event names:** they get their times from `Timeline.events`
  (`Timeline.fire()`) or from an `Emit` in any `Clip`.
  `Timeline._resolve()` iterates to a fixed point: a `Segment` starts at the
  time of its `waits_for` event, or when the previous one ends if that is
  later. While editing, resolution is lenient. `Timeline.resolve()` is
  strict and raises `UnresolvedEvents` or `InvalidCue`. With
  `Timeline.live`, unfired events hold.
- **`MonotoneMap`:** each `SegmentTiming` maps global time to compiled time
  with a monotone cubic (PCHIP) through these knots:
  - the start;
  - every `Cue` whose event is known, so the cue lands exactly on it;
  - the edges of every `Ramp` (`Clip.slow_motion()`);
  - the end.
- **Lazy resolution:** `Clip.cut()`, `Clip.slow_motion()`,
  `Clip.bind_cue()` and `Clip.set_scale()` mark the `Timeline` dirty, and the
  next query resolves it once.

## 6. `World2D`, `World3D`, `animation.Camera`, `View`

- **`World2D` and `World3D`:** every `Clip` has one `Clip.world`.
  `World2D.check_transform()` keeps its clips in the z = 0 plane.
  `Timeline.state_at(t, world)` returns one world's `RGroup`.
- **`animation.Camera`:** a `Node` whose `RootSignal`s are `position`,
  `rotation`, `frame_height`, `field_of_view` and `perspective`; it is
  animated like any other `Node`. `snapshot()` returns `None`.
  `Timeline.camera_at()` evaluates its `Clip` at `Clip.held_time()`, so it
  keeps its first and last placement outside the clip.
  `animation.Camera.render_camera()` returns a `renderer.Camera`.
- **`View(world, camera)`:** `Timeline.view(view)` is a function of time
  returning `ViewState(root, camera)`. `FrameComposer` uses
  `ViewState.camera` for every motion-blur sample.
- **Showing a `View` inside a `World2D`** (compositions) is not built yet.

## 7. `RNode`: the contract between both sides

`RNode` (`manimx.robjects`) is the only thing that crosses from
`manimx.animation` to `manimx.renderer`. These rules make every renderer
cache correct:

1. **`RNode.uid`:** every snapshot of one `Node` has the
   `Node.uid`. `new_uid()` never repeats within a process.
2. **Versions:** fields declared with `versioned()` (`RBezierPath.points`,
   `RBezierPath.stroke`, `RShape2D.fill`, `RShape3D.mesh`) have a token,
   `RNode.version(name)`. Every `snapshot()` passes tokens from
   `version_token()` of the `RootSignal`s a field is computed from, so
   equal tokens mean equal values:
   - `Shape2D`: `points` from `points`; `fill` from `fill_color` and
     `fill_opacity`, `stroke` from the five stroke signals. While a token
     holds, the snapshot reuses its `Fill` and `Stroke` values
     (PERFORMANCE.md, step F);
   - `Curve3D`: `points` and `stroke` the same way;
   - `Shape3D`: `mesh` from the `mesh` signal (a `DISCRETE` signal holding
     a `MeshData`, whose version changes only when another `MeshData` is
     set); `material` from the six factor signals.
     Moving, turning or scaling a `Shape3D` therefore keeps both tokens,
     and `MeshBatch` keeps its interleaved vertices;
   - lights have no versioned fields.

   `RNode.transform` is not versioned; it is compared directly.
   Neither is `RNode.alpha`: every snapshot carries its `Node.alpha` there,
   `SceneFlattener` multiplies the alphas down the tree (`DrawItem.alpha`,
   `LightItem.alpha`), and the renderer applies the product per object
   (the vector shader multiplies every color's alpha, `MeshBatch` the
   material's, `light_array()` the intensity). An `alpha` therefore never
   enters a style or its version token: a fade changes one number per
   object, and a group's fade reuses the snapshots of everything in it.
3. **Immutability:** an `RNode` is never changed after `state_at`
   returns it, and versioned arrays are read-only.
4. **Purity:** `state_at` has no side effects on its result.
5. **Lifecycle:** `renderer.submit(RemoveObjects(...))` drops the data of
   objects gone for good. Otherwise `RNodeDataCache.sweep()` ages entries
   out (section 8).

## 8. The render chain and `RenderService`

A tree becomes a frame through four stages, each its own object with one
input and one output:

```mermaid
flowchart LR
    T["RNode tree"] -->|"SceneFlattener.flatten()"| F["FlatScene"]
    F -->|"FramePacker.pack()"| P["FramePacket"]
    P -->|"WGPURenderer.render_layers()"| L["RenderLayers"]
    L -->|"FrameComposer"| Out["frame"]
```

```mermaid
classDiagram
    direction TB
    class RenderService {
        shared()$ RenderService
        cache: RNodeDataCache
        flattener: SceneFlattener
        packer: FramePacker
        renderer: WGPURenderer
        composer(effects) FrameComposer
        render(frame, config) image
        render_layers(root, config, encoder) RenderLayers
        submit(command)
    }
    class SceneFlattener {
        flatten(root) FlatScene
    }
    class FramePacker {
        cache: RNodeDataCache
        pack(FlatScene, config) FramePacket
    }
    class FramePacket {
        config
        vectors: VectorBatch
        meshes: MeshBatch or None
        shadows: ShadowViews or None
        frame, scope_parents, mask_count
        source
    }
    class RetainedVectors {
        arenas: _Arena per buffer
        slots: _Slot per uid and mask scope
        build(FlatScene) VectorBatch
    }
    class WGPURenderer {
        context: GPUContext
        resources: ResourceCache
        profiler: GpuProfiler
        render_layers(packet, encoder) RenderLayers
    }
    class FrameComposer {
        effects
        _TexturePool
        _Fullscreen
        compose_texture(scene, config)
    }
    RenderService *-- SceneFlattener
    RenderService *-- FramePacker
    RenderService *-- WGPURenderer
    RenderService "1" *-- "*" FrameComposer : one per effect tuple
    FramePacker *-- RNodeDataCache
    FramePacker *-- RetainedVectors
    FramePacker *-- RetainedMeshes
    FramePacker ..> FramePacket : builds
    WGPURenderer ..> FramePacket : reads
    WGPURenderer *-- GPUContext
    WGPURenderer *-- ResourceCache
    WGPURenderer *-- RenderTargets : per size
    WGPURenderer *-- VectorPass
    WGPURenderer *-- MeshPass
    WGPURenderer *-- LightingPass
    WGPURenderer *-- ShadowPass
    FrameComposer --> RenderService : runs the chain
```

- **`SceneFlattener`** resolves the hierarchy into a `FlatScene`:
  `DrawItem`s with world matrices and painter's order, `ClipScope`s,
  `LightItem`s.
- **`FramePacker`** turns a `FlatScene` into a `FramePacket`: the numpy
  records the shaders read (`VectorBatch` from `RetainedVectors`,
  `MeshBatch`, `ShadowViews`, the camera's frame record). It owns every
  per-object cache and never touches the GPU.
- **`WGPURenderer`** uploads a packet and encodes the passes into the
  encoder it is given (a `wgpu.GPUCommandEncoder`: commands are recorded,
  and the caller submits them in one go). It never sees an `RNode` and
  decides nothing about the scene.
- **`FrameComposer`** runs the chain once per sub-frame and view, applies
  the effects and composites.

**One `RenderService` per process:** `RenderService.shared()` creates the
stages once. `drivers.render()`, the CLI and the tests use it. One thread
renders at a time. It implements the `Renderer` interface: `render()`,
`submit()` for `RemoveObjects` and `ClearScene`, and `cache` (the
packer's `RNodeDataCache`).

**Each stage is usable alone.** A `WGPURenderer` renders packets from any
`FramePacker`: a packet's dirty ranges are relative to the previous packet
of the same packer (`FramePacket.source`), and a packet from another packer
is uploaded in full.

**What one pass through the chain does:**

| Step | `RNode` unchanged | `RNode` changed |
|---|---|---|
| `SceneFlattener.flatten()` → `FlatScene` | every call | every call |
| `RetainedVectors.build()` for `RShape2D` and `RCurve3D` | nothing written to its `_Slot` | new world matrix or alpha: the 96-byte object entry; new `fill`/`stroke` version: its records in place; new `points` version: records moved, curves and instances rewritten |
| `RNodeDataCache` (`tessellate_fill()` result, stroke records, center, `RShape3D` vertices) | hit | rebuilt for the new version |
| `MeshBatch.build()`, `build_shadow_views()` | skipped while the `FlatScene` contains no `RShape3D` | run when an `RShape3D` is present |
| `RetainedMeshes` (mesh vertices and indices) | its ranges kept, nothing written | new `mesh` version: written over its ranges if the size fits, else into new ones |
| a light's shadow views (`build_shadow_views()`) | cached in `RNodeDataCache` by its world matrix, settings, scope bounds and config | rebuilt |
| a shadow map layer (`ShadowLayers.mark()`) | no shadow pass while its view and its scope's casters (identity, versions, world matrix) are unchanged | rendered again |
| `GrowableBuffer.write()` in `WGPURenderer` | nothing uploaded | only the written record ranges |
| geometry and lighting passes | skipped without `RShape3D` (G-buffer and lit target cleared once) | run when an `RShape3D` is present |
| mask, shadow and vector passes | recorded into the `FrameComposer`'s command encoder | same |

**Caches and their lifetimes:**

| Cache | Owner | Key | Lives until |
|---|---|---|---|
| `RNodeDataCache` | `FramePacker` | (uid, kind) with a version | replaced on a new version; dropped after 1024 packs unused (`sweep()`), or by `RemoveObjects` / `ClearScene` |
| `ShadowLayers` keys | `FramePacker` | layer index | the next pack with meshes |
| `RetainedMeshes` slots | `FramePacker` | uid with the `mesh` version | freed when its `RShape3D` is missing from a pack with meshes; repacked when garbage outweighs live data |
| `RetainedVectors._Slot` | `FramePacker` | (uid, mask scope) | freed when its `RNode` is missing from a pack; `_Arena`s repacked when garbage outweighs live records |
| `ResourceCache.view()` | `WGPURenderer` | the texture (weak key) | its texture |
| `ResourceCache.bind_group()` | `WGPURenderer` | the layout plus the identities of the bound resources | LRU, 512 entries |
| `RenderTargets` | `WGPURenderer` | size, MSAA samples, mask capacity | the next size change |
| pipelines and shader modules | `WGPURenderer`, `FrameComposer` | variant | their owner |
| `FrameComposer` | `RenderService` | the tuple of effects | the `RenderService` |

**`FrameComposer.compose_texture()`**, for every time of
`MotionBlur.sample_times(t)` (or just `t`):
- calls the scene function;
- calls `RenderService.render_layers()` (flatten, pack, render);
- applies `Bloom` and `ShaderEffect(stage="lit")` to `RenderLayers.lit`;
- composites `RenderLayers.lit` and `RenderLayers.vector` over the background
  into the HDR accumulator, weighted by the sample (`ToneMap` for the lit
  part);
- submits one command encoder per sample.

Finally it applies `ShaderEffect(stage="frame")` and encodes to sRGB in the
output texture.

## 9. Drivers: `FramePipeline`, processes, `FrameSink`, audio

```mermaid
flowchart LR
    subgraph Main["main process: drivers.render()"]
        L["load_source(spec) → FrameSource<br/>+ FrameSource.audio()"] --> Loop
        Loop["_render_here, or the collector<br/>of _render_in_workers"] -->|"queue (bounded)"| Wt["_Writer thread<br/>reorders for ordered sinks"]
        Wt --> Sink["FrameSink"]
    end
    Sink -->|"VideoFile: stdin pipe"| FF["ffmpeg process"]
    Sink -->|"ImageSequence: PNG threads"| Files["frame_00000.png …"]
    subgraph Workers["_worker_main processes (workers > 1, spawn)"]
        W1["load_source(spec)<br/>RenderService.shared()<br/>FramePipeline<br/>chunks of frame indices"]
    end
    W1 -->|"SharedMemory slots per worker"| Loop
    Loop -->|"free slot ids"| W1
```

- **`FramePipeline`:** keeps up to `in_flight` frames on the GPU, with one
  staging buffer per frame mapped through `map_async`. `submit()` returns a
  `concurrent.futures.Future`, completed in submission order.
- **Worker processes:** each runs `_worker_main`, which rebuilds the source
  with `load_source()` from a `"module:Class"` string or a `ProjectSource`.
  Because `state_at` is pure, every process renders identical frames.
  Frames come back through `SharedMemory` slots owned per worker. A worker
  renders its frames in increasing order and waits only for its own slots,
  so in-order writing cannot deadlock.
- **`FrameSink`s:**
  - `VideoFile` runs ffmpeg and muxes the audio from `FrameFormat.audio`
    in the same pass;
  - `ImageSequence` writes PNG files plus `audio.wav`;
  - `FrameList` keeps the frames in memory.

  Sinks with `ordered = True` get frames in index order.
- **Audio:** `FrameSource.audio(start, end, sample_rate)` (from
  `Timeline.audio` or `PreparedScene.audio`) mixes exactly the rendered range
  before rendering starts. A silent mix means no audio track.

## 10. `Project` and the CLI

```mermaid
flowchart LR
    J["manimx.json<br/>output, effects, lighting, performance,<br/>scenes, timeline"] <-->|"Project.load() / Project.save()"| P["Project<br/>Settings, SceneEntry, ClipEntry"]
    P -->|"build_timeline()"| TL["Timeline"]
    P -->|"render_settings()"| RS["RenderSettings"]
    P -->|"source(scene)"| PSrc["ProjectSource"]
    CLI["cli.app: new, generate scene, remove scene,<br/>list, render, still"] --> P
    CLI --> Scaffold["scaffold.create_project(),<br/>scaffold.generate_scene(), templates"]
    UI["web UI (next)"] -.-> P
    J -.->|"$schema"| Schema[".manimx/project.schema.json<br/>from project_schema()"]
```

- **`Project` holds the state; `manimx.json` is its snapshot.** Code, the
  CLI and later the web UI change the same `Project` and call
  `Project.save()`. A `SceneEntry` references a `Scene` subclass by module
  and class.
- **`Project.load()` is strict.** It rejects unknown keys with a "did you
  mean" hint and checks types, limits, choices and colors. Older formats
  are migrated first (`_MIGRATIONS`).
- **Every setting is declared once**, with `setting()` in
  `manimx.settings`: default, description, limits, unit. The JSON schema
  (`project_schema()`), the `manimx settings` menu and its `--print`
  output, and the checks all read that declaration.
  `Settings.render_config()` and `Settings.effects_chain()` turn the
  settings into the `RenderConfig` and the composer's effects.
- **`Project.build_timeline()`** prepares every `ClipEntry`'s scene and
  calls `Timeline.add()`. A `ClipEntry` without `start` begins at the
  previous `Clip.end`.
- **`Project.scene_class()`** imports each scene module on its own, so a
  broken module only fails its own `SceneEntry`. A package of the same name
  from another project is evicted from `sys.modules` first.
- **`cli.ui`** prompts (questionary) only for missing arguments in a
  terminal. A `ProjectError` is printed as a panel with its `hint`. A
  `SceneError` comes with a rich traceback of the user's frames.

## 11. Lifetimes at a glance

| Instance | Created by | Owned by | Lives until | Changes? |
|---|---|---|---|---|
| `Node` and its signals | `Scene.construct()` | the `PreparedScene` (`roots`, `tracks`) | the `PreparedScene` | signal values on every evaluation |
| `MeshData`, `PbrMaterial`, `Texture` | user code, the mesh generators of `userland.geometry` (`box_mesh()`, `surface_mesh()`, ...), `load_gltf()` | shared: a `Shape3D`'s `mesh` signal and constant material, and the `RShape3D` snapshots | as long as anything references them | no (immutable) |
| `SceneCompiler` | `Scene.prepare()` | a ContextVar while `construct()` runs | the end of `prepare()` | yes |
| `PreparedScene` | `Scene.prepare()` (kept in `Scene._prepared`) | one `Clip` | the `Timeline` | only through evaluation |
| `SignalTrack`, `Placement`, `Entry` | `Scene.play()`, `authoring.change()` | the `PreparedScene` | the `PreparedScene` | `SignalTrack.refresh()` after edits |
| `Segment`, `Emit`, `Cue`, `SoundEvent` | `wait_for`, `emit`, `cue`, `play_sound`/`play_wave` | the `PreparedScene` | the `PreparedScene` | no (frozen) |
| `Timeline`, `Clip`, `World2D`/`World3D` | user code or `Project.build_timeline()` | the caller | the caller | yes; edits mark the `Timeline` dirty |
| `SegmentTiming`, `MonotoneMap` | `Timeline._resolve()` | the `Clip` | the next resolution | no |
| `RNode` tree | `state_at()` | the caller of `state_at()` | that render | no |
| `Project` | `Project.load()` / `Project.find()` | the CLI command | the command | yes; `Project.save()` |
| `RenderService` (`SceneFlattener`, `FramePacker`, `WGPURenderer`) | `RenderService.shared()` | the process | the process | its caches |
| `FramePacket` | `FramePacker.pack()` | the `RenderService` | one render | no |
| `FrameComposer` | `RenderService.composer(effects)` | the `RenderService` | the `RenderService` | its `_TexturePool` |
| `FramePipeline` | `drivers.render()` | the render call | the call | yes |
| `_Writer` thread, `_worker_main` processes, ffmpeg | `drivers.render()` | the render call | the call | |

## 12. Design decisions

**Animation package**

- **Two passes: `Scene.prepare()` once, `state_at()` anywhere.** Random
  access is what makes scrubbing, retiming, cuts, previews, `MotionBlur` and
  worker processes simple. Replaying from the start on every query would
  rule them out.
- **A `RootSignal` per property and per axis (`Vector3Signal`).**
  Animations of different properties or axes compose without interfering.
  `DerivedSignal` replaces updaters with pure functions, which keeps
  evaluation deterministic.
- **An `Animation` references `Node`s; it never copies them.**
  `contribution()` is a pure function of progress, and `SignalTrack` holds
  the composition rules.
- **Direct changes inside `construct()` become `SetValue` at the cursor,
  and reads (`PositionMixin.get_center()`) see everything recorded so far.** Code
  reads like Manim but never rewrites the past.
- **Versions, not hashes.** A version counter costs nothing per frame. A
  revisited time costs one recomputation and is never wrong.

**Timing**

- **`wait_for`, `emit` and `cue` take names; times come from the
  `Timeline`.** A `Scene` stays reusable: `Timeline.fire()`, an `Emit` in
  another `Clip`, or (later) transcripts bind the times.
- **`MonotoneMap` with PCHIP.** It passes exactly through each `Cue`, never
  runs backwards, and changes speed smoothly.
- **A `Sound` is never retimed.** `Clip.slow_motion()` moves sound starts
  apart, but pitch stays.

**`Simulation`**

- **Baked in `Scene.prepare()`, not run during rendering.** `state_at`
  stays random access and parallel. The user writes `step(dt)`; there is no
  engine dependency. Animated `RootSignal`s are inputs; `DerivedSignal`,
  `Emit` and `SoundEvent` carry reactions.

**`RNode` contract**

- **`uid` plus version tokens are the only change detection.** The renderer
  never compares data deeply. `RNodeDataCache` and `RetainedVectors` use
  the same keys.

**Render chain**

- **Four stages with one input and one output each.** `SceneFlattener`,
  `FramePacker`, `WGPURenderer` and `FrameComposer` can be used, tested and
  measured alone. All scene logic and every per-object cache sit before the
  `FramePacket`; the `WGPURenderer` only renders, so it could be replaced
  (or ported) without touching what decides what to draw.
- **A chain, not a wrapper.** `RenderService` holds the stages side by side
  and runs them in order; none of them wraps another.
- **`RenderService.shared()` is one service per process.** `GPUContext`,
  pipelines, `GrowableBuffer`s, `RenderTargets`, `ResourceCache` and
  per-object data survive across calls, which keeps a long-lived web UI
  cheap.
- **`RetainedVectors`** keeps records per `_Slot` in `Arena`s that mirror
  the GPU buffers, patches them, and uploads only changed ranges.
  `RetainedMeshes` does the same for mesh vertices and indices. The cost
  follows what changed, not the size of the `FlatScene`.
- **No `RShape3D` in a frame, no 3D passes.** Most Manim content is
  `RShape2D`.
- **`MotionBlur` renders real sub-frames.** Any change blurs correctly,
  including `animation.Camera` moves and appearing `Node`s.

**Drivers**

- **`FramePipeline`, the `_Writer` thread and worker processes run
  together:** GPU, CPU, ffmpeg and disk all work at the same time.
- **Workers rebuild the source with `load_source()`.** Purity guarantees
  identical frames. Spawn instead of fork, because GPU devices do not
  survive a fork.

**`Project`**

- **`manimx.json` is a snapshot of `Project`.** Code, CLI and UI share one
  API, with no separate intermediate representation and no override files.
- **`Project.load()` is strict, with a schema from `project_schema()`.**
  Typos fail loudly, and editors check the file.

**Performance policy**

- **Only measured improvements stay.** Every optimization is measured with
  `scripts/benchmark_project.py`, checked bit-identical with
  `--check-reference`, and recorded in PERFORMANCE.md, including what was
  dropped.

## 13. State of the work and next steps

**Done:**
- signals, `Node`, `Animation`, `Scene`;
- `Segment`, events, `Cue`, `MonotoneMap`;
- `World2D`/`World3D`, `Clip` placement, `animation.Camera`, `View`;
- `Simulation`;
- `Sound` (mono and stereo, files and arrays);
- `drivers.render()` with worker processes;
- `Project` and the CLI, `example-project/`, GETTING_STARTED.md;
- every render field has its signal on the scene class (checked by
  `tests/test_node_fields.py`): fills and strokes take one value or a
  list (`COLORS`, `SCALARS`: gradients, tapering widths, which
  interpolate across lengths), joints and caps are `DISCRETE`, every
  `PbrMaterial` field of a `Shape3D` is a signal, and
  `ClippingGroup.one_sided` is one. `RNode.visible` is the exception: a
  scene node is not drawn by leaving the scene (`present`) or fading out
  (`alpha`), never through that field;
- 3D scene nodes: `Curve3D` (`ParametricCurve3D`), `Shape3D` (`Box`,
  `Sphere`, `Torus`, `ParametricSurface`), `DirectionalLight`,
  `PointLight`, `SpotLight`; `cast_shadows` as a `DISCRETE` signal on every
  leaf. Known gap: the deferred mesh pass ignores `PbrMaterial.opacity`,
  so fading a `Shape3D` (its `alpha` or `opacity`) does not show yet;
- `load_gltf()` builds a node tree: one `Group` per glTF node, one
  `Shape3D` per primitive, glTF names as node names, material names as
  `material:<name>` tags;
- performance: 74.5 → 169.8 fps on `example-project/`, 2.3× faster than the
  baseline.

**Next, in the agreed order:**
1. **Web UI:** scrubbing preview, editing the `Project`, live presentation
   with clicks calling `Timeline.fire()`.
2. **Compositions:** a `View` shown as an image inside a `World2D`, nesting.
3. **HDR colors for `RShape2D`:** a float vector target and `Bloom` on
   `RenderLayers.vector`, while colors ≤ 1 stay exactly Manim's.
4. **Morphing** (a `Morph` animation, path alignment), and Whisper
   transcripts calling `Clip.bind_cue()`.
5. **Usability and performance, full time.** The remaining cost is CPU:
   `SceneFlattener.flatten()`, `RetainedVectors.build()` bookkeeping and `state_at()`.
   The GPU needs about 1.7 ms of 5.9 ms per frame.
6. **miette-style diagnostics** for `ProjectError` and `SceneError`, once
   the common errors are known.

**Working conventions:**
- The scene tree (`Node` and its subclasses) and the render tree (`RNode`
  and its subclasses) are parallel: one R class per scene class, the R
  prefix on every render tree class, and every tree class in a module
  named after it (`rshape2d.py`, `shape2d.py`). Value types (`Transform`,
  `Fill`, `MeshData`, ...) carry no prefix.
- Core and userland: everything a project could have written itself on
  top of the public API is userland (`manimx.userland`: shapes, the
  built-in animations, rate functions, constants, geometry generators,
  the glTF loader); everything that would need a change of the animation
  system or the rendering logic is core (`manimx.animation`,
  `manimx.robjects`, `manimx.renderer`). The core never imports
  userland. `manimx` itself exports the authoring API: core scene
  classes, userland and the value types nodes take; render classes stay
  in `manimx.robjects`.
- Everything that generates geometry (points of paths, meshes of solids)
  lives in `userland.geometry`; the render side has no constructors that
  generate shapes or aim lights. Examples and scenes use the shape classes
  (`Circle`, `Square`, `Polyline`, `Box`, `Sphere`, ...) and the node
  methods (`move_to`, `rotate`, `set_scale`), never render objects.
- Render objects carry no names: names belong to nodes. Identity across
  snapshots is the `uid`, which tests allocate with `new_uid()` too.
- Commits are kernel style, with the design decisions in the message and
  no self-attribution. Commit only `manimx/`; `docs/`, the root
  `pyproject.toml` and the root `uv.lock` belong to the user.
- Tests: `uv run --active pytest` in `manimx/`, with coverage at least 90%.
- Benchmark: check the power mode, run `scripts/benchmark_project.py
  tests/projects/benchmark --check-reference <ref.npz>`, report.
- Don't run `ruff check` without `--no-fix`: the project config fixes files
  in place.
- Describe everything with its name from the Python code.
