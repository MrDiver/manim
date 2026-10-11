# Future ideas

Ideas for manimx that are not built yet, and ideas that were built and
taken out again, with why. The order of the next steps lives in the
working todo list; this file keeps what was thought about, so it is not
thought about from scratch again.

## Rendering

- **Render samples.** Average several renders of a frame with jittered
  cameras and tap patterns even without motion blur, like Blender
  Eevee's samples: smooth soft shadows and antialiasing for stills.
- **Screen-space global illumination.** Colored light bounced off
  nearby surfaces, traced in the depth buffer of the frame (Eevee's
  "ray tracing" works like this, with light probes for what the screen
  does not show).
- **Area lights with ray-marched shadows,** replacing PCSS: march the
  shadow maps like rays towards points on the light.
- **Hardware ray tracing** (ray queries) for shadows and reflections,
  once wgpu exposes it on all platforms.
- **Soft translucent shadows** with PCSS penumbrae, and order-independent
  transparency (weighted blended or per-pixel lists) for crossing
  translucent meshes; 2D shapes depth-sorted with translucent meshes.
- **Volumetric light and god rays.** Ray-march each view ray through a
  medium, looking up the shadow maps; god rays follow from it. Open
  question: the medium as a scene node `Fog(density, color, anisotropy)`
  with signals, or a project setting; a per-light `volumetric` strength.
- **Cascaded shadow maps** for directional lights over large scenes.
- **Camera views at their own resolution** instead of the frame's, and
  an explicit recursion depth (views showing views, e.g. 8 steps).
- **Unlit and forward surface shaders,** and parameters (signals) for
  surface shaders.
- **HDR colors for 2D shapes:** a float vector layer and bloom on it,
  while colors up to 1 stay exactly Manim's.
- **Video encoding that keeps the dithering** of 3D gradients (10 bit,
  or a higher quality), so banding does not come back in H.264.
- **Image-based lighting:** an environment map for reflections and
  ambient light.

## Performance

- **Motion blur sub-frames in one submission,** frames rendered in
  parallel, a frame cache by content across runs.
- **Hardware video encoding** without reading frames back to the CPU.
- **`derive()` functions traced into Rust.** Derived signals are pure,
  so calling a function once with placeholder values records its
  operations into an expression graph that the core evaluates. Tracing
  fails on branches like `if x > 0` and on unknown calls; those fall
  back to Python. Open: automatic with a report of fallbacks, or opt-in.
- **Render object fields read from Python without copies (todo).**
  Reading a field of an `RNode` from Python (`shape.points`,
  `shape.fill`, `group.children`) makes a new Python object every time,
  and copies the arrays (points, particle arrays) into new numpy arrays.
  Rendering is not affected (the core reads the tree in Rust), but every
  Python reader is: tests, `web/labels.py`, a renderer written in
  Python. Fix: read-only numpy views onto the Rust memory, kept alive by
  the render object (numpy arrays with a base object), and each field's
  Python object made once on first read and kept in the render object,
  so `shape.fill is shape.fill` holds again.
- **Everything not strictly Python in Rust:** the node classes and
  signals in the core, Python as a thin authoring layer. Open: what must
  stay Python (`construct()`, `derive()` functions, custom animations
  and signal kinds) and how Python nodes map onto Rust ones.
- **One general caching mechanism** for all nodes instead of a cache
  per stage.

## Animation and authoring

- **Regrouping:** shapes in one group (a grid rotated as a whole) and a
  second, overlapping group rotating some of them. Known to break
  Manim's tree; idea: the renderer takes only resolved world transforms,
  and groups only order drawing.
- **Who owns cameras:** the scene (today, `Scene.camera`) or the
  timeline, possibly several per project.
- **Boolean operations on paths** (`Union`, `Intersection`,
  `Difference`): skia-pathops is a dependency now (`union_points`
  merges Typst bracket pieces with it), so these are small nodes on it.
- **ThreeDAxes that follow their ranges.** `plot_surface` and
  `plot_curve` are placed once, where the axes are when they are made;
  like `Axes.plot`, they could derive their points from the ranges and
  the axes' transform, so zooming a 3D range rescales them.
- **Whisper transcripts** calling `Clip.bind_cue()`, so animations wait
  for spoken words.
- **LaTeX** next to Typst, producing groups of `Shape2D` (so `Write` and
  `Morph` work on it). Typst is enough for now.
- **Diagnostics** in the style of miette for project and scene errors,
  once the common errors are known.

## Web UI and editing

- Move and retime clips (cuts, slow motion, scaling), zoom the
  timeline, edit fps and every project setting in the UI.
- Then editing scenes in the UI: keyframes, whole clips made in the UI,
  positioning with gizmos.
- Live presentation: clicks calling `Timeline.fire()`.
- The core compiled to WebAssembly with WebGPU, so the browser plays
  scenes without a Python server.

## Removed ideas

- **Worlds, views and clip transforms** (`World2D`, `World3D`, `View`,
  `Clip.place()`; added in 3e19f9c2, removed in d21070d7). Every clip
  lived in a world, a `World2D` kept its clips in the z = 0 plane, and a
  `View` showed one world through one camera. Removed when every scene
  got its own camera: a clip renders through its scene's camera, so its
  retiming moves the camera along, and all 3D clips share one coordinate
  system. Overlapping clips are drawn over each other, each with its own
  camera, depth, lights and shadows. Compositions (a view shown as an
  image inside another scene) are what `CameraView` does now.
- **An SVG renderer** (`manimx/renderer/svg.py`): the first scaffold to
  check the object model with, before the WebGPU renderer existed. Not
  kept up once the WebGPU renderer drew everything, then removed. Parity
  with Manim's Cairo output was never a goal: manimx's look and its
  reference images are its own.
- **The Python renderer** (wgpu-py: `SceneFlattener`, `FramePacker`,
  `WGPURenderer`, `FrameComposer`). It rendered trees of render objects
  and was the reference the core's Rust renderer was tested against.
  Every change to rendering had to be written twice, so once the core
  rendered everything it was removed; tests compare with reference
  images instead.
