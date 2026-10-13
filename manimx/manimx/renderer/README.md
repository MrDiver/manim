# manimx.renderer

How manimx turns a scene at one time into an image. A renderer is given
a render object tree (`manimx.robjects`: `RGroup`, `RShape2D`, ...) and a
`RenderConfig`, nothing else. The renderer itself is Rust on
[wgpu](https://wgpu.rs), in the core
([core/src/render/](../../core/src/render/mod.rs), `manimx._core.RenderPipeline`);
this package holds what it is given next to the tree:

- `RenderConfig` and `Camera` (config.py): size, background, camera,
  MSAA, lighting and shadow settings; `ViewState`, a tree with the
  camera looking at it;
- the effect settings (effects.py): motion blur, bloom, tone mapping,
  custom WGSL.

The WGSL shaders are in core/shaders/, compiled into the core.

`CoreFramePipeline` (manimx/drivers/core_pipeline.py) drives the
renderer; `manimx render`, `manimx still`, the render workers, the web UI
and `render_image` all go through it. The renderer keeps no scene state
of its own beyond caches, so frames can be rendered in any order and in
parallel processes.

What it draws (each render object is the snapshot of the scene node of
the same name without the R):

- **`RShape2D`**: vector shapes. Exact curved fills, strokes with
  per-point color, opacity and width, all joint and cap styles, painter's
  order.
- **`RCurve3D`**: stroke-only Bezier curves in 3D, always facing the
  camera, thinner with distance.
- **`RShape3D`** and **`RParticles`**: lit triangle meshes with glTF
  metallic-roughness materials, textures and custom surface shaders.
  glTF/GLB files are imported with `manimx.userland.load_gltf`, which
  builds scene nodes.
- **Lights**: directional, point and spot, with shadows: PCF (a fixed
  narrow soft edge) or PCSS (soft edges that widen with the distance to
  the caster).
- **`RClippingGroup`**: soft alpha masks, optionally one-sided (portals),
  with lights scoped to the group; casters inside one cast only the
  shadow of what their masks let through.
- **Camera views**: another camera's picture on an `RShape2D` (the snapshot of a `CameraView`).
- **Effects**: motion blur, depth of field (a camera's `aperture`),
  bloom, tone mapping and custom WGSL shaders.

---

## Usage

### Render a still

```python
from PIL import Image

from manimx.drivers import render_image
from manimx.renderer import RenderConfig
from manimx.userland import Circle, Square

shapes = [
    Square(2.0, color="#525893", fill_opacity=1.0),
    Circle(1.0, color="#87c2a5", fill_opacity=1.0).move_to((1.5, 0.0, 0.0)),
]
config = RenderConfig(pixel_width=1920, pixel_height=1080, background_color="#ece6e2")
image = render_image(shapes, config=config)  # (H, W, 4) uint8 RGBA, straight alpha
Image.fromarray(image).save("frame.png")

image = render_image(MyScene, time=2.5, config=config)  # a scene at a time
```

`render_image` takes a `Scene` class or instance, a prepared scene, a
timeline, or nodes (seen through `config.camera`). It keeps one
`CoreFramePipeline` per configuration (`StillRenderer`,
manimx/drivers/stills.py), so frames of the same size reuse its GPU
resources. Videos are rendered with `manimx render` (manimx/drivers/offline.py),
which keeps several frames on the GPU at once.

`RenderConfig` holds everything that is not part of the scene: size,
background, camera, MSAA, time, lighting and shadow settings.

```python
import math
from manimx.renderer import Camera, RenderConfig

config = RenderConfig(
    pixel_width=1280, pixel_height=720,
    camera=Camera.look_at((9, -12, 7), (0, 0, 1), field_of_view=math.radians(45)),  # perspective
    msaa_samples=4,
    shadows=True, shadow_resolution=2048, shadow_samples=16, shadow_method="pcf",
    ambient_intensity=0.15, exposure=1.0,
)
```

Without `field_of_view` the camera is orthographic and shows
`frame_height` world units vertically (8, like Manim). World `+z` is up in
3D scenes, as in Blender and Manim's 3D scenes. A scene renders through
its own camera (`Scene.camera`); `config.camera` is for nodes rendered
on their own.

### 3D: meshes, lights, glTF

```python
from manimx.animation import DirectionalLight, PointLight, SpotLight
from manimx.userland import Sphere
from manimx.robjects import PbrMaterial

ball = Sphere(1.0, name="ball", material=PbrMaterial("#FC6255", roughness=0.3)).move_to((0, 0, 1))
sun = DirectionalLight(direction=(-1, 1, -2), intensity=3.0, size=0.03)  # size: softness
lamp = PointLight(position=(-2, -2, 2), color="#58C4DD", intensity=15)
spot = SpotLight(position=(0, -4, 6), target=(0, 0, 0), outer_cone_angle=0.4)
```

Only meshes are lit. 2D shapes and curves keep their exact Manim colors,
but they still cast shadows onto meshes. Shadows can be turned off
globally (`RenderConfig.shadows`), per light (`cast_shadows`) and per
node (`cast_shadows` on any shape). If a scene has no lights at all, a
default key light is used.

`shadow_method` picks the filter: `"pcf"` (projects and previews by
default) blurs every shadow edge by a fixed 1.5 texels; `"pcss"` makes
edges sharp where an object touches the surface and softer with distance,
by the light's `size` (Randima Fernando's
[percentage-closer soft shadows](https://developer.download.nvidia.com/shaderlibrary/docs/shadow_PCSS.pdf)).

### Clipping and portals

```python
from manimx.animation import ClippingGroup

portal = ClippingGroup(square_face, world, one_sided=True)  # mask, content
```

The content is shown multiplied by the mask's alpha. The mask counts its
fill and stroke as they would be drawn, so a 40% opaque mask lets 40%
through. `one_sided=True` opens the mask only where it faces the camera.
Lights inside the content only light that content. See
`manimx/examples/portal_cube.py` for four worlds inside one cube.

### Effects

```python
from manimx.renderer import Bloom, RenderSamples, ShaderEffect, ToneMap, ToneMapOperator

effects = [
    RenderSamples(count=16, shutter=1 / 120),
    Bloom(threshold=1.0, intensity=0.1),
    ToneMap(ToneMapOperator.ACES),
]
image = render_image(MyScene, time=2.0, config=config, effects=effects)
```

`RenderSamples` averages `count` renders of a frame. Each moves the
camera by a fraction of a pixel (the Halton sequence in bases 2 and 3:
the projection is shifted in normalized device coordinates, so meshes,
2D shapes and their shadows all move together) and turns the shadow
taps, so edges and soft shadows come out smooth even from few taps per
render. With a `shutter` above 0 the renders are spread over it, and
moving things blur. Every render is a full render, so the cost of a
frame grows with `count`; projects render once per frame unless asked
(`output.samples`, with `effects.motion_blur` to spread them).

Depth of field is not an effect in the list but the camera's lens: with a
`Camera.aperture` above 0, its view is blurred by each pixel's distance from
`focus_distance` (a thin lens, see core/shaders/depth_of_field.wgsl).

`ShaderEffect` runs your own WGSL on the lit 3D layer (`stage="lit"`,
linear HDR) or on the final frame (`stage="frame"`):

```python
vignette = ShaderEffect("""
    fn effect(uv: vec2<f32>, pixel: vec2<i32>) -> vec4<f32> {
        let color = textureLoad(source, pixel, 0);
        return vec4<f32>(color.rgb * (1.0 - value(0u) * dot(uv - 0.5, uv - 0.5)), color.a);
    }
""", values=(1.5,))
```

### Profiling

```python
from manimx.drivers import CoreFramePipeline

pipeline = CoreFramePipeline((), config, profile=True)
pipeline.render(MyScene().prepare().frame_at, 1.0)
print(pipeline.gpu_timings())  # {"shadows": 0.4, "geometry": 0.2, "lighting": 0.6, ...} ms on the GPU
```

---

## How it works

How a frame is drawn, pass by pass with pictures, how the renderer is
built and why: [core/README.md, The renderer](../../core/README.md#the-renderer).

---

## Known limitations

- **Motion blur and depth of field cost per sub-frame:** 16 motion blur
  samples are 16 full renders, and depth of field runs in each of them.
- **Camera views render the scene once more per view**, at the frame's
  size.
- **Directional shadows use one map** fitted to the scope's bounds; very
  large scenes need cascades.
- **8-bit video encoders smooth the dithering** of 3D gradients away;
  banding can come back in H.264 output.
- **Painter's order between non-coplanar 2D planes:** where two 2D planes
  intersect, their layer bias can flip single pixels along the
  intersection.
- **Translucent meshes** sort per mesh, not per triangle, and write
  depth: where the sort is wrong, the mesh drawn later is hidden behind
  the earlier one instead of seen through it (two crossing translucent
  meshes show one order everywhere). The vector layer is
  composited over the lit layer, so 2D shapes behind a translucent mesh
  show in front of it. Texture alpha does not make a mesh translucent
  (no glTF alpha modes); opaque meshes clip at 50% mask coverage, and
  meshes cannot be masks themselves.
- **Translucent shadows** are hard-edged (2x2 texels, no PCSS penumbra)
  and ignore the clip masks between the light and their casters.
- **glTF import** skips animations, skins, morph targets, cameras, lights,
  sparse accessors, vertex colors, a second UV set, samplers and
  extensions.
- **No image-based lighting:** metals reflect only a uniform ambient term.
- **Requires** the `indirect-first-instance` feature and wgpu-native's
  `multi_draw_indirect`; `timestamp-query` is optional (profiler).

---

## Tests

```bash
uv run pytest                              # all tests
uv run pytest --update-references          # write the reference images again (review them!)
uv run python scripts/render_internals.py  # the pictures of "A frame, step by step" (core/README.md)
```

- `tests/test_render_features.py`: every feature checked through
  rendered pixels (order, strokes, masks, portals, lights, shadows), and
  pictures of each against reference images.
- `tests/test_rendering.py`: scenes, effects and projects against
  reference images; held frames.
- `tests/test_camera_view.py`, `test_depth_of_field.py`,
  `test_surface_shader.py`, `test_3d.py`: one feature each.

Reference images live in `tests/references/`. On a mismatch,
`*.actual.png` and `*.diff.png` are written next to the reference. GPU
tests are skipped when no adapter is available.
