# showcase

One short scene per manimx feature, played one after another on the
timeline. Render it with `uv run manimx render`, scrub it with
`uv run manimx ui`, or render one scene with `uv run manimx render Text`.

| Scene | Shows |
|---|---|
| `Shapes` | the 2D shapes, gradients along a path, tapering strokes, Create, DrawBorderThenFill, GrowFromCenter, FadeIn/FadeOut with a shift, Indicate, Rotate, ShrinkToCenter, Uncreate |
| `Signals` | `node.animate` chains (one end state), axes animating independently, `follow()` and `derive()` instead of updaters |
| `Text` | `Typst` and `TypstMath` (one Shape2D per glyph), `Write`, `Morph` between formulas, `SVG` files |
| `Morphing` | `Morph` between shapes and between groups of different sizes |
| `Clipping` | `ClippingGroup`: a moving spotlight mask and text as a mask |
| `Solids` | `Box`, `Sphere`, `Torus`, `ParametricSurface`, `ParametricCurve3D`, `Particles`; metallic-roughness materials, directional, point and spot lights with soft shadows, an emissive lamp with bloom |
| `Cameras` | moving the scene camera, a `CameraView` of a second camera in a nested mask, depth of field and focus pulls |
| `Shaders` | `SurfaceShader`: scrolling stripes, a glowing rim, a checker floor, switching shaders |
| `Physics` | a `Simulation` baked into keyframes, with panned sounds at every bounce |
| `LowerThird` | a clip on layer 1, drawn over the `Solids` clip (its start in manimx.json is the start of `Solids`) |

Colors are the project's own (`scenes/style.py`): manimx's classes take
colors but never pick them.
