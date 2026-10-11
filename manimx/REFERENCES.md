# References

Every algorithm manimx uses that someone else came up with, where it is
in the code, and where to read about it. The code cites the same source
next to the algorithm; this page lists them all in one place.

## Rendering

| Algorithm | Where | Source |
|---|---|---|
| Curved fills that stay exact at any zoom: triangles plus one curve triangle per quadratic, whose fragment test keeps the inside | core/src/geometry/tessellation.rs, shaders/vector.wgsl (`shade`) | Charles Loop and Jim Blinn, [Resolution Independent Curve Rendering using Programmable Graphics Hardware](https://www.microsoft.com/en-us/research/publication/resolution-independent-curve-rendering-using-programmable-graphics-hardware/) (SIGGRAPH 2005) |
| Polygons with holes as triangles (ear clipping) | core/src/geometry/tessellation.rs | [earcut](https://github.com/mapbox/earcut) (Rust port: [earcutr](https://crates.io/crates/earcutr)); David Eberly, [Triangulation by Ear Clipping](https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf) |
| A cubic Bezier as quadratics within a bounded error | core/src/geometry/tessellation.rs (`cubic_to_quadratics`) | [Approximating cubic Bezier curves by quadratic ones](http://caffeineowl.com/graphics/2d/vectorial/cubic2quad01.html) |
| Shadow maps | core/src/render/passes/shadow.rs, shaders/shadows.wgsl | Lance Williams, [Casting Curved Shadows on Curved Surfaces](https://history.siggraph.org/?p=120268) (SIGGRAPH 1978) |
| Percentage-closer filtering (PCF): many depth comparisons averaged | shaders/shadows.wgsl | Reeves, Salesin and Cook, [Rendering Antialiased Shadows with Depth Maps](https://graphics.pixar.com/library/ShadowMaps/) (SIGGRAPH 1987) |
| Percentage-closer soft shadows (PCSS): softer with the distance to the caster | shaders/shadows.wgsl | Randima Fernando, [Percentage-Closer Soft Shadows](https://developer.download.nvidia.com/shaderlibrary/docs/shadow_PCSS.pdf) (NVIDIA 2005) |
| A point light's six shadow maps, the faces of a cube | core/src/packing/shadows.rs, shaders/shadows.wgsl | Philipp Gerasimov, [Omnidirectional Shadow Mapping](https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-12-omnidirectional-shadow-mapping) (GPU Gems, ch. 12) |
| Sample points on a disk (sunflower / golden-angle spiral) | shaders/shadows.wgsl, shaders/depth_of_field.wgsl | H. Vogel, [A better way to construct the sunflower head](https://doi.org/10.1016/0025-5564(79)90080-4) (1979) |
| White noise from a hash (shadow taps, dithering) | shaders/shadows.wgsl, shaders/composite.wgsl | Jarzynski and Olano, [Hash Functions for GPU Rendering](https://jcgt.org/published/0009/03/02/) (JCGT 2020), the PCG hash |
| Deferred lighting with MSAA: edge pixels per sample, the rest once | shaders/deferred_lighting.wgsl, core/src/render/passes/lighting.rs | Andrew Lauritzen, [Deferred Rendering for Current and Future Rendering Pipelines](https://www.intel.com/content/www/us/en/developer/articles/technical/deferred-rendering-for-current-and-future-rendering-pipelines.html) (SIGGRAPH 2010) |
| Physically based materials: Lambert diffuse, GGX, Smith visibility, Schlick Fresnel | shaders/lighting.wgsl, shaders/material.wgsl | [glTF 2.0, appendix B](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation) |
| Light kinds and falloff | shaders/lighting.wgsl | [KHR_lights_punctual](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_lights_punctual) |
| Normal map tangents from UVs | manimx/robjects/mesh.py (`_tangents`) | Eric Lengyel, [Computing Tangent Space Basis Vectors for an Arbitrary Mesh](https://terathon.com/code/tangent.html) |
| Mip maps of textures | manimx/renderer/core_callbacks.py (`_mip_chain`) | Lance Williams, Pyramidal Parametrics (SIGGRAPH 1983), see [Mipmap](https://en.wikipedia.org/wiki/Mipmap) |
| Compositing premultiplied layers ("over") | shaders/composite.wgsl | Porter and Duff, [Compositing Digital Images](https://doi.org/10.1145/800031.808606) (SIGGRAPH 1984) |
| sRGB encoding | shaders/color.wgsl | [sRGB transfer function](https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22)) (IEC 61966-2-1) |
| Tone mapping: Khronos PBR Neutral | shaders/color.wgsl | [KhronosGroup/ToneMapping](https://github.com/KhronosGroup/ToneMapping) |
| Tone mapping: ACES fit | shaders/color.wgsl | Krzysztof Narkowicz, [ACES Filmic Tone Mapping Curve](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/) |
| Tone mapping: AgX | shaders/color.wgsl | Benjamin Wrensch, [Minimal AgX Implementation](https://iolite-engine.com/blog_posts/minimal_agx_implementation) |
| Tone mapping: Reinhard | shaders/color.wgsl | Reinhard et al., [Photographic Tone Reproduction for Digital Images](https://doi.org/10.1145/566654.566575) (SIGGRAPH 2002) |
| Bloom: 13-tap downsample and tent upsample over a mip chain | shaders/bloom.wgsl, core/src/render/effects/bloom.rs | Jorge Jimenez, [Next Generation Post Processing in Call of Duty: Advanced Warfare](https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/) (SIGGRAPH 2014) |
| Depth of field: the thin lens's circle of confusion | shaders/lens.wgsl, shaders/depth_of_field.wgsl | Potmesil and Chakravarty, A Lens and Aperture Camera Model for Synthetic Image Generation (SIGGRAPH 1981); derived in [PBR book, 6.2.3](https://pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models#TheThinLensModelandDepthofField) |
| Depth of field: scatter as gather, tiles of largest blur, interleaved gradient noise | shaders/depth_of_field*.wgsl, core/src/render/effects/depth_of_field.rs | Jorge Jimenez, [Next Generation Post Processing in Call of Duty: Advanced Warfare](https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/) (SIGGRAPH 2014) |
| Motion blur by averaging renders inside the shutter | manimx/renderer/effects.py (`MotionBlur`), core/src/render/composer.rs | Haeberli and Akeley, [The Accumulation Buffer](https://doi.org/10.1145/97880.97913) (SIGGRAPH 1990) |

## Geometry and animation

| Algorithm | Where | Source |
|---|---|---|
| Cutting a Bezier curve at a parameter | manimx/animation/signals/paths.py, core/src/evaluation/paths.rs | the de Casteljau algorithm, [A Primer on Bezier Curves](https://pomax.github.io/bezierinfo/#splitting) |
| Circular arcs as cubic Beziers (handles at `4/3 tan(theta/4)`) | manimx/userland/geometry.py (`arc_points`) | [Approximate a circle with cubic Bezier curves](https://spencermortensen.com/articles/bezier-circle/) |
| SVG elliptical arcs to center form | manimx/userland/svg.py (`_arc`) | [SVG 1.1, appendix F.6](https://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes) |
| Rotations interpolated along the shortest arc (slerp) | manimx/animation/signals/kinds.py (`ROTATION`) | Ken Shoemake, [Animating Rotation with Quaternion Curves](https://doi.org/10.1145/325165.325242) (SIGGRAPH 1985) |
| Monotone time maps (retiming without running backwards) | manimx/animation/time_map.py | Fritsch and Carlson, [Monotone Piecewise Cubic Interpolation](https://doi.org/10.1137/0717021) (1980) |
| Tick steps of 1 and 5 times powers of ten | manimx/userland/graphs.py | Paul Heckbert, [Nice Numbers for Graph Labels](https://www.ri.cmu.edu/publications/nice-numbers-for-graph-labels/) (Graphics Gems 1990) |
| Rate functions | manimx/userland/rate_functions.py | [easings.net](https://easings.net/) and Manim's rate functions |

## Ideas of manimx's own

These have no outside source; the code and the renderer README explain
them: the painter's order of 2D shapes as a depth bias per layer
(shaders/vector.wgsl), 2D colors kept exact beside tone-mapped 3D
(shaders/composite.wgsl), clip masks as alpha layers with lights scoped
to them (core/src/render/renderer.rs), soft-shadow taps in the
receiving surface's plane (shaders/shadows.wgsl), retained vector
records patched per frame (core/src/packing/vectors.rs), and signals
evaluated at any time from tracks of animations (manimx/animation).
