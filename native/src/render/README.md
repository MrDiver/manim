# render: the wgpu renderer

Scenes in world space (their `Update`s, `flat/`) and the settings of a
view in, the view's layers out. It knows nothing of mobjects, trees,
frames or Python.

```text
Update ──apply──▶ Packer ──Packet + Uploads──▶ Drawer ──▶ Layers
```

- `renderer.rs`: `Renderer`, as the compositor drives it (`apply`,
  `render`, `forget`).
- `settings.rs`: `ViewSettings`, the `RenderConfig` values read per
  render, and the camera's `Frame` uniform.
- `pack/`: keeps what it needs of each scene and turns it into GPU
  records (tessellation, stroke records, mesh records, draw lists).
- `packet/`: the `Packet` the drawer is given, and the GPU records field
  for field as the WGSL structs declare them (`records.rs`).
- `draw/`: the GPU side: geometry (pipelines, buffers), passes and
  targets. `draw/uploads.rs` defines the record buffers it uploads.

## Notes

- **Layers:** the lit layer (meshes, linear HDR) and the vector layer
  (shapes and curves, premultiplied), each with its depth.
- **Vector draws:** an opaque list (no blending) and a transparent list,
  sorted back to front by each object's center, blended. Both write
  depth and test "less or equal". Every stroke is transparent, because
  its edges fade by coverage.
- **Edges:** fills are smoothed by MSAA only; strokes, joints and caps
  fade by coverage at any sample count. What this means for
  self-crossing translucent strokes, with pictures:
  [docs/stroke_aa.md](../../docs/stroke_aa.md).
- **Crossing translucent shapes** in 3D can be sorted wrongly by their
  centers; the one drawn first then hides the other through depth
  (`ConvexHull3D` shows it). This is chosen: without depth writes,
  translucent meshes (drawn after the vectors) would cover translucent
  shapes in front of them instead.
- **Pixels depend on lyon's patch releases:** lyon_tessellation and
  lyon_geom triangulate slightly differently between patch versions
  (up to 4/255 seen in gradient fills), so the graphical references
  depend on `native/Cargo.lock`.
- The drawer keeps nothing but GPU buffers, pipelines and textures; how
  records came about (scenes, caches, culling) is the packer's.
