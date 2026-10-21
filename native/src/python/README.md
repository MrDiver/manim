# python: the classes of manim._native

- `robjects.rs`: the render object classes (`RNode`, `RGroup`,
  `RClippingGroup`, `RBezierPath`, `RShape2D`, `RCurve3D`, `RShape3D`,
  `RParticles`, the lights, and their values `Fill`, `Stroke`,
  `MeshData`, `Texture`, `SurfaceShader`, `PbrMaterial`), all `frozen`.
- `renderer.rs`: `WgpuRenderer`, the `Renderer` of
  `manim/renderer/protocol.py`: `render(root, config)` returns an
  `(H, W, 4)` uint8 array.

## Notes

- **Reading `RenderConfig`:** fields fixed per pipeline (size, samples,
  shadow resolution, GPU, GPU memory) make a new pipeline when they
  change, and every scene is sent whole again; everything else is read
  per call. `tests/module/test_wgpu_renderer.py` checks that every field
  is read, so a new field needs code here.
- **Errors:** `Error::Unsupported` becomes `CapabilityException`,
  `Error::Invalid` `ValueError`, GPU and shader errors `RuntimeError`.
- **Frame hold:** a frame that shows what the last one showed returns
  the last pixels without rendering (`FrameHold`).
