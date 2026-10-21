# composite: frames from layers

The compositor makes frames. `RenderPipeline` (`frames.rs`) drives the
renderer (`render/`), composites the layers it renders, applies the
effects and reads the result back:

```text
for each sub-frame (one, or several averaged, e.g. for motion blur):
    for each view:
        for each of its camera views: render and composite its picture
        layers = the view rendered (camera views showing their pictures)
        lit = lit-stage effects (bloom, custom shaders) on layers.lit
        image = composite(tone_map(lit), layers.vector) over the image
    accumulator += weight * image
frame = frame-stage effects on the accumulator
output = sRGB or NV12, read back (output/)
```

- `frames.rs`: `RenderPipeline`, `PipelineOptions` (fixed per pipeline:
  size, samples, shadow resolution, GPU, memory), `View`, `SubFrame`.
- `hold.rs`: `FrameHold`, whether a frame shows what the last one showed.
- `mod.rs`: `Compositor`, the composite and accumulate passes.
- `effects/`: bloom, depth of field, custom shader effects.
- `output/`: conversion to the pixel format and read-back, up to
  `in_flight` frames on the GPU.
- `pictures.rs`: the texture array of the camera views' pictures.

## Notes

- **The pipeline renders every frame it is asked for.** Whether to render
  at all is the caller's decision: `RenderPipeline::frame_key` says what a
  frame shows, `FrameHold` compares it with the last rendered one. Read
  the key before `render` and again after it (a scene sent for the first
  time has a key only once it is kept).
- **Per frame, not per pipeline:** the background (`render`) and each
  view's tone map (`View::tone_map`) can change every frame without a new
  pipeline; both are part of the frame key.
- The renderer is never told about frames, sub-frames or effects.
