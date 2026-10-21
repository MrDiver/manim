# tests

```sh
uv sync --extra typst      # the Typst tests need the optional typst package
uv run pytest              # all tests, in parallel (-n auto)
```

- `module/`: unit tests per manim module; `module/test_wgpu_renderer.py`
  tests the native renderer directly.
- `test_graphical_units/`: scenes rendered and compared with reference
  frames in `test_graphical_units/control_data/`.
- `test_scene_rendering/`: the CLI, caching and video output.
- `conftest.py`: every test runs in a temporary media directory with a
  fixed random seed (`seed=0`), so scenes that use randomness (such as
  `Graph`'s spring layout) draw the same frames on every run.

## Graphical references

The frames are drawn by the native renderer through the InsanityAdapter.
To draw a test's reference again after an intended change:

```sh
uv run pytest tests/test_graphical_units/test_geometry.py::test_Circle --set_test
```

`--show_diff` shows the difference when a comparison fails. Compare a
new reference with the old one by eye before committing it.

### What the pixels depend on

- **lyon's patch releases:** lyon_tessellation and lyon_geom triangulate
  slightly differently between patch versions (up to 4/255 seen in
  gradient fills). The references are made with the versions in
  `native/Cargo.lock`; update them together.
- **The GPU:** rounding differs between GPUs where a value lies exactly
  halfway (an alpha of 0.5 reads back as 127 on one GPU, 128 on another).
  Tests that read such values allow a difference of one; the comparison
  of reference frames allows one level per channel.

## GPU

The tests run on the GPU that `RenderConfig.gpu` chooses (the dedicated
one by default). To run them on another one, for example when the
dedicated GPU is busy, point Vulkan at its driver:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json uv run pytest
```

Every pytest worker opens one GPU device. The renderer reserves GPU
memory in small blocks by default (`RenderConfig.gpu_memory="compact"`),
so all workers fit on one GPU; "no GPU device: Not enough memory left"
means other programs hold its memory.
