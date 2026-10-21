# gpu: what every stage uses on the GPU

- `device.rs`: `Gpu`, the device and queue, the texture formats, the
  MSAA sample counts the GPU supports, and shader compilation with
  `#include "file.wgsl"` lines expanded (`Gpu::shader`).
- `shaders.rs`: the WGSL files of `native/shaders/`, compiled into the
  library (`include_str!`).
- `buffer.rs`: buffers that grow as needed.
- `fullscreen.rs`: fullscreen passes, the texture pool and uniform slots
  the compositor and its effects use.

## Notes

- **One device per process** for each kind of GPU and memory choice
  (`Gpu::shared`): drivers refuse more than a few devices, and pipelines
  and textures are cheaper shared.
- **Memory:** `RenderConfig.gpu_memory` chooses how the allocator
  reserves memory. `"compact"` (default) reserves blocks of 8 to 64 MiB;
  `"fast"` wgpu's 128 to 256 MiB. A process keeps the blocks it reserved
  after its renderers are gone, so with many processes (parallel tests)
  "fast" runs out of memory. The "no GPU device" error names the setting
  to try.
- **Samples:** WebGPU guarantees 1 and 4 MSAA samples; 2, 8 and 16 come
  from the adapter's format features where every multisampled format
  supports them. `msaa_samples=-1` takes the most.
- No timestamp queries: timings belong to profiling tools, not to the
  renderer.
- Shaders are compiled into the library: after changing one, build
  again.
