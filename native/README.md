# native: manim's renderer in Rust

The Rust crate behind `manim._native`. Python hands it a tree of render
objects and a `RenderConfig` per frame and gets the pixels back.

```text
Python scene ──RNode tree──▶ resolve ──Update──▶ composite ──▶ render ──layers──▶ composite ──▶ pixels
              (robjects)              (flat)      (frames)      (pack, draw)       (effects, output)
```

| Folder | What it does |
|---|---|
| [src/robjects](src/robjects/README.md) | The render object tree: plain values, the interface between the scene side and every renderer. |
| [src/resolve](src/resolve/README.md) | Turns trees into scenes in world space, sent as changes. |
| [src/flat](src/flat/README.md) | A scene in world space and its changes (`FlatScene`, `Update`). |
| [src/composite](src/composite/README.md) | Frames: drives the renderer, composites its layers, effects, read-back. |
| [src/render](src/render/README.md) | The wgpu renderer: one scene seen with one view's settings, into layers. |
| [src/gpu](src/gpu/README.md) | The device, shaders, buffers and fullscreen passes every stage uses. |
| [src/cache](src/cache/README.md) | `Memo`, the one shape every cache has. |
| [src/python](src/python/README.md) | The Python classes (`manim._native`). |
| `shaders/` | The WGSL shaders, compiled into the library. |
| [docs](docs/stroke_aa.md) | Notes with pictures: how stroke and fill edges are smoothed. |

## Building

`uv sync` builds the crate with maturin (`pyproject.toml`). While working
on it, a release build copied over the installed library is faster:

```sh
cd native && cargo build --release && cp target/release/lib_native.so ../manim/_native.abi3.so
```

`cargo test --release`, `cargo clippy --release` and `cargo fmt --check`
must stay clean (the dead-code warnings for parts not wired to Python
yet aside). The Python side is tested by `tests/module/test_wgpu_renderer.py`
and the graphical tests (`tests/README.md`).

## Rules

- Each folder knows only the ones before it in the picture above, plus
  `gpu`, `cache` and `error.rs`; they talk through their types (`RNode`,
  `Update`, `Packet`, `Uploads`, `Layers`), never through helpers of
  another folder. `robjects` and `cache` know nothing else; `flat` only
  `robjects`; the renderer never knows `resolve` (outside tests) or
  `composite`.
- Names come from the Python side (`RShape2D`, `RenderConfig`, ...).
- Every folder has a README: what it does, and what to know before
  changing it.
