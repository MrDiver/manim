# robjects: the render object tree

What a scene shows at one time, as a tree of plain values (`RNode`).
This tree and the `RenderConfig` are everything a renderer is given; it
knows nothing of mobjects, animations or time.

- `mod.rs`: `RNode` (`uid`, `transform` relative to the parent, `alpha`
  for the whole subtree, `Kind`). Only groups have children.
- `values.rs`: the values nodes carry (fills, strokes, meshes,
  materials, textures, ...) and the constants shared with the shaders.

## Notes

- **Nothing changes once made.** The Python classes are `frozen`; a new
  state is a new node. The resolver relies on this: an identical object
  (`Arc::ptr_eq`) is an unchanged subtree.
- **Versions:** values that are costly to turn into GPU records (points,
  fills, strokes, meshes, materials, particles) carry a `version`. Equal
  versions promise equal values; a new value gets a version never handed
  out before (`next_version()`). Reuse a value to keep its version.
- **uids** are the scene objects' own ids, the same in every frame; they
  are never Python object addresses.
- The Python classes are in `python/robjects.rs`.
