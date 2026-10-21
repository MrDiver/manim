# resolve: trees into changes

A `Resolver` turns each new render object tree of one scene into the
changes since the last one (`Update`, see `flat/`): every drawable leaf
in world space, and only what changed. `Resolvers` keeps one per scene
and drops scenes not synced for a while (`sweep`).

```text
RNode tree ──Resolver::sync──▶ Update ──▶ a renderer
```

## Notes

- **Three kinds of change:** the same structure gives `Change::Set` per
  changed leaf; nodes added at the end of the root give
  `Change::Append`; anything else resolves the tree again from scratch
  (`Change::Reset`). Keep new nodes at the end of the root to stay cheap.
- **Skipping:** a subtree that is the same object as last time under an
  unchanged parent is skipped whole (`Arc::ptr_eq`); this is safe because
  render objects are frozen and the mirror keeps every node alive.
- **Revisions:** every change gives the scene a new revision from one
  counter for all scenes. Equal revisions promise an equal scene; the
  frame hold (`composite/hold.rs`) relies on it.
- **Rules:** a node scaled to nothing is skipped with its subtree; a
  clipping group without a visible mask or content shows nothing; alphas
  multiply down the tree; a light inside a clip's content belongs to that
  clip scope, one inside a mask lights nothing.
- What no renderer can show (a mesh or a clipping group inside a mask)
  is `Error::Unsupported`, which Python raises as `CapabilityException`.
