# flat: a scene in world space

A `FlatScene` holds the drawable leaves of a resolved tree in painter's
order, each with its world matrix, alpha and clip scope, plus the masks
of its clip scopes and its lights. Nothing in it is relative to anything
else: no groups, no parents, only the nesting of clip scopes.

A renderer is never given a tree or a whole scene twice, only `Update`s:
the changes of one scene since the update before (`Change::Reset`,
`Change::Set`, `Change::Append`). It keeps a scene until told to forget
it.

## Notes

- This folder depends on `robjects` alone.
- `NO_CLIP` (-1, no clip scope) is repeated in `render/packet/` on
  purpose, so the drawer does not depend on this folder; change both
  together.
