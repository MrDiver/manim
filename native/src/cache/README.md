# cache: Memo

`Memo` is the one shape every cache of the renderer has: a key (a node's
uid, a texture's uid, a pipeline variant), a tag saying what it was built
from (a value's version, or several versions together), and what was
built. When the tag still matches, the build is skipped; when it does
not, the entry is built again.

## Notes

- A cache only ever saves time: dropping an entry costs a rebuild, never
  a wrong result.
- Whoever owns a memo decides what to drop (`retain`); the packer drops
  the entries of nodes in no scene it still keeps.
- Records that mirror GPU buffers are not caches (dropping them would
  leave the buffers wrong) and do not live in a `Memo`.
