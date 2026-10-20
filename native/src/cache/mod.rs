//! `Memo`: what was built from a value, kept while the value holds.
//!
//! Every cache of the renderer has the same shape: a key (a node's uid, a
//! texture's uid, a pipeline variant), a tag saying what it was built
//! from (a value's version, or several versions together), and what was
//! built. When the tag still matches, the build is skipped; when it does
//! not, the entry is built again. A cache only ever saves time: dropping
//! an entry costs a rebuild, never a wrong result.
//!
//! What may be dropped is decided by whoever owns the memo (`retain`):
//! the packer drops the entries of nodes in no scene it still keeps.
//! Records that mirror GPU buffers are not caches (dropping them would
//! leave the buffers wrong) and do not live in a `Memo`.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;

/// Values built per key, each with the tag it was built from.
pub struct Memo<K, T, V> {
    entries: HashMap<K, (T, V)>,
}

impl<K, T, V> Default for Memo<K, T, V> {
    fn default() -> Self {
        Memo {
            entries: HashMap::new(),
        }
    }
}

impl<K: Eq + Hash, T: PartialEq, V> Memo<K, T, V> {
    /// The value kept for `key` if it was built from `tag`, else what
    /// `build` makes, kept for `key` from now on.
    pub fn get_or_build(&mut self, key: K, tag: T, build: impl FnOnce() -> V) -> &V {
        match self.entries.entry(key) {
            Entry::Occupied(entry) => {
                let entry = entry.into_mut();
                if entry.0 != tag {
                    *entry = (tag, build());
                }
                &entry.1
            }
            Entry::Vacant(entry) => &entry.insert((tag, build())).1,
        }
    }

    /// Keep only the entries whose key `keep` accepts.
    pub fn retain(&mut self, mut keep: impl FnMut(&K) -> bool) {
        self.entries.retain(|key, _| keep(key));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_equal_tag_skips_the_build() {
        let mut memo: Memo<u64, u64, &str> = Memo::default();
        assert_eq!(*memo.get_or_build(1, 7, || "first"), "first");
        assert_eq!(*memo.get_or_build(1, 7, || panic!("built again")), "first");
    }

    #[test]
    fn a_new_tag_builds_again() {
        let mut memo: Memo<u64, Vec<u64>, &str> = Memo::default();
        memo.get_or_build(1, vec![1, 2], || "old");
        assert_eq!(*memo.get_or_build(1, vec![1, 3], || "new"), "new");
        assert_eq!(memo.len(), 1);
    }

    #[test]
    fn retain_drops_the_rest() {
        let mut memo: Memo<u64, (), u64> = Memo::default();
        for key in 0..4 {
            memo.get_or_build(key, (), || key);
        }
        memo.retain(|key| key % 2 == 0);
        assert_eq!(memo.len(), 2);
        assert_eq!(*memo.get_or_build(1, (), || 10), 10); // dropped: built again
    }
}
