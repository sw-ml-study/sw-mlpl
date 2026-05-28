//! Saga 33 step 003: per-binding `ValueTag` methods extracted
//! from `env.rs`. Saga 23 step 001: producer ops auto-tag their
//! outputs, predicate-checked consumers read tags, and
//! `:describe` / `:tags` enumerate them.

use mlpl_core::ValueTag;

use crate::env::Environment;

impl Environment {
    /// Saga 23 step 001: attach a `ValueTag` to a binding name.
    /// Overwrites any prior tag for the same name.
    pub fn set_tag(&mut self, name: String, tag: ValueTag) {
        self.tags.insert(name, tag);
    }

    /// Saga 23 step 001: look up a binding's tag, if any.
    #[must_use]
    pub fn get_tag(&self, name: &str) -> Option<&ValueTag> {
        self.tags.get(name)
    }

    /// Saga 23 step 001: clear any tag attached to `name`. No-op
    /// when `name` is not currently tagged.
    pub fn clear_tag(&mut self, name: &str) {
        self.tags.remove(name);
    }

    /// Saga 23 step 001: iterate every (name, tag) pair. Used by
    /// `:tags` listing in step 005.
    pub fn tags_iter(&self) -> impl Iterator<Item = (&String, &ValueTag)> {
        self.tags.iter()
    }
}
