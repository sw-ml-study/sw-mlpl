//! Saga 33 step 003: per-binding `ValueTag` methods extracted
//! from `env.rs`. Saga 23 step 001: producer ops auto-tag their
//! outputs, predicate-checked consumers read tags, and
//! `:describe` / `:tags` enumerate them.

use mlpl_core::ValueTag;

use mlpl_eval_env::Environment;

/// Per-binding `ValueTag` annotations.
pub trait EnvTags {
    /// Saga 23 step 001: attach a `ValueTag` to a binding name.
    /// Overwrites any prior tag for the same name.
    fn set_tag(&mut self, name: String, tag: ValueTag);
    /// Saga 23 step 001: look up a binding's tag, if any.
    #[must_use]
    fn get_tag(&self, name: &str) -> Option<&ValueTag>;
    /// Saga 23 step 001: clear any tag attached to `name`. No-op
    /// when `name` is not currently tagged.
    fn clear_tag(&mut self, name: &str);
}

impl EnvTags for Environment {
    fn set_tag(&mut self, name: String, tag: ValueTag) {
        self.tags.insert(name, tag);
    }

    fn get_tag(&self, name: &str) -> Option<&ValueTag> {
        self.tags.get(name)
    }

    fn clear_tag(&mut self, name: &str) {
        self.tags.remove(name);
    }
}
