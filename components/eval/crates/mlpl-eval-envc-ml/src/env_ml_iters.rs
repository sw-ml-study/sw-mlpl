//! Whole-map iterator accessors for the ML-state capabilities,
//! split from `EnvParams`/`EnvTags` so each module stays inside the
//! function-count budget (trait declarations count as functions).

use mlpl_array::DenseArray;
use mlpl_core::ValueTag;
use mlpl_eval_env::Environment;

/// Iterate the tracked-parameter bindings and the tag table.
pub trait EnvMlIters {
    /// Iterate over all (name, value) parameter bindings.
    fn params(&self) -> impl Iterator<Item = (&String, &DenseArray)>;
    /// Saga 23 step 001: iterate every (name, tag) pair. Used by
    /// `:tags` listing in step 005.
    fn tags_iter(&self) -> impl Iterator<Item = (&String, &ValueTag)>;
}

impl EnvMlIters for Environment {
    fn params(&self) -> impl Iterator<Item = (&String, &DenseArray)> {
        self.params
            .iter()
            .filter_map(move |n| self.vars.get(n).map(|v| (n, v)))
    }
    fn tags_iter(&self) -> impl Iterator<Item = (&String, &ValueTag)> {
        self.tags.iter()
    }
}
