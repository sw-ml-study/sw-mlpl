//! Name-keyed lookups over the generated per-demo metadata consts
//! (`DEMO_CAPABILITIES`, `LITERATE_DOCS`, `PROGRESS_NOTES`, included into the
//! crate root from `OUT_DIR/demos.rs`). The consts live in data (demos.toml ->
//! build.rs); these accessors are the only logic over them, kept out of the
//! lib.rs facade.

use crate::{Capability, ProgressNote};

/// The capability tier for `demo_name`, defaulting to
/// [`Capability::CPU_LIVE`] when the demo has no override.
#[must_use]
pub fn capability_for(demo_name: &str) -> Capability {
    crate::DEMO_CAPABILITIES
        .iter()
        .find(|(n, _)| *n == demo_name)
        .map_or(Capability::CPU_LIVE, |(_, c)| *c)
}

/// The companion literate HTML filename for `demo_name`, if any.
#[must_use]
pub fn literate_for(demo_name: &str) -> Option<&'static str> {
    crate::LITERATE_DOCS
        .iter()
        .find(|(n, _)| *n == demo_name)
        .map(|(_, f)| *f)
}

/// Look up progress notes for `(demo_name, line_idx)`. Returns the matching
/// entries (usually 0 or 1) without allocating; callers iterate.
pub fn progress_notes_for(
    demo_name: &str,
    line_idx: usize,
) -> impl Iterator<Item = &'static ProgressNote> {
    crate::PROGRESS_NOTES
        .iter()
        .filter(move |n| n.demo == demo_name && n.line_idx == line_idx)
}
