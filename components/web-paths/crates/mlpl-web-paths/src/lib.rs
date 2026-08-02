//! Learning paths surface for the web playground -- facade.
//!
//! - `view` -- the picker + walker (paths_view in the old layout).
//! - `diagrams` -- the diagrams gallery + the diagram statics
//!   list (diagrams_view in the old layout). Bundled with paths
//!   because the path walker renders diagram steps inline.
//!
//! Path data + types (`LearningPath`, `Step`, `PATHS`) live in
//! the sibling `mlpl-web-paths-data` crate. mlpl-web re-exports
//! that crate under `crate::paths` and exposes `view` /
//! `diagrams` under their original `paths_view` / `diagrams_view`
//! names.

pub mod diagrams;
pub mod view;
