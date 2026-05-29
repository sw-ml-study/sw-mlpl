//! Render-core: the per-render derivation helpers
//! (`callbacks::build_input_callbacks` +
//! `modes::derive_modes` + `modes::compute_modes`) plus the
//! main-panel renderer (`main::render_main` + `MainArgs`).
//!
//! Sits below the shell layer (which calls `render_main`
//! through `render_shell` + its own `render()` entry) and
//! above the types crate (whose structs we populate). DAG:
//! types <- aux <- core <- shell. No cycle.

pub mod callbacks;
pub mod modes;
pub mod panel;
