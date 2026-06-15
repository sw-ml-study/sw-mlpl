//! Web playground demo registry. The demo content (category / name / intro /
//! takeaway / MLPL `lines`) lives in `demos.toml`; `build.rs` generates the
//! aggregated `DEMOS` const from it into `OUT_DIR` (included below), so the
//! prose + demo code stay in data and out of Rust source. The shared `Demo`
//! type, the capability / progress-note helpers, and the per-demo side tables
//! are re-exported from `mlpl-web-demos-types` so `mlpl_web_demos::*` call
//! sites stay unchanged.

pub use mlpl_web_demos_types::{
    Capability, DEMO_CAPABILITIES, Demo, Device, LITERATE_DOCS, PROGRESS_NOTES, ProgressNote,
    capability_for, demo_disabled, literate_for, progress_notes_for,
};

// `pub const DEMOS: &[Demo]`, generated from demos.toml by build.rs.
include!(concat!(env!("OUT_DIR"), "/demos.rs"));
