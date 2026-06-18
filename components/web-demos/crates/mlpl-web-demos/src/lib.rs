//! Web playground demo registry. The demo content (category / name / intro /
//! takeaway / MLPL `lines`) AND the per-demo metadata (gating tier, literate
//! link, heads-up notes) all live in `demos.toml`; `build.rs` generates the
//! `DEMOS`, `DEMO_CAPABILITIES`, `LITERATE_DOCS`, and `PROGRESS_NOTES` consts
//! from it into `OUT_DIR` (included below), so the prose, demo code, and
//! metadata stay in data and out of Rust source. The shared `Demo` /
//! `Capability` / `ProgressNote` types + the pure `demo_disabled` gate are
//! re-exported from `mlpl-web-demos-types`, and the name-keyed lookups from
//! `accessors`, so `mlpl_web_demos::*` call sites stay unchanged.

mod accessors;

pub use accessors::{capability_for, literate_for, progress_notes_for};
pub use mlpl_web_demos_types::{Capability, Demo, Device, ProgressNote, demo_disabled};

// `DEMOS`, `DEMO_CAPABILITIES`, `LITERATE_DOCS`, `PROGRESS_NOTES`, generated
// from demos.toml by build.rs.
include!(concat!(env!("OUT_DIR"), "/demos.rs"));
