//! Learning paths: curated ordered walks through the
//! tutorial / demo / diagram / glossary surfaces.
//!
//! The path PROSE lives in `paths.toml`; `build.rs` generates the `PATHS`
//! const from it into `OUT_DIR` (included below), so content stays in data
//! and out of Rust source. `types` holds the `LearningPath` + `Step` defs the
//! generated const references. lib.rs stays a facade.

pub mod types;

pub use types::{LearningPath, Step};

// `pub const PATHS: &[LearningPath]`, generated from paths.toml by build.rs.
include!(concat!(env!("OUT_DIR"), "/paths.rs"));
