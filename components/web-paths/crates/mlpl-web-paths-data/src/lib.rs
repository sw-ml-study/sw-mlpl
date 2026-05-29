//! Learning paths: curated ordered walks through the
//! tutorial / demo / diagram / glossary surfaces.
//!
//! Saga 82 split the original `paths.rs` into a types module
//! (`LearningPath` + `Step`), an aggregator module (the `PATHS`
//! const), and per-theme data modules. lib.rs stays a facade.

pub mod aggregator;
pub mod architectures;
pub mod history;
pub mod skills;
pub mod types;
pub mod visual;

pub use aggregator::PATHS;
pub use types::{LearningPath, Step};
