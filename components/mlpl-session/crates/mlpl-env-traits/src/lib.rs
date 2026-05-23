//! Capability traits over mlpl-eval's `Environment`. Each
//! trait names one slice of state; downstream crates depend on
//! the slice they need, not on the concrete `Environment`. The
//! struct lives in `mlpl-eval` and impls all of these traits
//! once.
//!
//! Saga 33 step 009 seeded with `HasModels`; step 010 adds
//! `HasParams` + `HasFrozen`. Subsequent steps add more traits
//! as the corresponding sub-crates are extracted.

pub mod frozen;
pub mod models;
pub mod params;

pub use frozen::HasFrozen;
pub use models::HasModels;
pub use params::HasParams;
