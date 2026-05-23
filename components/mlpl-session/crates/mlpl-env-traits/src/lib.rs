//! Capability traits over mlpl-eval's `Environment`. Each
//! trait names one slice of state; downstream crates depend on
//! the slice they need, not on the concrete `Environment`. The
//! struct lives in `mlpl-eval` and impls all of these traits
//! once.
//!
//! Saga 33 step 009 seeded this crate with the single
//! `HasModels` capability. Subsequent steps add more traits as
//! the corresponding sub-crates are extracted (HasDevices when
//! mlpl-models-apply leaves mlpl-eval, etc.); putting them in
//! one foundation crate keeps the trait API in one discoverable
//! place even when the state impls eventually split across many
//! sibling crates.

pub mod models;

pub use models::HasModels;
