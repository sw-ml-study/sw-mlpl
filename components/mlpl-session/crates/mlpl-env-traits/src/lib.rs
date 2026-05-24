//! Capability traits over mlpl-eval's `Environment`. Each
//! trait names one slice of state; downstream crates depend on
//! the slice they need, not on the concrete `Environment`. The
//! struct lives in `mlpl-eval` and impls all of these traits
//! once.
//!
//! Trait inventory (saga 33 step 012):
//! - `HasModels` (models.rs)
//! - `HasParams` (params.rs)
//! - `HasFrozen` (frozen.rs)
//! - `HasVars` (vars.rs)
//! - `HasTensorDevices` + `HasModelIds` (devices.rs)

pub mod devices;
pub mod frozen;
pub mod models;
pub mod params;
pub mod vars;

pub use devices::{HasModelIds, HasTensorDevices};
pub use frozen::HasFrozen;
pub use models::HasModels;
pub use params::HasParams;
pub use vars::HasVars;
