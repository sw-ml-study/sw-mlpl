//! Capability traits over mlpl-eval's `Environment`. Each
//! trait names one slice of state OR one behavior; downstream
//! crates depend on the trait they need, not on the concrete
//! `Environment`. The struct lives in `mlpl-eval` and impls
//! all of these traits once.
//!
//! Trait inventory (saga 33 step 015):
//! State: `HasModels`, `HasParams` + `HasFrozen` (in
//! params.rs), `HasVars`, `HasTensorDevices` + `HasModelIds`
//! (in devices.rs), `HasStrings`
//! Behavior: `HasDispatch` (device-aware primitive op
//! dispatch).
//!
//! `params.rs` bundles `HasParams` + `HasFrozen`; `devices.rs`
//! bundles `HasTensorDevices` + `HasModelIds`. Bundling keeps
//! the crate at 7 modules (PASS Crate-Module-Count at max).

pub mod devices;
pub mod dispatch;
pub mod models;
pub mod params;
pub mod strings;
pub mod vars;

pub use devices::{HasModelIds, HasTensorDevices};
pub use dispatch::{DispatchError, HasDispatch};
pub use models::HasModels;
pub use params::{HasFrozen, HasParams};
pub use strings::HasStrings;
pub use vars::HasVars;
