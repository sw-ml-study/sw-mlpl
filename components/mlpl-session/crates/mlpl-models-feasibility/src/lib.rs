//! Feasibility toolkit layered on top of `estimate_train`:
//! - `calibrate_device([size])`: benchmark matmul on the
//!   active device, cache observed GFLOPS.
//! - `estimate_hypothetical(name, ...)`: answer "how big would
//!   a SmolLM / Llama / Qwen fine-tune be on my laptop?"
//!   without materializing weights.
//! - `feasible(est, budget) -> 0/1`: gate-pattern guard.
//!
//! Saga 33 step 015 -- the first sub-crate that uses
//! `HasDispatch` (for calibrate's matmul benchmark) in
//! addition to the existing state traits.

pub mod calibrate;
pub mod error;
pub mod feasible;
pub mod hypothetical;
pub mod hypothetical_specs;

pub use calibrate::calibrate_device_inner;
pub use error::FeasibilityError;
pub use feasible::feasible_inner;
pub use hypothetical::estimate_hypothetical_inner;
