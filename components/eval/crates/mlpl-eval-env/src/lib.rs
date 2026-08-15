//! The evaluator's `Environment` -- the session-state hub -- and its
//! inherent-impl extension modules, moved below the `mlpl-eval`
//! spine (eval decomposition, env-base-out; design in
//! docs/eval-env-design.md). TRANSIENT: this crate is over the
//! module budget until the capability peels convert the env_*
//! inherent impls into per-capability trait crates.
//!
//! `dispatch_hook` breaks the one upward call (device dispatch):
//! the hub installs its `dispatched_call` at eval entry.

pub mod dispatch_hook;
pub mod env;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
pub mod env_gpu;
pub mod trait_impls_data;
pub mod trait_impls_exec;
pub mod trait_impls_params;

pub use dispatch_hook::install_dispatch;
pub use env::{Environment, PORT_EXTENSION_ID, PORT_TYPE_ID, PortEndpoints};
