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
pub mod env_builtin_refs;
pub mod env_device;
pub mod env_device_tensors;
pub mod env_dirs;
pub mod env_exp_log;
pub mod env_frozen;
#[cfg(any(
    all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
pub mod env_gpu;
pub mod env_interrupt;
pub mod env_metric_sink;
pub mod env_models;
pub mod env_params;
pub mod env_peer;
pub mod env_tags;
pub mod env_tensor_device;
pub mod env_tokenizers;
pub mod env_trait_impls_devices;
pub mod env_trait_impls_dispatch;
pub mod env_trait_impls_models;
pub mod env_trait_impls_params;
pub mod env_trait_impls_strings;
pub mod env_trait_impls_vars;
pub mod env_user_fns;
pub mod env_user_fns_render;

pub use dispatch_hook::install_dispatch;
pub use env::Environment;
