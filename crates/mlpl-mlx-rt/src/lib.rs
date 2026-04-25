//! MLX-backed runtime target for MLPL (Saga 14, Apple Silicon).
//!
//! Sibling to `mlpl-rt`. Exposes the same primitive surface --
//! `matmul`, `add`/`sub`/`mul`/`div`/`neg`, `exp`/`log`/`tanh`/
//! `sigmoid`/`relu`, `reshape`/`transpose` -- but dispatches
//! through Apple's MLX library via `mlx-rs`. A program compiled
//! against `mlpl-rt::<op>` must be swappable for `mlpl_mlx_rt::<op>`
//! without source changes; outputs agree bit-for-bit where the
//! two runtimes share a dtype, and within a documented fp32
//! tolerance where MLX downcasts to f32 for GPU dispatch.
//!
//! MLX-backed code lives behind three conditions at once:
//!
//! 1. the `mlx` Cargo feature (opt-in, off by default),
//! 2. `target_os = "macos"`,
//! 3. `target_arch = "aarch64"`.
//!
//! On any host that fails any of those, the crate still builds
//! but exports no primitives -- the CPU path in `mlpl-rt` remains
//! authoritative. The Saga 14 plan treats cross-platform CI as an
//! invariant, not a per-step concern; this gating is how we honor
//! it.
//!
//! Phase 1 grows the primitive surface across three steps:
//! step 001 introduced `matmul`; step 002 added the elementwise,
//! unary-activation, and shape ops listed above; step 003 closes
//! the forward-pass gap with reductions, softmax/log_softmax, and
//! cross_entropy. Each primitive lives in its own module so the
//! crate stays under the sw-checklist function-count budget as
//! the surface expands.

pub use mlpl_array::{ArrayError, DenseArray, Shape};
pub use mlpl_core::LabeledShape;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod activations;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod common;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod elementwise;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod matmul;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod reductions;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod shapes;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use activations::{exp, log, relu, sigmoid, tanh};
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use elementwise::{add, div, mul, neg, sub};
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use matmul::matmul;
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use reductions::{argmax, cross_entropy, log_softmax, mean, reduce_mul, softmax};
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use shapes::{reshape, transpose};
