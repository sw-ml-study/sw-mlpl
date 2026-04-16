//! MLX-backed runtime target for MLPL (Saga 14, Apple Silicon).
//!
//! Sibling to `mlpl-rt`. Exposes the same primitive surface
//! (`matmul`, later `add`/`reduce_add`/...) but dispatches through
//! Apple's MLX library via `mlx-rs`. A program compiled against
//! `mlpl-rt::matmul` must be swappable for `mlpl_mlx::matmul`
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

pub use mlpl_array::{ArrayError, DenseArray, Shape};
pub use mlpl_core::LabeledShape;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod mlx_backend;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use mlx_backend::matmul;
