//! Device op-dispatch: the single place where `mlpl-eval` decides
//! whether a named op routes through a GPU backend (`mlpl-mlx-rt` on
//! Apple, the CUDA candle crates on Linux) or the CPU path.
//!
//! Extracted from `device.rs` (Saga cuda-foundation step 004) to
//! keep that module within the file-size budget. MLX and CUDA expose
//! an identical primitive surface, so both dispatchers are generated
//! from one `op_dispatch!` definition (loose-coupling: define once).
//! Each backend is triple-gated (its Cargo feature + target OS +
//! arch); off-target the dispatcher is a stub returning `None` and
//! the caller falls back to CPU.

use crate::env_api::*;
use mlpl_array::{ArrayError, DenseArray};
use mlpl_array_ops_element::prelude::*;

use crate::env::Environment;
use mlpl_eval_types::EvalError;

/// Map a named op + args onto `$backend`'s primitive surface and
/// return `Some(result)` for a known op, `None` to fall back to CPU.
#[allow(unused_macros)] // unused on CPU-only builds (neither GPU backend gated in).
macro_rules! op_dispatch {
    ($backend:path, $name:expr, $args:expr) => {{
        use $backend as b;
        let args = $args;
        Some(match ($name, args.len()) {
            ("matmul", 2) => b::matmul(&args[0], &args[1]),
            ("add", 2) => b::add(&args[0], &args[1]),
            ("sub", 2) => b::sub(&args[0], &args[1]),
            ("mul", 2) => b::mul(&args[0], &args[1]),
            ("div", 2) => b::div(&args[0], &args[1]),
            ("neg", 1) => Ok(b::neg(&args[0])),
            ("exp", 1) => Ok(b::exp(&args[0])),
            ("log", 1) => Ok(b::log(&args[0])),
            ("relu", 1) => Ok(b::relu(&args[0])),
            ("sigmoid", 1) => Ok(b::sigmoid(&args[0])),
            ("tanh" | "tanh_fn", 1) => Ok(b::tanh(&args[0])),
            ("transpose", 1) => Ok(b::transpose(&args[0])),
            ("reshape", 2) => {
                let dims: Vec<usize> = args[1].data().iter().map(|&d| d as usize).collect();
                b::reshape(&args[0], &dims)
            }
            ("softmax", 2) => b::softmax(&args[0], args[1].data()[0] as usize),
            ("log_softmax", 2) => b::log_softmax(&args[0], args[1].data()[0] as usize),
            ("cross_entropy", 2) => b::cross_entropy(&args[0], &args[1]),
            ("reduce_mul", 1) => b::reduce_mul(&args[0], None),
            ("reduce_mul", 2) => b::reduce_mul(&args[0], Some(args[1].data()[0] as usize)),
            ("mean", 1) => b::mean(&args[0], None),
            ("mean", 2) => b::mean(&args[0], Some(args[1].data()[0] as usize)),
            ("argmax", 1) => b::argmax(&args[0], None),
            ("argmax", 2) => b::argmax(&args[0], Some(args[1].data()[0] as usize)),
            _ => return None,
        })
    }};
}

/// Dispatch a named op through `mlpl-mlx-rt` (Apple build) or CPU.
/// `Some(_)` = handled by MLX; `None` = run on CPU.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // dims/axis from validated f64 array values.
pub(crate) fn try_mlx_dispatch(
    name: &str,
    args: &[DenseArray],
) -> Option<Result<DenseArray, ArrayError>> {
    op_dispatch!(mlpl_mlx_rt, name, args)
}

/// Stub for builds without MLX support: always CPU fallback.
#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn try_mlx_dispatch(
    _name: &str,
    _args: &[DenseArray],
) -> Option<Result<DenseArray, ArrayError>> {
    None
}

/// The CUDA op surface unified under one path so `op_dispatch!` can
/// drive it the same way as `mlpl_mlx_rt` (the ops live across three
/// sibling crates).
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod cuda_ops {
    pub use mlpl_cuda_elementwise::{add, div, exp, log, mul, neg, relu, sigmoid, sub, tanh};
    pub use mlpl_cuda_nn::{argmax, cross_entropy, log_softmax, mean, reduce_mul, softmax};
    pub use mlpl_cuda_rt::{matmul, reshape, transpose};
}

/// Dispatch a named op through the CUDA runtime (Linux build) or CPU.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // dims/axis from validated f64 array values.
pub(crate) fn try_cuda_dispatch(
    name: &str,
    args: &[DenseArray],
) -> Option<Result<DenseArray, ArrayError>> {
    op_dispatch!(cuda_ops, name, args)
}

/// Stub for builds without CUDA support: always CPU fallback.
#[cfg(not(all(feature = "cuda", target_os = "linux", target_arch = "x86_64")))]
pub(crate) fn try_cuda_dispatch(
    _name: &str,
    _args: &[DenseArray],
) -> Option<Result<DenseArray, ArrayError>> {
    None
}

/// Map a backend `ArrayError` to the eval-layer error so the caller
/// can route it through the same `?` chain the CPU path uses.
pub(crate) fn lift_array_error(err: ArrayError) -> EvalError {
    EvalError::from(err)
}

/// Run a named builtin with `env`'s active device in mind. When the
/// active device is a GPU backend and the op is in its surface, route
/// there; otherwise fall back to the CPU path (elementwise arithmetic
/// lowers to `DenseArray` ops, everything else to
/// `mlpl_runtime::call_builtin`). The device decision lives here so
/// `eval_binop`, `eval_fncall`, and every Model DSL forward helper
/// share it.
///
/// # Errors
/// Surfaces backend validation errors and `mlpl_runtime` errors.
pub(crate) fn dispatched_call(
    env: &Environment,
    name: &str,
    args: Vec<DenseArray>,
) -> Result<DenseArray, EvalError> {
    let device = env.device();
    if device == "mlx"
        && let Some(result) = try_mlx_dispatch(name, &args)
    {
        return result.map_err(lift_array_error);
    }
    if device == "cuda"
        && let Some(result) = try_cuda_dispatch(name, &args)
    {
        return result.map_err(lift_array_error);
    }
    match (name, args.len()) {
        ("add", 2) => Ok(args[0].apply_binop(&args[1], |a, b| a + b)?),
        ("sub", 2) => Ok(args[0].apply_binop(&args[1], |a, b| a - b)?),
        ("mul", 2) => Ok(args[0].apply_binop(&args[1], |a, b| a * b)?),
        ("div", 2) => Ok(args[0].apply_binop(&args[1], |a, b| a / b)?),
        ("neg", 1) => Ok(args[0].map(|v| -v)),
        // Model DSL activation names; `mlpl_runtime::call_builtin`
        // does not own `tanh`/`relu` directly, so the CPU fallback
        // lowers them to `DenseArray::map`.
        ("tanh", 1) => Ok(args[0].map(f64::tanh)),
        ("relu", 1) => Ok(args[0].map(|v| if v > 0.0 { v } else { 0.0 })),
        _ => Ok(mlpl_runtime::call_builtin(name, args)?),
    }
}
