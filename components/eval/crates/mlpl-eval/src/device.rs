//! `device("target") { body }` evaluator + MLX dispatch helper
//! (Saga 14 step 004).
//!
//! Two responsibilities live here:
//!
//! 1. `eval_device` runs the body of a `device("...")` block with
//!    the named target pushed onto `Environment::device_stack`. On
//!    exit it pops the entry, restoring whatever target was active
//!    in the surrounding scope. Nesting works in either direction
//!    (`experiment { device { ... } }` and the swap), and an inner
//!    `device("cpu")` overrides an outer `device("mlx")`.
//!
//! 2. `try_mlx_dispatch` is the single place where `mlpl-eval`
//!    decides whether an op should route through the `mlpl-mlx-rt`
//!    runtime. It is called from `eval_binop` and `eval_fncall`
//!    only when `Environment::device()` returns `"mlx"`. The
//!    helper returns `Some(result)` if the named op exists in
//!    `mlpl-mlx-rt`, otherwise `None` so the caller can fall back to
//!    the CPU path.
//!
//! Triple-gate: the `mlx` Cargo feature on `mlpl-eval` pulls in
//! `mlpl-mlx-rt` (which itself triple-gates on
//! `target_os = "macos"`, `target_arch = "aarch64"`, and its own
//! `mlx` feature). When any of those is missing the dispatch
//! helper is a stub returning `None`, the `device("mlx") { }`
//! block falls back to CPU, and `eval_device` emits a one-time
//! warning so the user knows their code ran on the wrong device.

use crate::env_api::*;
use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

/// Evaluate a `device("target") { body }` block. Returns the
/// value of the body's last statement -- mirrors `experiment`'s
/// shape so a `device(...)` block is a value-yielding expression
/// like every other scoped form.
pub(crate) fn eval_device(
    target: &str,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if let Some(dispatcher) = env.peer_dispatcher() {
        let source = crate::device_block::block_source(body);
        let bindings = crate::device_block::collect_array_bindings(env, &source);
        if let Some(result) = dispatcher.dispatch_block(target, &source, bindings) {
            return result;
        }
    }
    if !device_available(target) && env.take_device_fallback_warning() {
        eprintln!(
            "warning: device(\"{target}\") block requested but the \
             {target} feature is not compiled in; falling back to CPU."
        );
    }
    env.push_device(target.to_string());
    let mut last = Value::Array(DenseArray::from_scalar(0.0));
    for stmt in body {
        last = crate::eval::eval_expr(stmt, env, trace)?;
    }
    env.pop_device();
    Ok(last)
}

/// Whether `target`'s GPU backend can actually dispatch in this build
/// (`cpu`/anything else always "runs", on CPU). Drives the one-time
/// CPU-fallback warning in `eval_device` / `eval_to_device`.
pub(crate) fn device_available(target: &str) -> bool {
    match target {
        "mlx" => crate::device_block::mlx_available(),
        "cuda" => crate::device_block::cuda_available(),
        _ => true,
    }
}

// The op-dispatch machinery (the `op_dispatch!` macro,
// `try_mlx_dispatch`/`try_cuda_dispatch`, `lift_array_error`, and
// `dispatched_call`) lives in `device_dispatch` -- extracted to keep
// this module within the file-size budget. `dispatched_call` is
// re-exported so existing `crate::device::dispatched_call` call sites
// (Model DSL forward helpers, `eval_binop`/`eval_fncall`) resolve
// unchanged.
pub(crate) use crate::device_dispatch::dispatched_call;

// The `to_device(x, target)` builtin lives in `device_to` (its own
// concern). Re-exported so `crate::device::eval_to_device` call sites
// resolve unchanged.
pub(crate) use crate::device_to::eval_to_device;

// Saga E4 step 003: the tape now runs its FORWARD resident on the
// MLX backend (see mlpl-autograd-tape::resident). This replaced the
// old materialize_tape_on_mlx second-forward pass entirely.

/// Arm a fresh tape for device residency: register the MLX backend
/// (idempotent) and flip the tape's resident flag so forward ops
/// keep intermediates on the GPU. Backward still runs the CPU
/// formulas this step, reading MLX-rounded values through
/// `to_dense`.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn enable_resident_tape(tape: &mlpl_autograd::Tape) {
    mlpl_mlx_handle::register_mlx_device_ops();
    tape.resident
        .set(mlpl_tensor_handle::device_ops().is_some());
}

/// CPU-only stub: without the mlx feature/target the tape stays
/// host-resident and gradients match the all-CPU path exactly.
#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn enable_resident_tape(_tape: &mlpl_autograd::Tape) {}
