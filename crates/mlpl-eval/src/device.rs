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
//!    decides whether an op should route through the `mlpl-mlx`
//!    runtime. It is called from `eval_binop` and `eval_fncall`
//!    only when `Environment::device()` returns `"mlx"`. The
//!    helper returns `Some(result)` if the named op exists in
//!    `mlpl-mlx`, otherwise `None` so the caller can fall back to
//!    the CPU path.
//!
//! Triple-gate: the `mlx` Cargo feature on `mlpl-eval` pulls in
//! `mlpl-mlx` (which itself triple-gates on
//! `target_os = "macos"`, `target_arch = "aarch64"`, and its own
//! `mlx` feature). When any of those is missing the dispatch
//! helper is a stub returning `None`, the `device("mlx") { }`
//! block falls back to CPU, and `eval_device` emits a one-time
//! warning so the user knows their code ran on the wrong device.

use mlpl_array::{ArrayError, DenseArray};
use mlpl_parser::Expr;
use mlpl_trace::{Trace, TraceValue};

use crate::env::Environment;
use crate::error::EvalError;

/// Evaluate a `device("target") { body }` block. Returns the
/// value of the body's last statement -- mirrors `experiment`'s
/// shape so a `device(...)` block is a value-yielding expression
/// like every other scoped form.
pub(crate) fn eval_device(
    target: &str,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    if target == "mlx" && !mlx_available() && env.take_mlx_fallback_warning() {
        eprintln!(
            "warning: device(\"mlx\") block requested but the mlx \
             feature is not compiled in; falling back to CPU."
        );
    }
    env.push_device(target.to_string());
    let mut last = DenseArray::from_scalar(0.0);
    for stmt in body {
        last = crate::eval::eval_expr(stmt, env, trace)?.into_array()?;
    }
    env.pop_device();
    Ok(("device", vec![], last))
}

/// Whether the running build can actually dispatch through MLX.
/// Combines the Cargo feature gate with the Apple-Silicon target
/// gate so the answer is a single bool the rest of the module can
/// branch on.
const fn mlx_available() -> bool {
    cfg!(all(
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))
}

/// Dispatch a named op through `mlpl-mlx` when the running build
/// supports it. Returns `None` to mean "this op is not in the
/// MLX surface, run it on CPU"; `Some(Ok(arr))` for a successful
/// MLX result; `Some(Err(e))` for an MLX-side validation error
/// (caller should surface it the same way the CPU path would).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn try_mlx_dispatch(
    name: &str,
    args: &[DenseArray],
) -> Option<Result<DenseArray, ArrayError>> {
    use mlpl_mlx as mx;
    Some(match (name, args.len()) {
        ("matmul", 2) => mx::matmul(&args[0], &args[1]),
        ("add", 2) => mx::add(&args[0], &args[1]),
        ("sub", 2) => mx::sub(&args[0], &args[1]),
        ("mul", 2) => mx::mul(&args[0], &args[1]),
        ("div", 2) => mx::div(&args[0], &args[1]),
        ("neg", 1) => Ok(mx::neg(&args[0])),
        ("exp", 1) => Ok(mx::exp(&args[0])),
        ("log", 1) => Ok(mx::log(&args[0])),
        ("relu", 1) => Ok(mx::relu(&args[0])),
        ("sigmoid", 1) => Ok(mx::sigmoid(&args[0])),
        ("tanh" | "tanh_fn", 1) => Ok(mx::tanh(&args[0])),
        ("transpose", 1) => Ok(mx::transpose(&args[0])),
        ("reshape", 2) => {
            let dims: Vec<usize> = args[1].data().iter().map(|&d| d as usize).collect();
            mx::reshape(&args[0], &dims)
        }
        ("softmax", 2) => mx::softmax(&args[0], args[1].data()[0] as usize),
        ("log_softmax", 2) => mx::log_softmax(&args[0], args[1].data()[0] as usize),
        ("cross_entropy", 2) => mx::cross_entropy(&args[0], &args[1]),
        ("reduce_mul", 1) => mx::reduce_mul(&args[0], None),
        ("reduce_mul", 2) => mx::reduce_mul(&args[0], Some(args[1].data()[0] as usize)),
        ("mean", 1) => mx::mean(&args[0], None),
        ("mean", 2) => mx::mean(&args[0], Some(args[1].data()[0] as usize)),
        ("argmax", 1) => mx::argmax(&args[0], None),
        ("argmax", 2) => mx::argmax(&args[0], Some(args[1].data()[0] as usize)),
        _ => return None,
    })
}

/// Stub for builds without MLX support. Always returns `None` so
/// every dispatch site falls back to the CPU path.
#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn try_mlx_dispatch(
    _name: &str,
    _args: &[DenseArray],
) -> Option<Result<DenseArray, ArrayError>> {
    None
}

/// Map an `mlpl-mlx` `ArrayError` to the eval-layer error type so
/// the caller can route through the same `?` chain that the CPU
/// path uses. Pulled out so `eval_binop` and `eval_fncall` can
/// share it without each duplicating the conversion.
pub(crate) fn lift_array_error(err: ArrayError) -> EvalError {
    EvalError::from(err)
}

/// Run a named builtin with `env`'s active device in mind
/// (Saga 14 step 005). When `env.device() == "mlx"` and the op is
/// in the `mlpl-mlx` surface, route through `try_mlx_dispatch`.
/// Otherwise fall back to the CPU path: elementwise arithmetic
/// lowers to `DenseArray::apply_binop`/`map`, everything else
/// goes through `mlpl_runtime::call_builtin`. Used by
/// `eval_binop`, `eval_fncall`, and every Model DSL forward
/// helper so the device decision lives in exactly one place.
pub(crate) fn dispatched_call(
    env: &Environment,
    name: &str,
    args: Vec<DenseArray>,
) -> Result<DenseArray, EvalError> {
    if env.device() == "mlx"
        && let Some(result) = try_mlx_dispatch(name, &args)
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

/// `to_device(x, target)` builtin (Saga 14 step 005). Records the
/// target on the environment's per-tensor device map when `x` is a
/// variable reference, so subsequent `apply(model, ...)` calls can
/// see the placement. When `x` evaluates to a bare array literal,
/// the move is purely metadata and there is no named binding to
/// stamp; the tensor still round-trips cleanly because values
/// stay CPU-resident until gradient/optimizer steps ship in
/// later phases. Unknown targets are an error.
pub(crate) fn eval_to_device(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "to_device".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let target = match &args[1] {
        Expr::StrLit(s, _) => s.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "to_device: target must be a string literal".into(),
            ));
        }
    };
    if target != "cpu" && target != "mlx" {
        return Err(EvalError::Unsupported(format!(
            "to_device: unknown target '{target}' (expected 'cpu' or 'mlx')"
        )));
    }
    if target == "mlx" && !mlx_available() && env.take_mlx_fallback_warning() {
        eprintln!(
            "warning: to_device(..., \"mlx\") requested but the mlx \
             feature is not compiled in; placement recorded but \
             values stay CPU-resident."
        );
    }
    // If the argument is an identifier, stamp the device on the
    // binding so models that reference it downstream see the new
    // placement. Also propagate to model params when the name
    // resolves to a model.
    if let Expr::Ident(name, _) = &args[0] {
        let is_model = env.get_model(name).is_some();
        if is_model {
            let params: Vec<String> = env
                .get_model(name)
                .map(crate::model::ModelSpec::params)
                .unwrap_or_default();
            for p in params {
                env.set_tensor_device(p, target.clone());
            }
            // Models don't correspond to a single DenseArray value; return
            // a scalar zero so `to_device(model, "mlx")` can appear in
            // statement position the same way model assignments do.
            return Ok(DenseArray::from_scalar(0.0));
        }
        env.set_tensor_device(name.clone(), target.clone());
    }
    crate::eval::eval_expr(&args[0], env, trace)?.into_array()
}
