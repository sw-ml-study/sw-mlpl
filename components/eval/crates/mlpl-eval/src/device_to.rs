//! The `to_device(x, target)` builtin (Saga 14 step 005; CUDA target
//! added in cuda-foundation step 004). Extracted from `device.rs` to
//! keep that module within budget.
//!
//! Records the target on the environment's per-tensor device map when
//! `x` is a variable reference, so subsequent `apply(model, ...)`
//! calls see the placement. When `x` evaluates to a bare array
//! literal the move is purely metadata (no named binding to stamp);
//! values stay CPU-resident until later gradient/optimizer phases.
//! Valid targets are `cpu`, `mlx`, and `cuda`; anything else errors.

use crate::env_api::*;
use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::device::device_available;
use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

/// Evaluate `to_device(value, target)` / `to_device(target, value)`.
///
/// # Errors
/// `BadArity` if not 2 args; `Unsupported` if neither arg is a string
/// literal target, or the target is not `cpu`/`mlx`/`cuda`; propagates
/// peer-fetch and evaluation errors.
pub(crate) fn eval_to_device(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "to_device".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let (target, value_expr) = match (&args[0], &args[1]) {
        (Expr::StrLit(s, _), value) => (s.clone(), value),
        (value, Expr::StrLit(s, _)) => (s.clone(), value),
        _ => {
            return Err(EvalError::Unsupported(
                "to_device: one argument must be a string literal target".into(),
            ));
        }
    };
    if target != "cpu" && target != "mlx" && target != "cuda" {
        return Err(EvalError::Unsupported(format!(
            "to_device: unknown target '{target}' (expected 'cpu', 'mlx', or 'cuda')"
        )));
    }
    if target != "cpu" && !device_available(&target) && env.take_device_fallback_warning() {
        eprintln!(
            "warning: to_device(..., \"{target}\") requested but the \
             {target} feature is not compiled in; placement recorded \
             but values stay CPU-resident."
        );
    }
    if target == "cpu" {
        return to_cpu(value_expr, env, trace);
    }
    // A model has no single array value -- stamping its params yields a
    // scalar-zero placeholder so `to_device(model, "cuda")` works in
    // statement position. Otherwise stamp the binding and pass the
    // value through.
    if let Some(model_result) = stamp_target(&target, value_expr, env) {
        return Ok(model_result);
    }
    crate::eval::eval_expr(value_expr, env, trace)
        .and_then(Value::into_array)
        .map(Value::Array)
}

/// Move a value back to the CPU: fetch a remote `DeviceTensor`'s
/// bytes through the peer dispatcher, then clear the device tag.
fn to_cpu(
    value_expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let value = crate::eval::eval_expr(value_expr, env, trace)?;
    if let Value::DeviceTensor { peer, handle, .. } = value {
        let arr = env
            .peer_dispatcher()
            .ok_or_else(|| EvalError::Unsupported("to_device: no peer dispatcher".into()))?
            .fetch_tensor(&peer, &handle)?;
        if let Expr::Ident(name, _) = value_expr {
            env.set(name.clone(), arr.clone());
            env.remove_device_tensor(name);
            env.set_tensor_device(name.clone(), "cpu".into());
        }
        return Ok(Value::Array(arr));
    }
    if let Expr::Ident(name, _) = value_expr {
        env.set_tensor_device(name.clone(), "cpu".into());
    }
    value.into_array().map(Value::Array)
}

/// Stamp `target` onto an identifier binding so downstream `apply`
/// sees it. When the name resolves to a model, stamp every param and
/// return `Some(scalar zero)` (a model has no single array value);
/// otherwise stamp the binding and return `None`.
fn stamp_target(target: &str, value_expr: &Expr, env: &mut Environment) -> Option<Value> {
    let Expr::Ident(name, _) = value_expr else {
        return None;
    };
    let params = env
        .get_model(name)
        .map(mlpl_eval_core::model::ModelSpec::params);
    if let Some(params) = params {
        for p in params {
            env.set_tensor_device(p, target.to_string());
        }
        return Some(Value::Array(DenseArray::from_scalar(0.0)));
    }
    env.set_tensor_device(name.clone(), target.to_string());
    None
}
