//! Top-level FnCall dispatch for `eval::eval_expr`.
//!
//! Saga 33 step 023 split the FnCall name-driven block out of
//! `eval::eval_expr` to retire that file's File-LOC FAIL.
//! `try_dispatch` walks the family modules in order
//! (`fncall_models`, `fncall_axes`, `fncall_arrays`) and then
//! falls through to the loader / tools cluster handled inline
//! here. Each family returns `Option<Result<Value, EvalError>>`
//! so the caller chains them with `Option::or_else`.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

pub(crate) fn try_dispatch(
    expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    let Expr::FnCall { name, args, span } = expr else {
        return None;
    };
    crate::fncall_models::try_dispatch(name, args, env, trace)
        .or_else(|| crate::fncall_axes::try_dispatch(name, args, env, trace))
        .or_else(|| crate::fncall_arrays::try_dispatch(name, args, env, trace, span))
        .or_else(|| crate::fncall_engram::try_dispatch(name, args, env, trace))
        .or_else(|| try_loaders(name, args, env, trace))
        .or_else(|| try_tools(name, args, env, trace, span))
}

fn try_loaders(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "load" | "load_preloaded" => Some(eval_load_call(name, args, env)),
        "fetch_dataset" => Some(eval_fetch_dataset(args, env)),
        "reduce" => Some(crate::eval_reduce::eval_reduce(args, env, trace)),
        _ => None,
    }
}

fn eval_load_call(name: &str, args: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let Expr::StrLit(path, _) = &args[0] else {
        return Err(EvalError::Unsupported(format!(
            "{name}: argument must be a string literal"
        )));
    };
    if name == "load" {
        crate::loader::eval_load(env, path)
    } else {
        crate::loader::eval_load_preloaded(path)
    }
}

fn eval_fetch_dataset(args: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "fetch_dataset".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let Expr::StrLit(dataset, _) = &args[0] else {
        return Err(EvalError::Unsupported(
            "fetch_dataset: argument must be a string literal".into(),
        ));
    };
    crate::loader::eval_fetch_dataset(env, dataset)
}

fn try_tools(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    if let Some(r) = crate::tokenizer::dispatch(name, args, env, trace) {
        return Some(r);
    }
    match name {
        "llm_call" => Some(crate::llm_dispatch::dispatch(args, env, trace)),
        "emit_frame" => Some(crate::fncall_trace::eval_emit_frame(args, env, trace)),
        "compare" => Some(crate::experiment_compare::dispatch_compare(args, env)),
        "equal" | "repr" => Some(eval_structural(name, args, env, trace)),
        "call" => Some(eval_call(args, env, trace, span)),
        "map_ok" | "and_then" | "or_else" => Some(crate::fncall_combinators::eval_combinator(
            name, args, env, trace,
        )),
        "experiment_metric" => Some(crate::experiment_compare::eval_experiment_metric(args, env)),
        "momentum_sgd" | "adam" => {
            Some(crate::grad_optim::eval_optim(name, args, env, trace, span))
        }
        _ => {
            crate::eval_ops::eval_analysis_helper(name, args, env, trace).map(|r| r.map(Value::Str))
        }
    }
}

/// `equal(a, b)` / `repr(v)` -- the structural-assertion pair
/// (total equality never hard-errors; bounded deterministic
/// rendering). Cores live in mlpl-value-structural so every
/// surface shares one behavior.
fn eval_structural(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let need = if name == "equal" { 2 } else { 1 };
    if args.len() != need {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: need,
            got: args.len(),
        });
    }
    let a = crate::eval::eval_expr(&args[0], env, trace)?;
    if name == "repr" {
        return Ok(Value::Str(mlpl_value_structural::value_repr(&a)));
    }
    let b = crate::eval::eval_expr(&args[1], env, trace)?;
    let eq = mlpl_value_structural::value_equal(&a, &b);
    Ok(Value::Array(mlpl_array::DenseArray::from_scalar(
        f64::from(u8::from(eq)),
    )))
}

/// `call(f, args...)` -- uniform invocation of a reference value
/// (user `:u:name` or builtin `:name`): the referent is invoked
/// exactly as if written by name, so arity errors identify the
/// REFERENCED function and Ok/Err/? behavior is unchanged.
fn eval_call(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    span: &mlpl_core::Span,
) -> Result<Value, EvalError> {
    let (f_expr, rest) = args.split_first().ok_or_else(|| EvalError::BadArity {
        func: "call".into(),
        expected: 1,
        got: 0,
    })?;
    let fv = crate::eval::eval_expr(f_expr, env, trace)?;
    let (Value::UserFnRef { name } | Value::BuiltinRef { name }) = fv else {
        let kind = mlpl_eval_types::value_kind(&fv);
        return Err(EvalError::Unsupported(format!(
            "call: first argument must be a function reference (`:u:name` or `:name`) -- got {kind}"
        )));
    };
    let call = Expr::FnCall {
        name,
        args: rest.to_vec(),
        span: *span,
    };
    crate::eval::eval_expr(&call, env, trace)
}
