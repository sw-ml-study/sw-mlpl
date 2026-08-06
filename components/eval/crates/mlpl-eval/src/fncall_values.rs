//! Value-plane tools over the FnCall surface: structural
//! equality/rendering, uniform reference invocation, and the
//! test/annotation reflection trio (which lives in
//! `fncall_reflect` / `reflect_info`).

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "equal" | "repr" => Some(eval_structural(name, args, env, trace)),
        "call" => Some(eval_call(args, env, trace, span)),
        "bracket" => Some(crate::fncall_bracket::eval_bracket(args, env, trace)),
        "tests" | "test_info" | "annotations" => {
            Some(crate::fncall_reflect::eval_reflect(name, args, env, trace))
        }
        "expunge" => Some(crate::expunge::eval_expunge(args, env, trace)),
        _ => None,
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
