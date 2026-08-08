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
        "type_of" => Some(eval_type_of(args, env, trace)),
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

/// `type_of(v)` -- the stable kind string of any value ("array",
/// "string", "record", "result", "string-list", "model",
/// "tokenizer", "gen-state", "partial", "builtin-ref",
/// "user-fn-ref", "device-tensor"). Total: it inspects the
/// already-evaluated value and never errors, so a program can
/// branch on a root value's kind before calling kind-specific
/// accessors. Shares the one classifier with error messages and
/// the connect viz layer (`mlpl_eval_types::value_kind`).
fn eval_type_of(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [arg] = args else {
        return Err(EvalError::BadArity {
            func: "type_of".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let v = crate::eval::eval_expr(arg, env, trace)?;
    Ok(Value::Str(mlpl_eval_types::value_kind(&v).to_string()))
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
    // Builtins keep the expression path (family dispatch handles
    // their flexible arities); user refs and partials take the
    // APPLICATIVE path (docs/combinators-design.md).
    if let Value::BuiltinRef { name } = &fv
        && !rest.is_empty()
    {
        let call = Expr::FnCall {
            name: name.clone(),
            args: rest.to_vec(),
            span: *span,
        };
        return crate::eval::eval_expr(&call, env, trace);
    }
    let vals = rest
        .iter()
        .map(|a| crate::eval::eval_expr(a, env, trace))
        .collect::<Result<Vec<_>, _>>()?;
    crate::callable_apply::apply_callable(&fv, &vals, env, trace)
}
