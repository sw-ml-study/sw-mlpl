//! TOML codec dispatch: `to_toml(record)` (and, later,
//! `parse_toml(text)`), both Result-based like the JSON codec --
//! ok(...) on success, err(message) otherwise.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "to_toml" => Some(eval_to_toml(args, env, trace)),
        "parse_toml" => Some(eval_parse_toml(args, env, trace)),
        _ => None,
    }
}

fn eval_parse_toml(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (text, limits) = crate::decode_limits::text_and_limits("parse_toml", args, env, trace)?;
    Ok(match crate::toml_decode::decode(&text, &limits) {
        Ok(v) => Value::Result {
            ok: true,
            payload: Box::new(v),
        },
        Err(msg) => Value::Result {
            ok: false,
            payload: Box::new(Value::Str(msg)),
        },
    })
}

fn eval_to_toml(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    crate::grad::arity_check(args, 1, "to_toml")?;
    let v = crate::eval::eval_expr(&args[0], env, trace)?;
    Ok(match crate::toml_encode::to_toml(&v) {
        Ok(s) => Value::Result {
            ok: true,
            payload: Box::new(Value::Str(s)),
        },
        Err(msg) => Value::Result {
            ok: false,
            payload: Box::new(Value::Str(msg)),
        },
    })
}
