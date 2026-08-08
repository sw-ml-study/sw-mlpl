//! `parse_json(s)` -- JSON text to typed MLPL values, the
//! inverse of the test-event encoder: objects become records,
//! strings stay strings (Unicode and escapes exact), numbers
//! become scalars, homogeneous arrays become vectors or string
//! lists, `true`/`false` become 1/0, and `null` becomes the
//! empty vector (absence as data). Malformed input is an err
//! VALUE naming the byte position -- runner code composes with
//! `?`.

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
        "parse_json" => Some(eval_parse_json(args, env, trace)),
        "to_json" => Some(eval_to_json(args, env, trace)),
        _ => None,
    }
}

fn eval_parse_json(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (text, limits) = crate::decode_limits::text_and_limits("parse_json", args, env, trace)?;
    Ok(match crate::json_decode::decode(&text, &limits) {
        Ok(v) => Value::Result {
            ok: true,
            payload: Box::new(v),
        },
        Err(msg) => Value::Result {
            ok: false,
            payload: Box::new(Value::Str(format!("parse_json: {msg}"))),
        },
    })
}

fn eval_to_json(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    crate::grad::arity_check(args, 1, "to_json")?;
    let v = crate::eval::eval_expr(&args[0], env, trace)?;
    // Result-based like parse_json: ok(json) on success, err(msg)
    // for a non-data kind or a non-finite number.
    Ok(match crate::json_encode::to_json(&v) {
        Ok(s) => Value::Result {
            ok: true,
            payload: Box::new(Value::Str(s)),
        },
        Err(e) => Value::Result {
            ok: false,
            payload: Box::new(Value::Str(format!("{e}"))),
        },
    })
}
