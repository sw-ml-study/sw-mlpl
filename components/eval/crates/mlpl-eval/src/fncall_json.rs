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
    let (text, limits, reconstruct) =
        crate::decode_limits::text_and_options("parse_json", args, env, trace)?;
    Ok(match crate::json_decode::decode(&text, &limits) {
        Ok(v) => Value::Result {
            ok: true,
            payload: Box::new(if reconstruct {
                crate::result_reconstruct::reconstruct(v)
            } else {
                v
            }),
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
    if !(1..=2).contains(&args.len()) {
        return Err(EvalError::BadArity {
            func: "to_json".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let v = crate::eval::eval_expr(&args[0], env, trace)?;
    // {tagged: 1} -> reserved $mlpl envelope (lossless rank->=2 / Results).
    let value = if wants_tagged(args.get(1), env, trace)? {
        crate::envelope::wrap(&v)
    } else {
        v
    };
    let encoded = crate::json_encode::to_json(&value)
        .map(Value::Str)
        .map_err(|e| format!("{e}"));
    Ok(crate::result_str::ok_or_err(encoded))
}

/// Read the optional `{tagged: 1}` option of `to_json`. A missing
/// arg is false; a non-record option is a hard error.
fn wants_tagged(
    opt: Option<&Expr>,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<bool, EvalError> {
    let Some(a) = opt else {
        return Ok(false);
    };
    match crate::eval::eval_expr(a, env, trace)? {
        Value::Record { fields } => Ok(matches!(
            fields.get("tagged"),
            Some(Value::Array(t)) if t.rank() == 0 && t.data()[0] != 0.0
        )),
        _ => Err(EvalError::Unsupported(
            "to_json: options must be a record".into(),
        )),
    }
}
