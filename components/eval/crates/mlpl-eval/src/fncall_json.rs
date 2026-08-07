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
    (name == "parse_json").then(|| eval_parse_json(args, env, trace))
}

fn eval_parse_json(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [arg] = args else {
        return Err(EvalError::BadArity {
            func: "parse_json".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let Value::Str(text) = crate::eval::eval_expr(arg, env, trace)? else {
        return Err(EvalError::Unsupported(
            "parse_json: the argument is JSON text (a string)".into(),
        ));
    };
    Ok(match crate::json_decode::decode(&text) {
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
