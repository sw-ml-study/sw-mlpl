//! String-building builtins: `str_concat` (join two strings) and
//! `str_join` (join a string list with a separator). Exact,
//! byte-for-byte, Unicode-preserving; NO coercion -- a non-string
//! argument is an error, never a silent `to_string`. `str_join` is
//! the linear-time fold (`Vec::join`, O(total)), the answer to
//! "build a string from many pieces" without an O(n^2) reduce.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "str_concat" => Some(eval_str_concat(args, env, trace)),
        "str_join" => Some(eval_str_join(args, env, trace)),
        "to_string" => Some(eval_to_string(args, env, trace)),
        _ => None,
    }
}

/// `to_string(x)` -> the shortest round-trip decimal of a scalar
/// number, the honest inverse of `to_number`: integral values print
/// bare (`to_string(8 / 2)` is `"4"`, not `"4.0"`) using the same
/// formatting `to_json` gives a scalar, so `to_number(to_string(x))`
/// recovers `x` for every finite `f64`. A non-scalar / non-number is
/// an error (no format spec; rounding belongs in a library).
fn eval_to_string(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [x] = args else {
        return Err(EvalError::BadArity {
            func: "to_string".into(),
            expected: 1,
            got: args.len(),
        });
    };
    match crate::eval::eval_expr(x, env, trace)? {
        Value::Array(a) if a.rank() == 0 => {
            let mut s = String::new();
            crate::json_encode::push_number(&mut s, a.data()[0]);
            Ok(Value::Str(s))
        }
        _ => Err(EvalError::Unsupported(
            "to_string: expected a scalar number".into(),
        )),
    }
}

/// Exactly two argument expressions, or a `BadArity` error.
fn two_args<'a>(name: &str, args: &'a [Expr]) -> Result<(&'a Expr, &'a Expr), EvalError> {
    match args {
        [a, b] => Ok((a, b)),
        _ => Err(EvalError::BadArity {
            func: name.into(),
            expected: 2,
            got: args.len(),
        }),
    }
}

/// `str_concat(a, b)` -> the two strings joined. Both must be
/// strings (no coercion).
fn eval_str_concat(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (a, b) = two_args("str_concat", args)?;
    match (
        crate::eval::eval_expr(a, env, trace)?,
        crate::eval::eval_expr(b, env, trace)?,
    ) {
        (Value::Str(x), Value::Str(y)) => Ok(Value::Str(format!("{x}{y}"))),
        _ => Err(EvalError::Unsupported(
            "str_concat: both arguments must be strings (no coercion)".into(),
        )),
    }
}

/// `str_join(parts, separator)` -> the string list joined. `parts` is
/// a string list (an empty list yields `""`); `separator` is a
/// string. Linear in the total length.
fn eval_str_join(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (parts_arg, sep_arg) = two_args("str_join", args)?;
    let parts = crate::eval::eval_expr(parts_arg, env, trace)?;
    let Value::Str(separator) = crate::eval::eval_expr(sep_arg, env, trace)? else {
        return Err(EvalError::Unsupported(
            "str_join: the separator must be a string".into(),
        ));
    };
    let items: Vec<&str> = match &parts {
        Value::StrList { items } => items.iter().map(String::as_str).collect(),
        // An empty list literal is an empty array, not a StrList.
        Value::Array(a) if a.elem_count() == 0 => Vec::new(),
        _ => {
            return Err(EvalError::Unsupported(
                "str_join: the first argument must be a list of strings".into(),
            ));
        }
    };
    Ok(Value::Str(items.join(separator.as_str())))
}
