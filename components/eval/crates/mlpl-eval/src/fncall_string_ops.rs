//! Character-based string ops (algebra work-order B3): `str_len` and
//! `str_slice` here, `str_find` / `str_split` in `string_search`.
//! Indexing is by CHARACTER (Unicode scalar value), not byte -- so
//! `str_len("héllo")` is 5, and `str_slice` counts characters.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::bytes_args::expect_offset;
use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value, value_kind};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "str_len" => Some(eval_str_len(args, env, trace)),
        "str_slice" => Some(eval_str_slice(args, env, trace)),
        "str_eq" => Some(eval_str_eq(args, env, trace)),
        "str_find" => Some(crate::string_search::eval_str_find(args, env, trace)),
        "str_split" => Some(crate::string_search::eval_str_split(args, env, trace)),
        _ => None,
    }
}

/// `str_eq(a, b)` -> `1` if the two strings are exactly equal, else `0`.
/// WHOLE-string equality, not substring: `str_find` is the trap this
/// avoids -- `str_find("semigroup", "group")` is `4`, not `-1`, so a
/// substring test wrongly reports "semigroup" as "group". `eq` rejects
/// strings (it is an array op), so this is how two strings are compared.
fn eval_str_eq(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [a_arg, b_arg] = args else {
        return Err(EvalError::BadArity {
            func: "str_eq".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let a = expect_string("str_eq", a_arg, env, trace)?;
    let b = expect_string("str_eq", b_arg, env, trace)?;
    Ok(Value::Array(DenseArray::from_scalar(f64::from(a == b))))
}

/// Evaluate `arg` to a string, erroring on any other value kind.
pub(crate) fn expect_string(
    func: &str,
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<String, EvalError> {
    match eval_expr(arg, env, trace)? {
        Value::Str(s) => Ok(s),
        other => Err(EvalError::Unsupported(format!(
            "{func}: expected a string, got {}",
            value_kind(&other)
        ))),
    }
}

/// `str_len(s)` -> the number of CHARACTERS (not bytes).
fn eval_str_len(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [s_arg] = args else {
        return Err(EvalError::BadArity {
            func: "str_len".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let s = expect_string("str_len", s_arg, env, trace)?;
    Ok(Value::Array(DenseArray::from_scalar(
        s.chars().count() as f64
    )))
}

/// `str_slice(s, start, len)` -> the `len`-character substring starting
/// at character `start` (clamped to the end of the string).
fn eval_str_slice(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [s_arg, start_arg, len_arg] = args else {
        return Err(EvalError::BadArity {
            func: "str_slice".into(),
            expected: 3,
            got: args.len(),
        });
    };
    let s = expect_string("str_slice", s_arg, env, trace)?;
    let start = expect_offset("str_slice", start_arg, env, trace)?;
    let len = expect_offset("str_slice", len_arg, env, trace)?;
    Ok(Value::Str(s.chars().skip(start).take(len).collect()))
}
