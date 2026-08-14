//! `str_find` / `str_split` (algebra work-order B3), split from
//! `fncall_string_ops` to stay within the module function budget.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_string_ops::expect_string;
use mlpl_eval_types::{EvalError, Value};

/// `str_find(s, needle)` -> the first CHARACTER index of `needle` in
/// `s`, or -1 if absent. Empty needle matches at 0.
pub(crate) fn eval_str_find(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [s_arg, needle_arg] = args else {
        return Err(EvalError::BadArity {
            func: "str_find".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let s = expect_string("str_find", s_arg, env, trace)?;
    let needle = expect_string("str_find", needle_arg, env, trace)?;
    let idx = match s.find(&needle) {
        Some(byte_pos) => s[..byte_pos].chars().count() as f64,
        None => -1.0,
    };
    Ok(Value::Array(DenseArray::from_scalar(idx)))
}

/// `str_split(s, sep)` -> the pieces of `s` between `sep`, as a string
/// list. An absent separator yields the whole string; an empty
/// separator splits into individual characters.
pub(crate) fn eval_str_split(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [s_arg, sep_arg] = args else {
        return Err(EvalError::BadArity {
            func: "str_split".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let s = expect_string("str_split", s_arg, env, trace)?;
    let sep = expect_string("str_split", sep_arg, env, trace)?;
    let items: Vec<String> = if sep.is_empty() {
        s.chars().map(|c| c.to_string()).collect()
    } else {
        s.split(sep.as_str()).map(str::to_string).collect()
    };
    Ok(Value::StrList { items })
}
