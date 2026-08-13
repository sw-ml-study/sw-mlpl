//! String-list builtins split out of `eval_intercepts` (per
//! docs/code_metrics.md: split by responsibility). `list_len(xs)`
//! and `list_get(xs, i)` operate on `Value::StrList` values --
//! the line-oriented outputs of `read_stdin_lines()` and friends.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::{Value, value_kind};

/// `list_len(xs)` -> element count of a string list. An empty list
/// literal `[]` is an empty (numeric) array, so it reads as length 0.
pub(crate) fn eval_list_len(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    check_arity("list_len", args.len(), 1)?;
    let n = match eval_expr(&args[0], env, trace)? {
        Value::StrList { items } => items.len() as f64,
        Value::Array(a) if a.elem_count() == 0 => 0.0,
        other => {
            let msg = format!(
                "list_len: expected a string-list, got {}",
                value_kind(&other)
            );
            return Err(EvalError::Unsupported(msg));
        }
    };
    Ok(Value::Array(mlpl_array::DenseArray::from_scalar(n)))
}

/// `list_get(xs, i)` -> `ok(item)` or `err(msg)` for out-of-bounds.
pub(crate) fn eval_list_get(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    check_arity("list_get", args.len(), 2)?;
    let xs = eval_expr(&args[0], env, trace)?;
    let Value::StrList { items } = xs else {
        let msg = format!("list_get: expected a string-list, got {}", value_kind(&xs));
        return Err(EvalError::Unsupported(msg));
    };
    let i = parse_strlist_index(&args[1], env, trace)?;
    let n = items.len();
    let (ok, s) = match items.into_iter().nth(i) {
        Some(s) => (true, s),
        None => (
            false,
            format!("list_get: index {i} out of bounds (list has {n} items)"),
        ),
    };
    Ok(Value::Result {
        ok,
        payload: Box::new(Value::Str(s)),
    })
}

fn check_arity(name: &str, got: usize, expected: usize) -> Result<(), EvalError> {
    if got == expected {
        Ok(())
    } else {
        Err(EvalError::BadArity {
            func: name.into(),
            expected,
            got,
        })
    }
}

fn parse_strlist_index(
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<usize, EvalError> {
    let idx = eval_expr(arg, env, trace)?.into_array()?;
    if idx.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "list_get: index must be a scalar, got rank {}",
            idx.rank()
        )));
    }
    let v = idx.data()[0];
    if v < 0.0 || v.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "list_get: index must be a non-negative integer, got {v}"
        )));
    }
    Ok(v as usize)
}
