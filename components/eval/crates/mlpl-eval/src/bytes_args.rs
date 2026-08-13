//! Argument helpers shared by the packed-byte builtins: evaluate a
//! dtype-name argument to a `ByteDtype`. Split out of `fncall_bytes`
//! so the dispatch module stays within its function-count budget as
//! the byte-buffer family grows.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{ByteDtype, EvalError, Value, value_kind};

/// Evaluate a dtype-name argument to a `ByteDtype`, erroring on a
/// non-string or an unknown dtype name.
pub(crate) fn parse_dtype(
    func: &str,
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<ByteDtype, EvalError> {
    let name = match eval_expr(arg, env, trace)? {
        Value::Str(s) => s,
        other => {
            let msg = format!("{func}: dtype must be a string, got {}", value_kind(&other));
            return Err(EvalError::Unsupported(msg));
        }
    };
    ByteDtype::parse(&name)
        .ok_or_else(|| EvalError::Unsupported(format!("{func}: unknown dtype '{name}'")))
}

/// Evaluate `arg` to a packed byte buffer, returning its raw bytes;
/// errors if the value is not a `Value::Bytes`.
pub(crate) fn expect_bytes(
    func: &str,
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Vec<u8>, EvalError> {
    match eval_expr(arg, env, trace)? {
        Value::Bytes { data, .. } => Ok(data),
        other => Err(EvalError::Unsupported(format!(
            "{func}: first argument must be a byte buffer, got {}",
            value_kind(&other)
        ))),
    }
}
