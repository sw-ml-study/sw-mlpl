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

/// The dtype a typed-reader builtin decodes, from its name:
/// `read_u32_le` -> u32, `read_u8` -> u8, `read_f64_le` -> f64.
/// `None` for any non-reader name (so `read_bytes` / `read_stdin`
/// fall through to their own dispatchers).
pub(crate) fn reader_dtype(name: &str) -> Option<ByteDtype> {
    let token = name.strip_prefix("read_")?;
    ByteDtype::parse(token.strip_suffix("_le").unwrap_or(token))
}

/// Evaluate `arg` to a non-negative integer byte offset.
pub(crate) fn expect_offset(
    func: &str,
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<usize, EvalError> {
    let arr = eval_expr(arg, env, trace)?.into_array()?;
    let n = arr.data().first().copied().unwrap_or(f64::NAN);
    if arr.rank() != 0 || n < 0.0 || n.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "{func}: offset must be a non-negative integer, got {n}"
        )));
    }
    Ok(n as usize)
}
