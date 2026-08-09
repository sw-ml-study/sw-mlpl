//! Typed-native binary codec dispatch: `to_native(value)` and
//! (step 2) `parse_native(bytes[, limits])`, both Result-based
//! like the JSON/TOML codecs. The encoded bytes are a rank-1
//! array of byte values (`0..256`), so they feed straight into
//! `write_bytes` / `write_atomic` and come back from `read_bytes`.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "to_native" => Some(eval_to_native(args, env, trace)),
        _ => None,
    }
}

fn eval_to_native(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    crate::grad::arity_check(args, 1, "to_native")?;
    let v = eval_expr(&args[0], env, trace)?;
    let encoded = crate::native_encode::to_native(&v)
        .map(bytes_to_array)
        .map_err(|e| format!("{e}"));
    Ok(crate::result_str::ok_or_err(encoded))
}

/// A byte vector as a rank-1 `Value::Array` of `0..256` values.
fn bytes_to_array(bytes: Vec<u8>) -> Value {
    Value::Array(DenseArray::from_vec(
        bytes.iter().map(|&b| f64::from(b)).collect(),
    ))
}
