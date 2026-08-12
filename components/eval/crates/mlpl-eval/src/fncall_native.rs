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
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "to_native" => Some(eval_to_native(args, env, trace)),
        "parse_native" => Some(eval_parse_native(args, env, trace)),
        _ => None,
    }
}

fn eval_parse_native(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if !(1..=2).contains(&args.len()) {
        return Err(EvalError::BadArity {
            func: "parse_native".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let arr = eval_expr(&args[0], env, trace)?.into_array()?;
    let bytes = crate::fs_bytes::array_to_bytes("parse_native", &arr)?;
    let opt = match args.get(1) {
        Some(a) => Some(eval_expr(a, env, trace)?),
        None => None,
    };
    let limits = crate::decode_limits::limits_only("parse_native", opt.as_ref())?;
    let decoded =
        crate::native_decode::decode(&bytes, &limits).map_err(|m| format!("parse_native: {m}"));
    Ok(crate::result_str::ok_or_err(decoded))
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
