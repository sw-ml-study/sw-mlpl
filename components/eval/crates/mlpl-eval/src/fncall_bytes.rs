//! Typed packed byte-buffer builtins. `pack(array, "dtype")` packs an
//! f64 array into a canonical little-endian `Value::Bytes` buffer of
//! the named element dtype (u8..f64). The separate systems-data path:
//! bounded, index/offset-only, Result-valued on domain errors.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::bytes_args::parse_dtype;
use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value, pack_f64s, value_kind};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "pack" => Some(eval_pack(args, env, trace)),
        "size_bytes" => Some(eval_size_bytes(args, env, trace)),
        "reinterpret" => Some(eval_reinterpret(args, env, trace)),
        _ => None,
    }
}

/// `size_bytes(x)` -> the packed storage footprint of `x`, in bytes.
fn eval_size_bytes(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [x] = args else {
        return Err(EvalError::BadArity {
            func: "size_bytes".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let v = eval_expr(x, env, trace)?;
    let n = crate::bytes_size::packed_size(&v).ok_or_else(|| {
        EvalError::Unsupported(format!(
            "size_bytes: no defined footprint for {}",
            value_kind(&v)
        ))
    })?;
    Ok(Value::Array(DenseArray::from_scalar(n as f64)))
}

/// `pack(array, "dtype")` -> a little-endian packed byte buffer.
fn eval_pack(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [arr_arg, dtype_arg] = args else {
        return Err(EvalError::BadArity {
            func: "pack".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let arr = eval_expr(arr_arg, env, trace)?.into_array()?;
    let dtype = parse_dtype("pack", dtype_arg, env, trace)?;
    let data = pack_f64s(arr.data(), dtype).map_err(EvalError::Unsupported)?;
    Ok(Value::Bytes { dtype, data })
}

/// `reinterpret(bytes, "dtype")` -> the SAME bytes re-viewed under a
/// new element dtype (no numeric conversion). The byte length must be
/// a whole number of the new dtype's elements.
fn eval_reinterpret(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [buf_arg, dtype_arg] = args else {
        return Err(EvalError::BadArity {
            func: "reinterpret".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let data = crate::bytes_args::expect_bytes("reinterpret", buf_arg, env, trace)?;
    let dtype = parse_dtype("reinterpret", dtype_arg, env, trace)?;
    if data.len() % dtype.width() != 0 {
        return Err(EvalError::Unsupported(format!(
            "reinterpret: {} bytes is not a whole number of {dtype} values",
            data.len()
        )));
    }
    Ok(Value::Bytes { dtype, data })
}
