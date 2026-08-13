//! Typed packed byte-buffer builtins. `pack(array, "dtype")` packs an
//! f64 array into a canonical little-endian `Value::Bytes` buffer of
//! the named element dtype (u8..f64). The separate systems-data path:
//! bounded, index/offset-only, Result-valued on domain errors.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{ByteDtype, EvalError, Value, pack_f64s, value_kind};

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

/// Evaluate a dtype-name argument to a `ByteDtype`, erroring on a
/// non-string or an unknown dtype name.
fn parse_dtype(
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
