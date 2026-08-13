//! The typed little-endian reader family: `read_u8`, `read_u16_le`,
//! ... `read_f64_le`, each `(bytes, offset) -> scalar`. One handler
//! decodes them all (the dtype comes from the builtin name via
//! `bytes_args::reader_dtype`), so the family is defined once.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::bytes_args::{expect_bytes, expect_offset};
use crate::env::Environment;
use mlpl_eval_types::{ByteDtype, EvalError, Value, read_le};

/// Read a `dtype` value at `args = (bytes, offset)`, little-endian,
/// as a scalar. Out-of-bounds or a non-buffer argument is a catchable
/// error.
pub(crate) fn eval_read_typed(
    dtype: ByteDtype,
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [buf_arg, off_arg] = args else {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    };
    let data = expect_bytes(name, buf_arg, env, trace)?;
    let offset = expect_offset(name, off_arg, env, trace)?;
    let value = read_le(&data, offset, dtype).ok_or_else(|| {
        EvalError::Unsupported(format!(
            "{name}: reading {} bytes at offset {offset} exceeds the {}-byte buffer",
            dtype.width(),
            data.len()
        ))
    })?;
    Ok(Value::Array(DenseArray::from_scalar(value)))
}
