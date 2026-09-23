//! `unpack(bytes, "dtype")` -> a flat 1-D array: the bulk inverse of `pack`.
//! The whole buffer decodes in one native pass (`mlpl_bytes::unpack_le`), so a
//! model-weight tensor stored as bf16 loads without one interpreter call per
//! value. Every dtype `reinterpret` accepts is supported, including the
//! decode-only `bf16` / `f16` (subnormals, infinities and NaN preserved).

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::bytes_args::{expect_bytes, parse_dtype};
use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value, unpack_le};

pub(crate) fn eval_unpack(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [buf_arg, dtype_arg] = args else {
        return Err(EvalError::BadArity {
            func: "unpack".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let data = expect_bytes("unpack", buf_arg, env, trace)?;
    let dtype = parse_dtype("unpack", dtype_arg, env, trace)?;
    let vals = unpack_le(&data, dtype).ok_or_else(|| {
        EvalError::Unsupported(format!(
            "unpack: {} bytes is not a whole number of {dtype} values",
            data.len()
        ))
    })?;
    let n = vals.len();
    Ok(Value::Array(DenseArray::new(Shape::new(vec![n]), vals)?))
}
