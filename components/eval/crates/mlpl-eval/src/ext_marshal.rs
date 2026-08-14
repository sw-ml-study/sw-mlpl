//! Value marshaling for the extension boundary: MLPL `Value` <->
//! the V1 scalar `ExtValue`, and mapping a contained call result
//! back into an MLPL value (domain failure -> `err(...)`,
//! contained panic -> hard `EvalError`).

use mlpl_array::{DenseArray, Shape};
use mlpl_extension_abi::{ExtDtype, ExtError, ExtValue};

use mlpl_eval_types::{EvalError, Value};

/// Marshal an MLPL value into the V1 boundary set. Rank-0 arrays are
/// scalars (int/float); a rank>=1 array crosses as a dense `f64`
/// array (the extension narrows to its wire dtype).
pub(crate) fn to_ext(name: &str, v: Value) -> Result<ExtValue, EvalError> {
    match v {
        Value::Array(a) if a.rank() == 0 => {
            let n = a.data()[0];
            if n.is_finite() && n.fract() == 0.0 {
                Ok(ExtValue::I64(n as i64))
            } else {
                Ok(ExtValue::F64(n))
            }
        }
        Value::Array(a) => Ok(ExtValue::Array {
            dtype: ExtDtype::F64,
            shape: a.shape().dims().to_vec(),
            data: a.data().to_vec(),
        }),
        Value::Str(s) => Ok(ExtValue::Str(s)),
        other => Err(EvalError::ExtensionError {
            function: name.to_string(),
            message: format!(
                "unsupported argument kind {} (extensions take scalars/strings in this slice)",
                mlpl_eval_types::value_kind(&other)
            ),
        }),
    }
}

/// Map a contained extension call result into an MLPL value: a
/// domain failure becomes `err(message)`, a contained panic a hard
/// `EvalError::ExtensionError`.
pub(crate) fn finish(name: &str, r: Result<ExtValue, ExtError>) -> Result<Value, EvalError> {
    match r {
        Ok(v) => Ok(from_ext(v)),
        Err(e) if e.panicked => Err(EvalError::ExtensionError {
            function: name.to_string(),
            message: e.message,
        }),
        Err(e) => Ok(Value::Result {
            ok: false,
            payload: Box::new(Value::Str(e.message)),
        }),
    }
}

/// Marshal a V1 boundary value back into an MLPL value.
fn from_ext(v: ExtValue) -> Value {
    match v {
        ExtValue::Nil => Value::Array(DenseArray::from_vec(Vec::new())),
        ExtValue::Bool(b) => Value::Array(DenseArray::from_scalar(f64::from(u8::from(b)))),
        ExtValue::I64(i) => Value::Array(DenseArray::from_scalar(i as f64)),
        ExtValue::F64(x) => Value::Array(DenseArray::from_scalar(x)),
        ExtValue::Str(s) => Value::Str(s),
        ExtValue::Bytes(b) => Value::Array(DenseArray::from_vec(
            b.iter().map(|&x| f64::from(x)).collect(),
        )),
        ExtValue::Array { shape, data, .. } => Value::Array(
            DenseArray::new(Shape::new(shape), data.clone())
                .unwrap_or_else(|_| DenseArray::from_vec(data)),
        ),
    }
}
