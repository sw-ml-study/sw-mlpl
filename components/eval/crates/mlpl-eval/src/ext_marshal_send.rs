//! SEND direction of the extension boundary: an MLPL `Value`
//! argument -> the V1 `ExtValue` set. Rank-0 arrays cross as
//! scalars, rank>=1 arrays as a dense `f64` array, records recurse,
//! packed buffers cross as bytes, and an `ExtHandle` passes by value. The RECEIVE direction (call
//! result -> `Value`) lives in `ext_marshal_recv`.

use mlpl_extension_abi::{ExtDtype, ExtHandle, ExtValue};

use mlpl_eval_types::{EvalError, Value};

/// Marshal an MLPL value into the V1 boundary set. Rank-0 arrays are
/// scalars (int/float); a rank>=1 array crosses as a dense `f64`
/// array (the extension narrows to its wire dtype).
pub(crate) fn to_ext(name: &str, v: Value) -> Result<ExtValue, EvalError> {
    match v {
        Value::Array(a) if a.rank() == 0 => Ok(scalar_ext(a.data()[0])),
        Value::Array(a) => Ok(ExtValue::Array {
            dtype: ExtDtype::F64,
            shape: a.shape().dims().to_vec(),
            data: a.data().to_vec(),
        }),
        Value::Str(s) => Ok(ExtValue::Str(s)),
        Value::Bytes { data, .. } => Ok(ExtValue::Bytes(data)),
        Value::Record { fields } => record_ext(name, fields),
        Value::ExtHandle {
            extension_id,
            type_id,
            slot,
            generation,
        } => Ok(ExtValue::Handle(ExtHandle {
            extension_id,
            type_id,
            slot,
            generation,
        })),
        other => Err(unsupported(name, &other)),
    }
}

/// Recursively marshal a sorted MLPL record into provider order.
fn record_ext(
    name: &str,
    fields: std::collections::BTreeMap<String, Value>,
) -> Result<ExtValue, EvalError> {
    fields
        .into_iter()
        .map(|(field, value)| to_ext(name, value).map(|value| (field, value)))
        .collect::<Result<Vec<_>, _>>()
        .map(ExtValue::Record)
}

/// A rank-0 array crosses as an integer when its single element is
/// a whole finite number, else as a float (the boundary scalar set
/// has no separate rank-0 array kind).
fn scalar_ext(n: f64) -> ExtValue {
    if n.is_finite() && n.fract() == 0.0 {
        ExtValue::I64(n as i64)
    } else {
        ExtValue::F64(n)
    }
}

/// The boundary-contract error for an argument kind the V1 set
/// cannot carry (for example a model or tokenizer).
fn unsupported(name: &str, v: &Value) -> EvalError {
    EvalError::ExtensionError {
        function: name.to_string(),
        message: format!(
            "unsupported argument kind {} at extension boundary",
            mlpl_eval_types::value_kind(v)
        ),
    }
}
