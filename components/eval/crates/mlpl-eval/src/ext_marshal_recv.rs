//! RECEIVE direction of the extension boundary: map a contained
//! call result back into an MLPL `Value` -- a domain failure to an
//! `err(...)` Result, a contained panic to a hard `EvalError`, and
//! a returned `ExtValue` (scalar / array / handle) to its `Value`.
//! The SEND direction (argument marshaling) lives in
//! `ext_marshal_send`.

use mlpl_array::{DenseArray, Shape};
use mlpl_extension_abi::{ExtError, ExtValue};

use mlpl_eval_types::{EvalError, Value};

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
        ExtValue::Handle(h) => Value::ExtHandle {
            extension_id: h.extension_id,
            type_id: h.type_id,
            slot: h.slot,
            generation: h.generation,
        },
    }
}
