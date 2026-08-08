//! The decode half of the `$mlpl` tagged envelope: walk a decoded
//! value and rebuild any envelope back into its original kind --
//! `array` (shape + flat data) into a rank->=2 array, `result`
//! into ok/err, `record` (the escape) into the raw record.
//! Applied UNCONDITIONALLY after parse_json (the reserved key is
//! never application data). A malformed or unknown envelope is
//! left as an ordinary record (best-effort, never a crash).

use std::collections::BTreeMap;

use mlpl_array::{DenseArray, Shape};
use mlpl_eval_types::Value;

/// Rebuild `$mlpl` envelopes throughout a decoded value.
pub(crate) fn unwrap_envelopes(v: Value) -> Value {
    let Value::Record { fields } = v else {
        return v;
    };
    if fields.len() == 1
        && let Some(Value::Record { fields: inner }) = fields.get("$mlpl")
        && let Some(rebuilt) = rebuild(inner)
    {
        return rebuilt;
    }
    Value::Record {
        fields: fields
            .into_iter()
            .map(|(k, x)| (k, unwrap_envelopes(x)))
            .collect(),
    }
}

/// Rebuild from the inner envelope record, or None if unrecognized.
fn rebuild(inner: &BTreeMap<String, Value>) -> Option<Value> {
    let Some(Value::Str(ty)) = inner.get("type") else {
        return None;
    };
    match ty.as_str() {
        "array" => rebuild_array(inner),
        "result" => rebuild_result(inner),
        "record" => match inner.get("fields") {
            Some(Value::Record { fields }) => Some(unwrap_envelopes(Value::Record {
                fields: fields.clone(),
            })),
            _ => None,
        },
        _ => None,
    }
}

fn rebuild_array(inner: &BTreeMap<String, Value>) -> Option<Value> {
    let (Some(Value::Array(shape)), Some(Value::Array(data))) =
        (inner.get("shape"), inner.get("data"))
    else {
        return None;
    };
    let dims: Vec<usize> = shape.data().iter().map(|&d| d as usize).collect();
    DenseArray::new(Shape::new(dims), data.data().to_vec())
        .ok()
        .map(Value::Array)
}

fn rebuild_result(inner: &BTreeMap<String, Value>) -> Option<Value> {
    let Some(Value::Str(variant)) = inner.get("variant") else {
        return None;
    };
    let (ok, key) = match variant.as_str() {
        "ok" => (true, "value"),
        "err" => (false, "error"),
        _ => return None,
    };
    inner.get(key).map(|payload| Value::Result {
        ok,
        payload: Box::new(unwrap_envelopes(payload.clone())),
    })
}
