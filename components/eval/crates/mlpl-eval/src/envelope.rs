//! The reserved `$mlpl` tagged envelope (see
//! `docs/serialization-variant-encoding.md`). `wrap` rewrites a
//! value into a plain-JSON-representable form where a rank->=2
//! array becomes a `{shape, data}` envelope, a Result a
//! `{variant, value|error}` envelope, and a record that literally
//! holds a `$mlpl` key is escaped as a `{fields}` envelope;
//! everything else passes through, recursively. The existing
//! sorted-key encoder then serializes it. `unwrap_envelopes` is
//! the inverse, applied unconditionally after decoding.

use std::collections::BTreeMap;

use mlpl_array::{DenseArray, Shape};
use mlpl_eval_types::Value;

/// Rewrite a value into its `$mlpl`-enveloped form for tagged
/// encoding. Pure; recursive.
pub(crate) fn wrap(v: &Value) -> Value {
    match v {
        Value::Array(a) if a.rank() >= 2 => {
            let dims: Vec<f64> = a.shape().dims().iter().map(|&d| d as f64).collect();
            envelope(
                "array",
                vec![
                    ("shape".into(), rank1(dims)),
                    ("data".into(), rank1(a.data().to_vec())),
                ],
            )
        }
        Value::Result { ok, payload } => {
            let key = if *ok { "value" } else { "error" };
            let variant = if *ok { "ok" } else { "err" };
            envelope(
                "result",
                vec![
                    ("variant".into(), Value::Str(variant.into())),
                    (key.into(), wrap(payload)),
                ],
            )
        }
        Value::Record { fields } if fields.contains_key("$mlpl") => {
            envelope("record", vec![("fields".into(), wrap_fields(fields))])
        }
        Value::Record { fields } => wrap_fields(fields),
        other => other.clone(),
    }
}

/// A record with every field wrapped.
fn wrap_fields(fields: &BTreeMap<String, Value>) -> Value {
    Value::Record {
        fields: fields.iter().map(|(k, v)| (k.clone(), wrap(v))).collect(),
    }
}

/// Build `{"$mlpl": {"v": 1, "type": type_name, ...extra}}`.
fn envelope(type_name: &str, extra: Vec<(String, Value)>) -> Value {
    let mut inner = BTreeMap::new();
    inner.insert("v".into(), Value::Array(DenseArray::from_scalar(1.0)));
    inner.insert("type".into(), Value::Str(type_name.into()));
    inner.extend(extra);
    let mut outer = BTreeMap::new();
    outer.insert("$mlpl".into(), Value::Record { fields: inner });
    Value::Record { fields: outer }
}

fn rank1(data: Vec<f64>) -> Value {
    Value::Array(DenseArray::new(Shape::vector(data.len()), data).expect("rank-1 build"))
}
