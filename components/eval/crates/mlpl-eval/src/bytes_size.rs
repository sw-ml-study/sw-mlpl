//! `size_bytes` support: the packed storage footprint of a value, in
//! bytes. Answers demo-memory's bytes-per-key question -- a numeric
//! array reports its f64 backing (8 bytes/element), a packed
//! `Value::Bytes` reports its exact length, and containers sum their
//! parts. Opaque kinds (models, refs, ...) have no defined footprint.

use mlpl_eval_types::Value;
use std::collections::BTreeMap;

/// The packed byte footprint of `v`, or `None` for a kind with no
/// well-defined storage size (model, tokenizer, gen-state, function
/// reference, partial, device tensor).
pub(crate) fn packed_size(v: &Value) -> Option<usize> {
    match v {
        Value::Bytes { data, .. } => Some(data.len()),
        Value::Array(a) => Some(a.elem_count() * 8),
        Value::Str(s) => Some(s.len()),
        Value::StrList { items } => Some(items.iter().map(String::len).sum()),
        Value::Result { payload, .. } => packed_size(payload),
        Value::Record { fields } => record_size(fields),
        _ => None,
    }
}

/// A record's footprint: each key's UTF-8 bytes plus its value's
/// footprint. `None` if any field value is an opaque kind.
fn record_size(fields: &BTreeMap<String, Value>) -> Option<usize> {
    let mut total = 0;
    for (key, val) in fields {
        total += key.len() + packed_size(val)?;
    }
    Some(total)
}
