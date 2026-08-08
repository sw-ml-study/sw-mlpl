//! Deterministic JSON-line serialization for test events: record
//! fields in BTreeMap (sorted) order, strings escaped per JSON
//! (exact text and Unicode preserved), scalars as numbers.

use std::collections::BTreeMap;

use mlpl_eval_types::Value;

/// One event record as a single-line JSON object.
pub(crate) fn event_line(fields: &BTreeMap<String, Value>) -> String {
    let mut out = String::from("{");
    for (i, (k, v)) in fields.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        crate::json_encode::push_str_json(&mut out, k);
        out.push(':');
        push_value(&mut out, v);
    }
    out.push('}');
    out
}

fn push_value(out: &mut String, v: &Value) {
    match v {
        Value::Str(s) => crate::json_encode::push_str_json(out, s),
        Value::Array(a) if a.rank() == 0 => crate::json_encode::push_number(out, a.data()[0]),
        Value::Array(a) => {
            out.push('[');
            for (i, n) in a.data().iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                crate::json_encode::push_number(out, *n);
            }
            out.push(']');
        }
        Value::StrList { items } => {
            out.push('[');
            for (i, s) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                crate::json_encode::push_str_json(out, s);
            }
            out.push(']');
        }
        Value::Record { fields } => out.push_str(&event_line(fields)),
        other => crate::json_encode::push_str_json(out, &format!("{other}")),
    }
}
