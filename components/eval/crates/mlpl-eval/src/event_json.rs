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
        push_str_json(&mut out, k);
        out.push(':');
        push_value(&mut out, v);
    }
    out.push('}');
    out
}

fn push_value(out: &mut String, v: &Value) {
    match v {
        Value::Str(s) => push_str_json(out, s),
        Value::Array(a) if a.rank() == 0 => push_number(out, a.data()[0]),
        Value::Array(a) => {
            out.push('[');
            for (i, n) in a.data().iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_number(out, *n);
            }
            out.push(']');
        }
        Value::StrList { items } => {
            out.push('[');
            for (i, s) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_str_json(out, s);
            }
            out.push(']');
        }
        Value::Record { fields } => out.push_str(&event_line(fields)),
        other => push_str_json(out, &format!("{other}")),
    }
}

/// Integers print bare (line numbers, counts); other floats keep
/// their Rust display form.
fn push_number(out: &mut String, n: f64) {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        out.push_str(&format!("{}", n as i64));
    } else {
        out.push_str(&format!("{n}"));
    }
}

fn push_str_json(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
