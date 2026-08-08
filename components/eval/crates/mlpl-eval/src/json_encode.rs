//! General deterministic JSON encoder -- the encode half of the
//! `parse_json` <-> `to_json` round trip. Records become objects
//! with SORTED keys (BTreeMap order), rank-1 arrays flat lists,
//! higher-rank arrays nest by shape, strings are JSON-escaped
//! with exact Unicode, numbers print bare-integer or float, and
//! Results carry an `{ok, value|error}` shape. Value kinds that
//! are not DATA (models, tokenizers, generation state, partials,
//! references, device tensors) are a loud error.

use std::collections::BTreeMap;

use mlpl_array::DenseArray;
use mlpl_eval_types::{EvalError, Value};

/// Encode any data value to a JSON string, or error on a
/// non-serializable kind.
pub(crate) fn to_json(value: &Value) -> Result<String, EvalError> {
    let mut out = String::new();
    encode(value, &mut out)?;
    Ok(out)
}

fn encode(value: &Value, out: &mut String) -> Result<(), EvalError> {
    match value {
        Value::Str(s) => push_str_json(out, s),
        Value::Array(a) => return encode_array(a, out),
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
        Value::Record { fields } => return encode_object(fields, out),
        Value::Result { ok, payload } => {
            out.push_str(if *ok {
                "{\"ok\":true,\"value\":"
            } else {
                "{\"ok\":false,\"error\":"
            });
            encode(payload, out)?;
            out.push('}');
        }
        other => {
            return Err(EvalError::Unsupported(format!(
                "to_json: cannot serialize a {} (only numbers, strings, arrays, string \
                 lists, records, and results are JSON data)",
                mlpl_eval_types::value_kind(other)
            )));
        }
    }
    Ok(())
}

/// Encode a record as a JSON object with sorted keys.
pub(crate) fn encode_object(
    fields: &BTreeMap<String, Value>,
    out: &mut String,
) -> Result<(), EvalError> {
    out.push('{');
    for (i, (k, v)) in fields.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_str_json(out, k);
        out.push(':');
        encode(v, out)?;
    }
    out.push('}');
    Ok(())
}

/// rank-0 -> number; rank-1 -> flat list; rank>=2 -> nested by
/// shape (round-trips through parse_json only up to rank 1). A
/// non-finite cell (NaN / +-Inf) is an error -- JSON has no such
/// token, so emitting one would produce output no parser accepts.
fn encode_array(a: &DenseArray, out: &mut String) -> Result<(), EvalError> {
    if let Some(n) = a.data().iter().copied().find(|n| !n.is_finite()) {
        return Err(EvalError::Unsupported(format!(
            "to_json: cannot serialize the non-finite number {n} (JSON has no NaN or infinity)"
        )));
    }
    let dims = a.shape().dims();
    if dims.is_empty() {
        push_number(out, a.data()[0]);
    } else {
        encode_nd(a.data(), dims, out);
    }
    Ok(())
}

fn encode_nd(data: &[f64], dims: &[usize], out: &mut String) {
    out.push('[');
    if dims.len() == 1 {
        for (i, n) in data.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            push_number(out, *n);
        }
    } else {
        let stride: usize = dims[1..].iter().product();
        for (i, chunk) in data.chunks(stride.max(1)).enumerate() {
            if i > 0 {
                out.push(',');
            }
            encode_nd(chunk, &dims[1..], out);
        }
    }
    out.push(']');
}

/// Integers print bare; other floats keep their Rust display.
pub(crate) fn push_number(out: &mut String, n: f64) {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        out.push_str(&format!("{}", n as i64));
    } else {
        out.push_str(&format!("{n}"));
    }
}

/// JSON string escaping, preserving exact text and Unicode.
pub(crate) fn push_str_json(out: &mut String, s: &str) {
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
