//! Section-valued TOML fields: nested records, and Results
//! encoded as a `{ok, value|error}` sub-table (so `ok(x)` /
//! `err(e)` round-trip with `parse_toml(..., {results: 1})`).
//! Each becomes a `[prefix.key]` table whose body is emitted by
//! `toml_encode::encode_table`.

use std::collections::BTreeMap;

use mlpl_array::DenseArray;
use mlpl_eval_types::Value;

/// Emit every record and Result field of `fields` as its own
/// `[section]` table, in sorted order.
pub(crate) fn encode_sections(
    fields: &BTreeMap<String, Value>,
    prefix: &str,
    out: &mut String,
) -> Result<(), String> {
    for (k, v) in fields {
        match v {
            Value::Record { fields: sub } => encode_section(k, sub, prefix, out)?,
            Value::Result { ok, payload } => {
                encode_section(k, &result_table(*ok, payload), prefix, out)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Emit one table as `[prefix.key]` (blank line before it unless
/// it opens the document), then its body.
fn encode_section(
    k: &str,
    sub: &BTreeMap<String, Value>,
    prefix: &str,
    out: &mut String,
) -> Result<(), String> {
    let key = crate::toml_scalar::bare_key(k)?;
    let path = if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    };
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(&format!("[{path}]\n"));
    crate::toml_encode::encode_table(sub, &path, out)
}

/// A Result as its `{ok, value|error}` record (ok as 1/0).
fn result_table(ok: bool, payload: &Value) -> BTreeMap<String, Value> {
    let mut sub = BTreeMap::new();
    let flag = if ok { 1.0 } else { 0.0 };
    sub.insert(
        "ok".to_string(),
        Value::Array(DenseArray::from_scalar(flag)),
    );
    let field = if ok { "value" } else { "error" };
    sub.insert(field.to_string(), payload.clone());
    sub
}
