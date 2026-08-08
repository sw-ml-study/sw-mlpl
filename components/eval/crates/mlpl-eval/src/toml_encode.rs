//! TOML encoder -- the encode half of the TOML codec. A TOML
//! document is a table, so the root must be a record. Fields emit
//! DETERMINISTICALLY (BTreeMap sorted order): scalar / string /
//! array fields as `key = value` first, then nested-record fields
//! as `[section]` / `[section.sub]` tables. Result-based via
//! `fncall_toml`; here the return is a raw message on error.

use std::collections::BTreeMap;

use mlpl_eval_types::Value;

/// Encode a record to a TOML document, or a message on error.
pub(crate) fn to_toml(value: &Value) -> Result<String, String> {
    let Value::Record { fields } = value else {
        return Err(format!(
            "to_toml: the top-level value must be a record (a TOML document is a table), got {}",
            mlpl_eval_types::value_kind(value)
        ));
    };
    let mut out = String::new();
    encode_table(fields, "", &mut out)?;
    Ok(out)
}

/// Scalar/string/array fields first (as `key = value`), then the
/// section-valued fields (records and Results) via
/// `toml_sections` -- TOML requires a table's own keys to precede
/// its sub-table headers.
pub(crate) fn encode_table(
    fields: &BTreeMap<String, Value>,
    prefix: &str,
    out: &mut String,
) -> Result<(), String> {
    let is_section = |v: &Value| matches!(v, Value::Record { .. } | Value::Result { .. });
    for (k, v) in fields.iter().filter(|(_, v)| !is_section(v)) {
        out.push_str(crate::toml_scalar::bare_key(k)?);
        out.push_str(" = ");
        encode_value(v, out)?;
        out.push('\n');
    }
    crate::toml_sections::encode_sections(fields, prefix, out)
}

/// A scalar / string / array field value (never a record or
/// Result -- those are handled as sub-tables by `toml_sections`).
pub(crate) fn encode_value(v: &Value, out: &mut String) -> Result<(), String> {
    match v {
        Value::Str(s) => {
            crate::json_encode::push_str_json(out, s);
            Ok(())
        }
        Value::StrList { items } => {
            out.push('[');
            for (i, s) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                crate::json_encode::push_str_json(out, s);
            }
            out.push(']');
            Ok(())
        }
        Value::Array(a) => crate::toml_scalar::encode_array(a, out),
        other => Err(format!(
            "to_toml: cannot represent a {} as a TOML value",
            mlpl_eval_types::value_kind(other)
        )),
    }
}
