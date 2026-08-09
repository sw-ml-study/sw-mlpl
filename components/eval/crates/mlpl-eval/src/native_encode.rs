//! Typed-native binary encoder (`to_native`): a versioned,
//! self-describing byte format that losslessly encodes every MLPL
//! data value. Canonical little-endian. Header: magic `MLPB` +
//! version 1 + payload length (u32 LE); then one tagged value.
//! Non-data kinds are a loud error (never partial). Deterministic:
//! records encode BTreeMap-sorted, all numbers as f64 (one numeric
//! element type). Container encoders live in `native_encode_parts`
//! to keep each module within budget.

use mlpl_eval_types::{EvalError, Value};

/// Format magic + version. The payload length follows as u32 LE.
pub(crate) const MAGIC: [u8; 4] = *b"MLPB";
pub(crate) const VERSION: u8 = 1;

/// Value tags (the first byte of each encoded value).
pub(crate) const TAG_SCALAR: u8 = 0;
pub(crate) const TAG_ARRAY: u8 = 1;
pub(crate) const TAG_STR: u8 = 2;
pub(crate) const TAG_STRLIST: u8 = 3;
pub(crate) const TAG_RECORD: u8 = 4;
pub(crate) const TAG_RESULT_OK: u8 = 5;
pub(crate) const TAG_RESULT_ERR: u8 = 6;

/// Encode a value into the native byte format, or error on a
/// non-data kind.
pub(crate) fn to_native(value: &Value) -> Result<Vec<u8>, EvalError> {
    let mut payload = Vec::new();
    encode_value(value, &mut payload)?;
    let mut out = Vec::with_capacity(payload.len() + 9);
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Encode one value (tag byte + payload), recursively.
pub(crate) fn encode_value(v: &Value, out: &mut Vec<u8>) -> Result<(), EvalError> {
    match v {
        Value::Array(a) if a.rank() == 0 => {
            out.push(TAG_SCALAR);
            out.extend_from_slice(&a.data()[0].to_le_bytes());
            Ok(())
        }
        Value::Array(a) => crate::native_encode_parts::encode_array(a, out),
        Value::Str(s) => {
            out.push(TAG_STR);
            push_str(out, s);
            Ok(())
        }
        Value::StrList { items } => crate::native_encode_parts::encode_strlist(items, out),
        Value::Record { fields } => crate::native_encode_parts::encode_record(fields, out),
        Value::Result { ok, payload } => {
            out.push(if *ok { TAG_RESULT_OK } else { TAG_RESULT_ERR });
            encode_value(payload, out)
        }
        other => Err(EvalError::Unsupported(format!(
            "to_native: cannot serialize a {} (only numbers, strings, arrays, string lists, \
             records, and results are native data)",
            mlpl_eval_types::value_kind(other)
        ))),
    }
}

/// Push a length-prefixed UTF-8 string (u32 LE len + bytes).
pub(crate) fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}
