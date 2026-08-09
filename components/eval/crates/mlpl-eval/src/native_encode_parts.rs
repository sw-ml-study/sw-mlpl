//! Container encoders for the native binary format (arrays,
//! string lists, records), split from `native_encode` to keep
//! each module within the function-count budget.

use std::collections::BTreeMap;

use mlpl_array::DenseArray;

use crate::native_encode::{TAG_ARRAY, TAG_RECORD, TAG_STRLIST, encode_value, push_str};
use mlpl_eval_types::{EvalError, Value};

/// `TAG_ARRAY` + rank (u8) + dims (u32 LE each) + row-major f64 LE
/// data. Rank-0 is encoded as a scalar by `encode_value`, so this
/// handles rank >= 1.
pub(crate) fn encode_array(a: &DenseArray, out: &mut Vec<u8>) -> Result<(), EvalError> {
    let dims = a.shape().dims();
    out.push(TAG_ARRAY);
    out.push(dims.len() as u8);
    for &d in dims {
        out.extend_from_slice(&(d as u32).to_le_bytes());
    }
    for &x in a.data() {
        out.extend_from_slice(&x.to_le_bytes());
    }
    Ok(())
}

/// `TAG_STRLIST` + count (u32 LE) + length-prefixed items.
pub(crate) fn encode_strlist(items: &[String], out: &mut Vec<u8>) -> Result<(), EvalError> {
    out.push(TAG_STRLIST);
    out.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for s in items {
        push_str(out, s);
    }
    Ok(())
}

/// `TAG_RECORD` + field count (u32 LE) + [key (length-prefixed) +
/// value (recursive)]*, in BTreeMap-sorted order (deterministic).
pub(crate) fn encode_record(
    fields: &BTreeMap<String, Value>,
    out: &mut Vec<u8>,
) -> Result<(), EvalError> {
    out.push(TAG_RECORD);
    out.extend_from_slice(&(fields.len() as u32).to_le_bytes());
    for (k, v) in fields {
        push_str(out, k);
        encode_value(v, out)?;
    }
    Ok(())
}
