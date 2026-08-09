//! Container decoders for the native binary format (arrays,
//! string lists, records), split from `native_decode` to keep
//! each module within the function-count budget. Every length is
//! validated against the remaining input by the cursor reads, so a
//! corrupt count cannot over-allocate silently.

use std::collections::BTreeMap;

use mlpl_array::{DenseArray, Shape};

use crate::native_cursor::{read_f64, read_str, read_u8, read_u32};
use crate::native_decode::decode_value;
use mlpl_eval_types::Value;

/// Decode `TAG_ARRAY` (already consumed): rank + dims + row-major
/// f64 data. Rebuilds an exact-shape array.
pub(crate) fn decode_array(bytes: &[u8], pos: &mut usize, _depth: usize) -> Result<Value, String> {
    let rank = read_u8(bytes, pos)? as usize;
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(read_u32(bytes, pos)? as usize);
    }
    let count: usize = dims.iter().product();
    let mut data = Vec::with_capacity(count.min(bytes.len().saturating_sub(*pos) / 8 + 1));
    for _ in 0..count {
        data.push(read_f64(bytes, pos)?);
    }
    DenseArray::new(Shape::new(dims), data)
        .map(Value::Array)
        .map_err(|e| format!("native array: {e}"))
}

/// Decode `TAG_STRLIST` (already consumed): count + items.
pub(crate) fn decode_strlist(bytes: &[u8], pos: &mut usize) -> Result<Value, String> {
    let count = read_u32(bytes, pos)? as usize;
    let mut items = Vec::with_capacity(count.min(bytes.len().saturating_sub(*pos) / 4 + 1));
    for _ in 0..count {
        items.push(read_str(bytes, pos)?);
    }
    Ok(Value::StrList { items })
}

/// Decode `TAG_RECORD` (already consumed): field count + [key +
/// value]*; values recurse with the given depth budget.
pub(crate) fn decode_record(bytes: &[u8], pos: &mut usize, depth: usize) -> Result<Value, String> {
    let count = read_u32(bytes, pos)? as usize;
    let mut fields = BTreeMap::new();
    for _ in 0..count {
        let key = read_str(bytes, pos)?;
        let value = decode_value(bytes, pos, depth)?;
        fields.insert(key, value);
    }
    Ok(Value::Record { fields })
}
