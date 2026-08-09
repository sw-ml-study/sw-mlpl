//! Typed-native binary decoder (`parse_native`): validate the
//! MLPB header (via `native_integrity`), recursively decode tagged
//! values enforcing the decode budget BEFORE unbounded recursion,
//! then verify the v2 CRC32 integrity trailer. Reads both v1 (no
//! checksum) and v2 buffers. Malformed / truncated / oversized /
//! too-deep / unknown-tag / checksum-mismatch input is an err
//! (never a panic). Container decoders live in
//! `native_decode_parts`.

use mlpl_array::DenseArray;

use crate::decode_limits::Limits;
use crate::native_cursor::{read_f64, read_str, read_u8};
use crate::native_encode::{
    TAG_ARRAY, TAG_RECORD, TAG_RESULT_ERR, TAG_RESULT_OK, TAG_SCALAR, TAG_STR, TAG_STRLIST,
};
use mlpl_eval_types::Value;

/// Decode a native byte buffer into a value, budget-bounded.
pub(crate) fn decode(bytes: &[u8], limits: &Limits) -> Result<Value, String> {
    if bytes.len() > limits.max_bytes {
        return Err(format!(
            "input of {} bytes exceeds max_bytes {}",
            bytes.len(),
            limits.max_bytes
        ));
    }
    let (version, payload_start, payload_len) = crate::native_integrity::read_header(bytes)?;
    let mut pos = payload_start;
    let v = decode_value(bytes, &mut pos, limits.max_depth)?;
    if pos != payload_start + payload_len {
        return Err(format!("trailing bytes after value at {pos}"));
    }
    crate::native_integrity::verify_checksum(bytes, version, payload_start, payload_len)?;
    crate::element_count::check(&v, limits.max_elements)?;
    Ok(v)
}

/// Decode one tagged value; `depth` is the remaining nesting
/// budget (a container opened at depth 0 exceeds the limit).
pub(crate) fn decode_value(bytes: &[u8], pos: &mut usize, depth: usize) -> Result<Value, String> {
    let tag = read_u8(bytes, pos)?;
    match tag {
        TAG_SCALAR => Ok(Value::Array(DenseArray::from_scalar(read_f64(bytes, pos)?))),
        TAG_STR => Ok(Value::Str(read_str(bytes, pos)?)),
        TAG_STRLIST => crate::native_decode_parts::decode_strlist(bytes, pos),
        TAG_ARRAY => {
            spend(depth, *pos).and_then(|d| crate::native_decode_parts::decode_array(bytes, pos, d))
        }
        TAG_RECORD => spend(depth, *pos)
            .and_then(|d| crate::native_decode_parts::decode_record(bytes, pos, d)),
        TAG_RESULT_OK | TAG_RESULT_ERR => spend(depth, *pos).and_then(|d| {
            Ok(Value::Result {
                ok: tag == TAG_RESULT_OK,
                payload: Box::new(decode_value(bytes, pos, d)?),
            })
        }),
        _ => Err(format!("unknown native tag {tag} at byte {}", *pos - 1)),
    }
}

/// Consume one unit of nesting depth, erroring when exhausted.
fn spend(depth: usize, pos: usize) -> Result<usize, String> {
    depth
        .checked_sub(1)
        .ok_or_else(|| format!("maximum nesting depth exceeded at byte {pos}"))
}
