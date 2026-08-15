//! Scalar marshaling for the SEND direction: an `ExtValue` scalar
//! (nil / bool / i64 / f64 / utf8 / bytes) -> `AbiValue`, plus the
//! shared `read_abi_slice` helper. Dense-array arguments go through
//! `marshal_array` (send) and the RECEIVE direction through
//! `marshal_array_out`.
//!
//! Borrowed-span contract: an input `Utf8`/`Bytes` `AbiSlice` points
//! into the host-owned `ExtValue` argument, which outlives the invoke
//! call.

use mlpl_extension_abi::ExtValue;

use crate::model::{AbiHandle, AbiSlice, AbiValue, ValuePayload, ValueTag};

/// Largest byte span the host will read from a boundary slice.
const MAX_SLICE_BYTES: usize = 16 * 1024 * 1024;

/// Copy a provider-supplied byte span into an owned `Vec`, with
/// null and length guards. An empty span (`len == 0`) is `[]`
/// regardless of `data`.
pub(crate) fn read_abi_slice(label: &str, slice: AbiSlice) -> Result<Vec<u8>, String> {
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    if slice.data.is_null() {
        return Err(format!("{label}: null data with non-zero length"));
    }
    if slice.len > MAX_SLICE_BYTES {
        return Err(format!("{label}: {} bytes exceeds cap", slice.len));
    }
    Ok(unsafe { std::slice::from_raw_parts(slice.data, slice.len) }.to_vec())
}

/// Marshal one host argument into an `AbiValue`. Fixed-size kinds
/// go through `scalar_payload`, borrowed spans through
/// `slice_payload`; `Array` routes through `marshal_array` instead,
/// so an unmatched value falls back to `Nil` defensively.
pub(crate) fn ext_to_abi(value: &ExtValue) -> AbiValue {
    let (tag, payload) = scalar_payload(value)
        .or_else(|| slice_payload(value))
        .unwrap_or((ValueTag::Nil, ValuePayload { integer: 0 }));
    AbiValue {
        tag: tag as u32,
        reserved: 0,
        payload,
    }
}

/// The fixed-size (non-pointer) payloads -- scalars and the native
/// handle -- or `None` for the borrowed-span / array kinds.
fn scalar_payload(value: &ExtValue) -> Option<(ValueTag, ValuePayload)> {
    Some(match value {
        ExtValue::Nil => (ValueTag::Nil, ValuePayload { integer: 0 }),
        ExtValue::Bool(b) => (
            ValueTag::Bool,
            ValuePayload {
                boolean: u8::from(*b),
            },
        ),
        ExtValue::I64(i) => (ValueTag::I64, ValuePayload { integer: *i }),
        ExtValue::F64(f) => (ValueTag::F64, ValuePayload { float: *f }),
        ExtValue::Handle(h) => (
            ValueTag::NativeHandle,
            ValuePayload {
                handle: AbiHandle {
                    extension_id: h.extension_id,
                    type_id: h.type_id,
                    slot: h.slot,
                    generation: h.generation,
                },
            },
        ),
        _ => return None,
    })
}

/// The borrowed-span payloads (`Str`/`Bytes`): a slice pointing
/// into the host-owned argument, valid for the invoke call. `None`
/// for the fixed-size and array kinds handled elsewhere.
fn slice_payload(value: &ExtValue) -> Option<(ValueTag, ValuePayload)> {
    let (tag, bytes): (ValueTag, &[u8]) = match value {
        ExtValue::Str(s) => (ValueTag::Utf8, s.as_bytes()),
        ExtValue::Bytes(b) => (ValueTag::Bytes, b.as_slice()),
        _ => return None,
    };
    Some((
        tag,
        ValuePayload {
            slice: AbiSlice {
                data: bytes.as_ptr(),
                len: bytes.len(),
            },
        },
    ))
}

// The RECEIVE direction (provider output -> `ExtValue`) and the
// invoke-error decode live in `marshal_array_out`.
