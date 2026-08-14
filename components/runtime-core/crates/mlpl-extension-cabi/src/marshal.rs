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

use crate::model::{AbiSlice, AbiValue, ValuePayload, ValueTag};

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

/// Marshal one host argument into an `AbiValue`. `Str`/`Bytes`
/// produce a slice borrowing the argument (valid for the call).
pub(crate) fn ext_to_abi(value: &ExtValue) -> AbiValue {
    let (tag, payload) = match value {
        ExtValue::Nil => (ValueTag::Nil, ValuePayload { integer: 0 }),
        ExtValue::Bool(b) => (
            ValueTag::Bool,
            ValuePayload {
                boolean: u8::from(*b),
            },
        ),
        ExtValue::I64(i) => (ValueTag::I64, ValuePayload { integer: *i }),
        ExtValue::F64(f) => (ValueTag::F64, ValuePayload { float: *f }),
        ExtValue::Str(s) => (
            ValueTag::Utf8,
            ValuePayload {
                slice: AbiSlice {
                    data: s.as_ptr(),
                    len: s.len(),
                },
            },
        ),
        ExtValue::Bytes(b) => (
            ValueTag::Bytes,
            ValuePayload {
                slice: AbiSlice {
                    data: b.as_ptr(),
                    len: b.len(),
                },
            },
        ),
        // Arrays go through `marshal_array::marshal_args`; unreachable.
        ExtValue::Array { .. } => (ValueTag::Nil, ValuePayload { integer: 0 }),
    };
    AbiValue {
        tag: tag as u32,
        reserved: 0,
        payload,
    }
}

// The RECEIVE direction (provider output -> `ExtValue`) and the
// invoke-error decode live in `marshal_array_out`.
