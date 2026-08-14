//! Scalar marshaling across the C boundary. `ExtValue` <->
//! `AbiValue` for the V1 scalar set only (nil / bool / i64 / f64 /
//! utf8 / bytes). `DenseArray` and `NativeHandle` are the deferred
//! arrays-handles saga and produce a boundary error rather than
//! reading the union's pointer fields.
//!
//! Borrowed-span contract: an input `Utf8`/`Bytes` `AbiSlice`
//! points into the host-owned `ExtValue` argument, which outlives
//! the invoke call. An output `Utf8`/`Bytes` span is owned by the
//! provider and must stay valid until invoke returns; the host
//! copies it immediately here before returning.

use mlpl_extension_abi::{ExtError, ExtValue};

use crate::model::{AbiErrorV1, AbiSlice, AbiValue, ErrorCode, ValuePayload, ValueTag};

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

/// Marshal a provider-returned `AbiValue` back into an
/// `ExtValue`. Arrays and handles are not yet supported.
pub(crate) fn abi_to_ext(out: &AbiValue) -> Result<ExtValue, ExtError> {
    let value = match out.tag {
        t if t == ValueTag::Nil as u32 => ExtValue::Nil,
        t if t == ValueTag::Bool as u32 => ExtValue::Bool(unsafe { out.payload.boolean } != 0),
        t if t == ValueTag::I64 as u32 => ExtValue::I64(unsafe { out.payload.integer }),
        t if t == ValueTag::F64 as u32 => ExtValue::F64(unsafe { out.payload.float }),
        t if t == ValueTag::Utf8 as u32 => {
            let bytes = read_abi_slice("output string", unsafe { out.payload.slice })
                .map_err(ExtError::new)?;
            ExtValue::Str(
                String::from_utf8(bytes)
                    .map_err(|_| ExtError::new("output string is not valid UTF-8"))?,
            )
        }
        t if t == ValueTag::Bytes as u32 => ExtValue::Bytes(
            read_abi_slice("output bytes", unsafe { out.payload.slice }).map_err(ExtError::new)?,
        ),
        other => {
            return Err(ExtError::new(format!(
                "unsupported output value tag {other} (arrays and handles are not yet marshaled)"
            )));
        }
    };
    Ok(value)
}

/// Map a non-zero invoke status + filled `AbiErrorV1` into an
/// `ExtError`. A `Panic` code sets `panicked` so the host raises a
/// hard error rather than an `err(...)` Result.
pub(crate) fn abi_error_to_ext(err: &AbiErrorV1, status: u32) -> ExtError {
    let message = match read_abi_slice("error message", err.message) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(e) => e,
    };
    let message = if message.is_empty() {
        format!("extension failed with status {status}")
    } else {
        message
    };
    ExtError {
        message,
        panicked: err.code == ErrorCode::Panic as u32,
    }
}
