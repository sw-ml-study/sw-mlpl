//! The RECEIVE direction: a provider-returned `AbiValue` -> `ExtValue`
//! (scalars, native handle, and dense arrays), and the invoke-error
//! decode. A returned array's shape/data are provider-owned, so they
//! are COPIED here immediately (the borrowed-span contract) and
//! validated against dtype / rank / element count. A native handle is
//! a fixed-size payload read inline by `scalar_from_abi`.

use mlpl_extension_abi::{ExtDtype, ExtError, ExtHandle, ExtValue};

use crate::marshal::read_abi_slice;
use crate::model::{AbiArrayView, AbiErrorV1, AbiValue, ErrorCode, ValueTag};

/// Highest supported rank (mirrors the send path).
const MAX_RANK: usize = 8;

/// Marshal a provider-returned value back into an `ExtValue`.
pub(crate) fn abi_to_ext(out: &AbiValue) -> Result<ExtValue, ExtError> {
    if let Some(scalar) = scalar_from_abi(out) {
        return Ok(scalar);
    }
    let value = match out.tag {
        t if t == ValueTag::Utf8 as u32 => {
            let raw = read_abi_slice("output string", unsafe { out.payload.slice });
            ExtValue::Str(
                String::from_utf8(raw.map_err(ExtError::new)?)
                    .map_err(|_| ExtError::new("output string is not valid UTF-8"))?,
            )
        }
        t if t == ValueTag::Bytes as u32 => ExtValue::Bytes(
            read_abi_slice("output bytes", unsafe { out.payload.slice }).map_err(ExtError::new)?,
        ),
        t if t == ValueTag::DenseArray as u32 => {
            unsafe { read_array_view(out.payload.array) }.map_err(ExtError::new)?
        }
        other => {
            return Err(ExtError::new(format!(
                "unsupported output value tag {other} (handles are not yet marshaled)"
            )));
        }
    };
    Ok(value)
}

/// The fixed-size outputs (nil/bool/i64/f64 + native handle), or
/// `None` for the variable-length tags handled by `abi_to_ext`.
fn scalar_from_abi(out: &AbiValue) -> Option<ExtValue> {
    Some(match out.tag {
        t if t == ValueTag::Nil as u32 => ExtValue::Nil,
        t if t == ValueTag::Bool as u32 => ExtValue::Bool(unsafe { out.payload.boolean } != 0),
        t if t == ValueTag::I64 as u32 => ExtValue::I64(unsafe { out.payload.integer }),
        t if t == ValueTag::F64 as u32 => ExtValue::F64(unsafe { out.payload.float }),
        t if t == ValueTag::NativeHandle as u32 => {
            let h = unsafe { out.payload.handle };
            ExtValue::Handle(ExtHandle {
                extension_id: h.extension_id,
                type_id: h.type_id,
                slot: h.slot,
                generation: h.generation,
            })
        }
        _ => return None,
    })
}

/// Copy + decode a provider-returned `AbiArrayView` into an
/// `ExtValue::Array`.
///
/// # Safety
/// `view` must be null or a valid `AbiArrayView` whose `shape`
/// (`rank` usizes) and `data` slice are live for this call.
pub(crate) unsafe fn read_array_view(view: *const AbiArrayView) -> Result<ExtValue, String> {
    if view.is_null() {
        return Err("returned array view is null".to_string());
    }
    let view = unsafe { &*view };
    let dtype = ExtDtype::from_wire_tag(view.dtype)
        .ok_or_else(|| format!("returned array has unknown dtype {}", view.dtype))?;
    let rank = view.rank as usize;
    if rank == 0 || rank > MAX_RANK || view.shape.is_null() {
        return Err(format!(
            "returned array rank {rank} unsupported (1..={MAX_RANK})"
        ));
    }
    let shape: Vec<usize> = unsafe { std::slice::from_raw_parts(view.shape, rank) }.to_vec();
    let bytes = read_abi_slice("returned array data", view.data)?;
    let elems: usize = shape.iter().product();
    if bytes.len() != elems.saturating_mul(dtype.width()) {
        return Err(format!(
            "returned array {} bytes != shape {shape:?}",
            bytes.len()
        ));
    }
    let data = dtype.decode_le(&bytes);
    Ok(ExtValue::Array { dtype, shape, data })
}

/// Map a non-zero invoke status + filled `AbiErrorV1` into an
/// `ExtError`. A `Panic` code sets `panicked` so the host raises a hard
/// error rather than an `err(...)` Result.
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
