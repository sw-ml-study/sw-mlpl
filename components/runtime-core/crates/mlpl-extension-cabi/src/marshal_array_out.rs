//! The RECEIVE direction: a provider-returned `AbiValue` -> `ExtValue`
//! (scalars, native handle, dense arrays, and records), and the
//! invoke-error decode. A returned array's shape/data and a record's
//! fields are provider-owned, so they are COPIED here immediately (the
//! borrowed-span contract); arrays are validated against dtype / rank /
//! element count, records recurse field-by-field. A native handle is a
//! fixed-size payload read inline by `scalar_from_abi`.

use mlpl_extension_abi::{ExtDtype, ExtError, ExtHandle, ExtValue};

use crate::marshal::read_abi_slice;
use crate::model::{AbiArrayView, AbiRecordView, AbiValue, ValueTag};

/// Highest supported rank (mirrors the send path).
const MAX_RANK: usize = 8;

/// Largest field count the host will read from a returned record.
const MAX_FIELDS: usize = 1024;

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
        t if t == ValueTag::Record as u32 => unsafe { read_record_view(out.payload.record) }?,
        other => {
            return Err(ExtError::new(format!(
                "unsupported output value tag {other}"
            )));
        }
    };
    Ok(value)
}

/// Copy a provider-returned `AbiRecordView` (named fields) into an
/// `ExtValue::Record`, recursing into each field value.
///
/// # Safety
/// `view` must be null or a valid `AbiRecordView` whose `fields`
/// array (`field_count` entries) is live for this call.
pub(crate) unsafe fn read_record_view(view: *const AbiRecordView) -> Result<ExtValue, ExtError> {
    if view.is_null() {
        return Err(ExtError::new("returned record view is null"));
    }
    let view = unsafe { &*view };
    if view.field_count > MAX_FIELDS {
        return Err(ExtError::new(format!(
            "returned record has {} fields (max {MAX_FIELDS})",
            view.field_count
        )));
    }
    let raw = if view.field_count == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(view.fields, view.field_count) }
    };
    let mut fields = Vec::with_capacity(raw.len());
    for f in raw {
        let raw_name = read_abi_slice("record field name", f.name).map_err(ExtError::new)?;
        let name = String::from_utf8(raw_name)
            .map_err(|_| ExtError::new("record field name is not UTF-8"))?;
        fields.push((name, abi_to_ext(&f.value)?));
    }
    Ok(ExtValue::Record(fields))
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
