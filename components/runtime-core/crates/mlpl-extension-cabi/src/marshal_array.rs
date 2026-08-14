//! Marshal dense-array ARGUMENTS across the C boundary (the send
//! direction, for e.g. `native3d:set_lines`). A `Value::Array` arrives
//! as an `ExtValue::Array` of f64; it is encoded to the wire dtype (via
//! `ExtDtype`) and wrapped in an `AbiArrayView` whose backing buffers
//! are OWNED and heap-stable, so they outlive the invoke call. Reading
//! arrays RETURNED by a provider is the next step.

use mlpl_extension_abi::{ExtDtype, ExtValue};

use crate::marshal::ext_to_abi;
use crate::model::{AbiArrayView, AbiSlice, AbiValue, ValuePayload, ValueTag};

/// Largest array the host will hand across the boundary.
const MAX_ARRAY_BYTES: usize = 16 * 1024 * 1024;
/// Highest supported rank.
const MAX_RANK: usize = 8;

/// Marshaled arguments for one invoke call: the `AbiValue` array plus
/// the owned backing for any dense-array arguments (kept alive here).
pub(crate) struct AbiArgs {
    pub(crate) values: Vec<AbiValue>,
    _holders: Vec<ArrayHolder>,
}

/// Owned backing for one dense-array argument. All fields are heap
/// boxes, so `view`'s pointers into them stay valid as the holder Vec
/// grows or a holder moves.
struct ArrayHolder {
    _data: Box<[u8]>,
    _shape: Box<[usize]>,
    _strides: Box<[isize]>,
    view: Box<AbiArrayView>,
}

/// Marshal every argument, routing dense arrays to owned holders and
/// scalars/strings through the existing `ext_to_abi`.
pub(crate) fn marshal_args(args: &[ExtValue]) -> Result<AbiArgs, String> {
    let mut values = Vec::with_capacity(args.len());
    let mut holders = Vec::new();
    for a in args {
        if let ExtValue::Array { dtype, shape, data } = a {
            let holder = build_holder(*dtype, shape, data)?;
            values.push(AbiValue {
                tag: ValueTag::DenseArray as u32,
                reserved: 0,
                payload: ValuePayload {
                    array: &*holder.view,
                },
            });
            holders.push(holder);
        } else {
            values.push(ext_to_abi(a));
        }
    }
    Ok(AbiArgs {
        values,
        _holders: holders,
    })
}

/// Validate one array's rank, element count, and total byte size.
fn check_dims(dtype: ExtDtype, shape: &[usize], data: &[f64]) -> Result<(), String> {
    if shape.is_empty() || shape.len() > MAX_RANK {
        return Err(format!(
            "array rank {} unsupported (1..={MAX_RANK})",
            shape.len()
        ));
    }
    let elems: usize = shape.iter().product();
    if elems != data.len() {
        return Err(format!(
            "array shape {shape:?} needs {elems} elements, got {}",
            data.len()
        ));
    }
    match elems.checked_mul(dtype.width()) {
        Some(t) if t <= MAX_ARRAY_BYTES => Ok(()),
        _ => Err("array size exceeds the boundary cap".to_string()),
    }
}

/// Validate, then build the owned backing + `AbiArrayView` for one
/// array (the dtype does its own encoding + strides).
fn build_holder(dtype: ExtDtype, shape: &[usize], data: &[f64]) -> Result<ArrayHolder, String> {
    check_dims(dtype, shape, data)?;
    Ok(assemble(
        dtype,
        dtype.encode_le(data).into_boxed_slice(),
        shape.to_vec().into_boxed_slice(),
        dtype.byte_strides(shape).into_boxed_slice(),
    ))
}

/// Assemble the `AbiArrayView` (pointing into the owned boxes) and its
/// holder. Pure -- no validation, no borrowing across the boundary.
#[allow(clippy::cast_possible_truncation)]
fn assemble(
    dtype: ExtDtype,
    data: Box<[u8]>,
    shape: Box<[usize]>,
    strides: Box<[isize]>,
) -> ArrayHolder {
    let view = Box::new(AbiArrayView {
        dtype: dtype.wire_tag(),
        rank: shape.len() as u32,
        data: AbiSlice {
            data: data.as_ptr(),
            len: data.len(),
        },
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
    });
    ArrayHolder {
        _data: data,
        _shape: shape,
        _strides: strides,
        view,
    }
}
