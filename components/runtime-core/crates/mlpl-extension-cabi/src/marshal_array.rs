//! Marshal dense-array ARGUMENTS across the C boundary (the send
//! direction, for e.g. `native3d:set_lines`). A `Value::Array` arrives
//! as an `ExtValue::Array` of f64; it is encoded to the wire dtype (via
//! `ExtDtype`) and wrapped in an `AbiArrayView` whose backing buffers
//! are OWNED and heap-stable, so they outlive the invoke call. Reading
//! arrays RETURNED by a provider is the next step.

use mlpl_extension_abi::{ExtDtype, ExtValue};

use crate::marshal::ext_to_abi;
use crate::model::{
    AbiArrayView, AbiField, AbiRecordView, AbiSlice, AbiValue, ValuePayload, ValueTag,
};

/// Largest array the host will hand across the boundary.
const MAX_ARRAY_BYTES: usize = 16 * 1024 * 1024;
/// Highest supported rank.
const MAX_RANK: usize = 8;
/// Largest record accepted at any one nesting level.
const MAX_FIELDS: usize = 1024;
/// Maximum nesting depth, including the root value.
const MAX_DEPTH: usize = 64;

/// Marshaled arguments for one invoke call: the `AbiValue` array plus
/// the owned backing for any dense-array arguments (kept alive here).
pub(crate) struct AbiArgs {
    pub(crate) values: Vec<AbiValue>,
    _holders: Vec<ValueHolder>,
}

/// Heap-stable backing retained for a recursively marshaled value.
enum ValueHolder {
    Array { _holder: ArrayHolder },
    Record { _holder: RecordHolder },
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

/// Owned backing for a record argument. Child holders keep nested
/// array/record pointers live; boxed fields and view are themselves
/// stable while the provider call runs.
struct RecordHolder {
    _children: Vec<ValueHolder>,
    _fields: Box<[AbiField]>,
    view: Box<AbiRecordView>,
}

/// Marshal every argument, routing dense arrays to owned holders and
/// scalars/strings through the existing `ext_to_abi`.
pub(crate) fn marshal_args(args: &[ExtValue]) -> Result<AbiArgs, String> {
    let mut values = Vec::with_capacity(args.len());
    let mut holders = Vec::new();
    for a in args {
        let (value, holder) = marshal_value(a, 1)?;
        values.push(value);
        if let Some(holder) = holder {
            holders.push(holder);
        }
    }
    Ok(AbiArgs {
        values,
        _holders: holders,
    })
}

/// Recursively marshal one value and return any backing which must
/// remain alive for the provider invocation.
fn marshal_value(
    value: &ExtValue,
    depth: usize,
) -> Result<(AbiValue, Option<ValueHolder>), String> {
    if depth > MAX_DEPTH {
        return Err(format!("record nesting exceeds boundary cap {MAX_DEPTH}"));
    }
    match value {
        ExtValue::Array { dtype, shape, data } => {
            let holder = build_holder(*dtype, shape, data)?;
            let abi = AbiValue {
                tag: ValueTag::DenseArray as u32,
                reserved: 0,
                payload: ValuePayload {
                    array: &*holder.view,
                },
            };
            Ok((abi, Some(ValueHolder::Array { _holder: holder })))
        }
        ExtValue::Record(fields) => marshal_record(fields, depth),
        other => Ok((ext_to_abi(other), None)),
    }
}

/// Build a stable record view while retaining every nested holder.
fn marshal_record(
    fields: &[(String, ExtValue)],
    depth: usize,
) -> Result<(AbiValue, Option<ValueHolder>), String> {
    if fields.len() > MAX_FIELDS {
        return Err(format!(
            "record has {} fields (max {MAX_FIELDS})",
            fields.len()
        ));
    }
    let (abi_fields, children) = marshal_fields(fields, depth)?;
    let abi_fields = abi_fields.into_boxed_slice();
    let view = Box::new(AbiRecordView {
        fields: abi_fields.as_ptr(),
        field_count: abi_fields.len(),
    });
    let holder = RecordHolder {
        _children: children,
        _fields: abi_fields,
        view,
    };
    let abi = AbiValue::record(&*holder.view);
    Ok((abi, Some(ValueHolder::Record { _holder: holder })))
}

/// Marshal record fields and collect backing for nested values.
fn marshal_fields(
    fields: &[(String, ExtValue)],
    depth: usize,
) -> Result<(Vec<AbiField>, Vec<ValueHolder>), String> {
    let mut children = Vec::new();
    let mut abi_fields = Vec::with_capacity(fields.len());
    for (name, value) in fields {
        let (value, holder) = marshal_value(value, depth + 1)?;
        children.extend(holder);
        abi_fields.push(AbiField {
            name: AbiSlice {
                data: name.as_ptr(),
                len: name.len(),
            },
            value,
        });
    }
    Ok((abi_fields, children))
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
