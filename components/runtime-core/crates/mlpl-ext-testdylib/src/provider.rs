//! The fixture's exported functions. `answer` returns a scalar; `sum`
//! takes a dense f64 array IN -- proving a non-scalar value crosses the
//! `dlopen` edge, not just a scalar. (Record/handle RETURNS use the
//! same host marshaling, proven by the static extension tests.)

use std::mem::size_of;

use mlpl_extension_cabi::{
    AbiArrayView, AbiErrorV1, AbiSlice, AbiValue, ErrorCode, ValuePayload, ValueTag,
};

use crate::ANSWER;

/// A borrowed span over a `'static` byte string.
pub(crate) fn slice(bytes: &'static [u8]) -> AbiSlice {
    AbiSlice {
        data: bytes.as_ptr(),
        len: bytes.len(),
    }
}

/// `answer() -> 42.0`.
pub(crate) unsafe extern "C" fn inv_answer(
    _a: *const AbiValue,
    _n: usize,
    out: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    unsafe {
        *out = AbiValue {
            tag: ValueTag::F64 as u32,
            reserved: 0,
            payload: ValuePayload { float: ANSWER },
        };
    }
    ErrorCode::Ok as u32
}

/// `sum(array) -> f64` -- a dense f64 array crosses IN through the
/// dlopen'd provider and its total comes back out.
pub(crate) unsafe extern "C" fn inv_sum(
    a: *const AbiValue,
    n: usize,
    out: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    unsafe {
        if n != 1 || (*a).tag != ValueTag::DenseArray as u32 {
            return ErrorCode::ExtensionFailure as u32;
        }
        let view: &AbiArrayView = &*(*a).payload.array;
        let elems = view.data.len / size_of::<f64>();
        let data = std::slice::from_raw_parts(view.data.data.cast::<f64>(), elems);
        *out = AbiValue {
            tag: ValueTag::F64 as u32,
            reserved: 0,
            payload: ValuePayload {
                float: data.iter().sum(),
            },
        };
    }
    ErrorCode::Ok as u32
}
