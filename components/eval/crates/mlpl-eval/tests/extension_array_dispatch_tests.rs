//! A `#[repr(C)]` provider receives a DENSE ARRAY argument across the
//! C boundary (the send direction, for e.g. `native3d:set_lines`):
//! MLPL sends an `f64` array, the provider reads the `AbiArrayView`
//! and sums it. Rank>8 arrays are rejected before the call (an err
//! Result, not a crash).

use std::mem::size_of;
use std::ptr;

use mlpl_eval::{Environment, Value};
use mlpl_extension_cabi::{
    ABI_VERSION_V1, AbiArrayView, AbiErrorV1, AbiSlice, AbiValue, DTypeTag, ErrorCode,
    ExtensionDescriptorV1, FunctionDescriptorV1, ValuePayload, ValueTag, register_c_extension,
};

static VERSION: &[u8] = b"0.1.0";
static NOT_ARRAY: &[u8] = b"sum_array: argument is not an f64 array";

/// Sum a rank>=1 f64 `AbiArrayView` argument.
unsafe extern "C" fn inv_sum_array(
    a: *const AbiValue,
    n: usize,
    output: *mut AbiValue,
    err: *mut AbiErrorV1,
) -> u32 {
    unsafe {
        let ok_array = n == 1 && (*a).tag == ValueTag::DenseArray as u32 && {
            let view = &*(*a).payload.array;
            view.dtype == DTypeTag::F64 as u32
        };
        if !ok_array {
            *err = AbiErrorV1 {
                code: ErrorCode::ExtensionFailure as u32,
                reserved: 0,
                message: AbiSlice {
                    data: NOT_ARRAY.as_ptr(),
                    len: NOT_ARRAY.len(),
                },
            };
            return ErrorCode::ExtensionFailure as u32;
        }
        let view: &AbiArrayView = &*(*a).payload.array;
        let elems = view.data.len / size_of::<f64>();
        let data = std::slice::from_raw_parts(view.data.data.cast::<f64>(), elems);
        let sum: f64 = data.iter().sum();
        *output = AbiValue {
            tag: ValueTag::F64 as u32,
            reserved: 0,
            payload: ValuePayload { float: sum },
        };
    }
    ErrorCode::Ok as u32
}

fn register_c_provider() {
    let functions = [FunctionDescriptorV1 {
        name: AbiSlice {
            data: b"sum_array".as_ptr(),
            len: 9,
        },
        arity: 1,
        reserved: 0,
        invoke: Some(inv_sum_array),
    }];
    let name = b"cabi_arr";
    let d = ExtensionDescriptorV1 {
        struct_size: size_of::<ExtensionDescriptorV1>() as u32,
        abi_version: ABI_VERSION_V1,
        name: AbiSlice {
            data: name.as_ptr(),
            len: name.len(),
        },
        version: AbiSlice {
            data: VERSION.as_ptr(),
            len: VERSION.len(),
        },
        functions: functions.as_ptr(),
        function_count: functions.len(),
        metadata: AbiSlice {
            data: ptr::null(),
            len: 0,
        },
    };
    let _ = unsafe { register_c_extension(&d) };
}

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(env: &mut Environment, src: &str) -> f64 {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar from {src}, got {other:?}"),
    }
}

#[test]
fn a_rank1_array_argument_crosses_the_boundary() {
    register_c_provider();
    let mut env = Environment::new();
    assert_eq!(
        scalar(&mut env, "cabi_arr:sum_array([1, 2, 3, 4, 5, 6])"),
        21.0
    );
}

#[test]
fn a_rank2_array_argument_crosses_the_boundary() {
    register_c_provider();
    let mut env = Environment::new();
    // reshape [1..6] -> [2,3]; the flat f64 data still sums to 21.
    assert_eq!(
        scalar(&mut env, "cabi_arr:sum_array(reshape(iota(6) + 1, [2, 3]))"),
        21.0
    );
}

#[test]
fn an_over_rank_array_is_rejected_as_an_err() {
    register_c_provider();
    let mut env = Environment::new();
    // rank 9 (> the 8-D cap) is refused before the call -> err Result.
    let src = "is_ok(cabi_arr:sum_array(reshape(iota(1), [1,1,1,1,1,1,1,1,1])))";
    assert_eq!(scalar(&mut env, src), 0.0);
}

// ----- Return direction: a provider RETURNS a dense array. -----

static RANGE_DATA: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
static RANGE_SHAPE: [usize; 2] = [2, 3];
static RANGE_STRIDES: [isize; 2] = [24, 8];
// Same shape [2,3] (6 elems) but only 4 f64 of data -> a mismatch.
static BAD_DATA: [f64; 4] = [0.0, 1.0, 2.0, 3.0];

/// Return a well-formed f64 [2,3] array of 0..6.
unsafe extern "C" fn inv_make_range(
    _a: *const AbiValue,
    _n: usize,
    output: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    // Leak the view so it outlives the call; the host copies immediately.
    let view: &'static AbiArrayView = Box::leak(Box::new(AbiArrayView {
        dtype: DTypeTag::F64 as u32,
        rank: 2,
        data: AbiSlice {
            data: RANGE_DATA.as_ptr().cast::<u8>(),
            len: 48,
        },
        shape: RANGE_SHAPE.as_ptr(),
        strides: RANGE_STRIDES.as_ptr(),
    }));
    unsafe {
        *output = AbiValue {
            tag: ValueTag::DenseArray as u32,
            reserved: 0,
            payload: ValuePayload { array: view },
        };
    }
    ErrorCode::Ok as u32
}

/// Return a view whose declared shape [2,3] (6 elems) does not match
/// its 4-element (32-byte) data -> the host rejects it.
unsafe extern "C" fn inv_make_bad(
    _a: *const AbiValue,
    _n: usize,
    output: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    let view: &'static AbiArrayView = Box::leak(Box::new(AbiArrayView {
        dtype: DTypeTag::F64 as u32,
        rank: 2,
        data: AbiSlice {
            data: BAD_DATA.as_ptr().cast::<u8>(),
            len: 32,
        },
        shape: RANGE_SHAPE.as_ptr(),
        strides: RANGE_STRIDES.as_ptr(),
    }));
    unsafe {
        *output = AbiValue {
            tag: ValueTag::DenseArray as u32,
            reserved: 0,
            payload: ValuePayload { array: view },
        };
    }
    ErrorCode::Ok as u32
}

fn register_out_provider() {
    let functions = [
        FunctionDescriptorV1 {
            name: AbiSlice {
                data: b"make_range".as_ptr(),
                len: 10,
            },
            arity: 0,
            reserved: 0,
            invoke: Some(inv_make_range),
        },
        FunctionDescriptorV1 {
            name: AbiSlice {
                data: b"make_bad".as_ptr(),
                len: 8,
            },
            arity: 0,
            reserved: 0,
            invoke: Some(inv_make_bad),
        },
    ];
    let name = b"cabi_out";
    let d = ExtensionDescriptorV1 {
        struct_size: size_of::<ExtensionDescriptorV1>() as u32,
        abi_version: ABI_VERSION_V1,
        name: AbiSlice {
            data: name.as_ptr(),
            len: name.len(),
        },
        version: AbiSlice {
            data: VERSION.as_ptr(),
            len: VERSION.len(),
        },
        functions: functions.as_ptr(),
        function_count: functions.len(),
        metadata: AbiSlice {
            data: ptr::null(),
            len: 0,
        },
    };
    let _ = unsafe { register_c_extension(&d) };
}

#[test]
fn a_returned_array_round_trips_to_mlpl() {
    register_out_provider();
    let mut env = Environment::new();
    // returns [[0,1,2],[3,4,5]]: values sum to 15, shape is [2,3] (2+3).
    assert_eq!(scalar(&mut env, "reduce_add(cabi_out:make_range())"), 15.0);
    assert_eq!(
        scalar(&mut env, "reduce_add(shape(cabi_out:make_range()))"),
        5.0
    );
}

#[test]
fn a_shape_mismatched_returned_array_is_an_err() {
    register_out_provider();
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "is_ok(cabi_out:make_bad())"), 0.0);
}
