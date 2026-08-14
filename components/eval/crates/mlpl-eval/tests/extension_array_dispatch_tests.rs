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
        let ok_array =
            n == 1 && (*a).tag == ValueTag::DenseArray as u32 && {
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
    assert_eq!(scalar(&mut env, "cabi_arr:sum_array([1, 2, 3, 4, 5, 6])"), 21.0);
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
