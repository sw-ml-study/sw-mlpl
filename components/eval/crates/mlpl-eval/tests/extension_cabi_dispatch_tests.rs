//! A `#[repr(C)]` provider (registered via the C-ABI adapter)
//! dispatches through the interpreter exactly like a safe static
//! provider: `cabi_ns:answer()` marshals across the C boundary and
//! yields 42; a non-zero status becomes an err Result.

use std::mem::size_of;
use std::ptr;

use mlpl_eval::{Environment, Value};
use mlpl_extension_cabi::{
    ABI_VERSION_V1, AbiErrorV1, AbiSlice, AbiValue, ErrorCode, ExtensionDescriptorV1,
    FunctionDescriptorV1, ValuePayload, ValueTag, register_c_extension,
};

static VERSION: &[u8] = b"0.1.0";
static FAIL_MSG: &[u8] = b"c boundary failure";

unsafe extern "C" fn inv_answer(
    _a: *const AbiValue,
    _n: usize,
    output: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    unsafe {
        *output = AbiValue {
            tag: ValueTag::I64 as u32,
            reserved: 0,
            payload: ValuePayload { integer: 42 },
        };
    }
    ErrorCode::Ok as u32
}

unsafe extern "C" fn inv_fail(
    _a: *const AbiValue,
    _n: usize,
    _o: *mut AbiValue,
    err: *mut AbiErrorV1,
) -> u32 {
    unsafe {
        *err = AbiErrorV1 {
            code: ErrorCode::ExtensionFailure as u32,
            reserved: 0,
            message: AbiSlice {
                data: FAIL_MSG.as_ptr(),
                len: FAIL_MSG.len(),
            },
        };
    }
    ErrorCode::ExtensionFailure as u32
}

fn register_c_provider() {
    let functions = [
        FunctionDescriptorV1 {
            name: AbiSlice {
                data: b"answer".as_ptr(),
                len: 6,
            },
            arity: 0,
            reserved: 0,
            invoke: Some(inv_answer),
        },
        FunctionDescriptorV1 {
            name: AbiSlice {
                data: b"fail".as_ptr(),
                len: 4,
            },
            arity: 0,
            reserved: 0,
            invoke: Some(inv_fail),
        },
    ];
    let name = b"cabi_demo";
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
    // Idempotent across the tests in this binary: a duplicate
    // registration is fine to ignore.
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
fn c_extension_call_returns_its_value_through_the_interpreter() {
    register_c_provider();
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "cabi_demo:answer()"), 42.0);
}

#[test]
fn c_extension_failure_is_an_err_result() {
    register_c_provider();
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "is_ok(cabi_demo:fail())"), 0.0);
    assert_eq!(scalar(&mut env, "is_ok(ok(cabi_demo:answer()))"), 1.0);
}
