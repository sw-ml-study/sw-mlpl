//! Native handles (B5, the spine root): a `#[repr(C)]` provider
//! RETURNS an opaque handle from `make_viewer`, MLPL stores it in a
//! variable and passes it back to `use_viewer`, which recovers the
//! same resource. A stale (generation-bumped) or foreign
//! (cross-extension) handle is a clean `err` Result, not a crash.
//! MLPL cannot forge a handle -- there is no numeric constructor for
//! one, so the only source is a provider return.

use std::mem::size_of;
use std::ptr;

use mlpl_eval::{Environment, Value};
use mlpl_extension_cabi::{
    ABI_VERSION_V1, AbiErrorV1, AbiHandle, AbiSlice, AbiValue, ErrorCode, ExtensionDescriptorV1,
    FunctionDescriptorV1, ValuePayload, ValueTag, register_c_extension,
};

static VERSION: &[u8] = b"0.1.0";
static BAD_HANDLE: &[u8] = b"use_viewer: stale or foreign handle";

const VIEWERS_EXT: u64 = 0xA11CE;
const OTHERS_EXT: u64 = 0xB0B;
const VIEWER_TYPE: u64 = 1;
const LIVE_GEN: u32 = 1;
const STORED: f64 = 42.0; // the "resource" behind viewer slot 0

/// Build an `AbiValue` carrying a native handle output.
fn handle_output(output: *mut AbiValue, handle: AbiHandle) -> u32 {
    unsafe {
        *output = AbiValue {
            tag: ValueTag::NativeHandle as u32,
            reserved: 0,
            payload: ValuePayload { handle },
        };
    }
    ErrorCode::Ok as u32
}

/// `make_viewer()` -> a live handle for viewer slot 0.
unsafe extern "C" fn inv_make_viewer(
    _a: *const AbiValue,
    _n: usize,
    output: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    handle_output(
        output,
        AbiHandle {
            extension_id: VIEWERS_EXT,
            type_id: VIEWER_TYPE,
            slot: 0,
            generation: LIVE_GEN,
        },
    )
}

/// `make_stale()` -> a handle whose generation no longer matches
/// the live slot (the resource was recycled).
unsafe extern "C" fn inv_make_stale(
    _a: *const AbiValue,
    _n: usize,
    output: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    handle_output(
        output,
        AbiHandle {
            extension_id: VIEWERS_EXT,
            type_id: VIEWER_TYPE,
            slot: 0,
            generation: LIVE_GEN + 1,
        },
    )
}

/// `others:make_token()` -> a handle minted by a DIFFERENT
/// extension (foreign to `viewers`).
unsafe extern "C" fn inv_make_token(
    _a: *const AbiValue,
    _n: usize,
    output: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    handle_output(
        output,
        AbiHandle {
            extension_id: OTHERS_EXT,
            type_id: VIEWER_TYPE,
            slot: 0,
            generation: LIVE_GEN,
        },
    )
}

/// `use_viewer(h)` -> the stored resource, or an err for a stale /
/// foreign / wrong-type handle. The provider owns the slot table
/// and validates the handle it is handed back.
unsafe extern "C" fn inv_use_viewer(
    a: *const AbiValue,
    n: usize,
    output: *mut AbiValue,
    err: *mut AbiErrorV1,
) -> u32 {
    unsafe {
        let valid = n == 1 && (*a).tag == ValueTag::NativeHandle as u32 && {
            let h = (*a).payload.handle;
            h.extension_id == VIEWERS_EXT && h.type_id == VIEWER_TYPE && h.generation == LIVE_GEN
        };
        if !valid {
            *err = AbiErrorV1 {
                code: ErrorCode::ExtensionFailure as u32,
                reserved: 0,
                message: AbiSlice {
                    data: BAD_HANDLE.as_ptr(),
                    len: BAD_HANDLE.len(),
                },
            };
            return ErrorCode::ExtensionFailure as u32;
        }
        *output = AbiValue {
            tag: ValueTag::F64 as u32,
            reserved: 0,
            payload: ValuePayload { float: STORED },
        };
    }
    ErrorCode::Ok as u32
}

fn fd(
    name: &'static [u8],
    invoke: unsafe extern "C" fn(*const AbiValue, usize, *mut AbiValue, *mut AbiErrorV1) -> u32,
) -> FunctionDescriptorV1 {
    FunctionDescriptorV1 {
        name: AbiSlice {
            data: name.as_ptr(),
            len: name.len(),
        },
        arity: 0,
        reserved: 0,
        invoke: Some(invoke),
    }
}

fn register(name: &[u8], functions: &[FunctionDescriptorV1]) {
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

fn register_providers() {
    let viewers = [
        fd(b"make_viewer", inv_make_viewer),
        fd(b"make_stale", inv_make_stale),
        // use_viewer has arity 1; patch the descriptor below.
        FunctionDescriptorV1 {
            name: AbiSlice {
                data: b"use_viewer".as_ptr(),
                len: 10,
            },
            arity: 1,
            reserved: 0,
            invoke: Some(inv_use_viewer),
        },
    ];
    register(b"viewers", &viewers);
    let others = [fd(b"make_token", inv_make_token)];
    register(b"others", &others);
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
fn a_handle_round_trips_through_a_single_call() {
    register_providers();
    let mut env = Environment::new();
    assert_eq!(
        scalar(&mut env, "viewers:use_viewer(viewers:make_viewer())"),
        STORED
    );
}

#[test]
fn a_handle_survives_an_mlpl_variable() {
    register_providers();
    let mut env = Environment::new();
    // The handle is stored in `h` and passed back on a later call.
    let src = "h = viewers:make_viewer()\nviewers:use_viewer(h)";
    assert_eq!(scalar(&mut env, src), STORED);
}

#[test]
fn a_stale_handle_is_a_clean_err() {
    register_providers();
    let mut env = Environment::new();
    assert_eq!(
        scalar(&mut env, "is_ok(viewers:use_viewer(viewers:make_stale()))"),
        0.0
    );
}

#[test]
fn a_foreign_extension_handle_is_a_clean_err() {
    register_providers();
    let mut env = Environment::new();
    assert_eq!(
        scalar(&mut env, "is_ok(viewers:use_viewer(others:make_token()))"),
        0.0
    );
}
