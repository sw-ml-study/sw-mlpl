//! Structured returns: a `#[repr(C)]` provider RETURNS a record (an
//! `AbiRecordView` of named fields) so an event batch like
//! `poll_events` can hand back `{kind, x, y}` rows. A record whose
//! field values are themselves records is the "list of records" shape
//! MLPL reads by nested field access. Field data is provider-owned, so
//! the host copies it immediately (the borrowed-span contract).

use std::mem::size_of;
use std::ptr;

use mlpl_eval::{Environment, Value};
use mlpl_extension_cabi::{
    ABI_VERSION_V1, AbiErrorV1, AbiField, AbiRecordView, AbiSlice, AbiValue, ErrorCode,
    ExtensionDescriptorV1, FunctionDescriptorV1, ValuePayload, ValueTag, register_c_extension,
};

static VERSION: &[u8] = b"0.1.0";

fn slice(bytes: &'static [u8]) -> AbiSlice {
    AbiSlice {
        data: bytes.as_ptr(),
        len: bytes.len(),
    }
}

/// A record field carrying a scalar f64 value.
fn f64_field(name: &'static [u8], v: f64) -> AbiField {
    AbiField {
        name: slice(name),
        value: AbiValue {
            tag: ValueTag::F64 as u32,
            reserved: 0,
            payload: ValuePayload { float: v },
        },
    }
}

/// Leak an owned field array into an `AbiValue` record output (the
/// host copies it during the call, so the leak is harmless in-test).
fn record_value(fields: Vec<AbiField>) -> AbiValue {
    let fields: &'static [AbiField] = Box::leak(fields.into_boxed_slice());
    let view: &'static AbiRecordView = Box::leak(Box::new(AbiRecordView {
        fields: fields.as_ptr(),
        field_count: fields.len(),
    }));
    AbiValue {
        tag: ValueTag::Record as u32,
        reserved: 0,
        payload: ValuePayload { record: view },
    }
}

/// `poll_one()` -> a single event record `{kind:1, x:10, y:20}`.
unsafe extern "C" fn inv_poll_one(
    _a: *const AbiValue,
    _n: usize,
    output: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    let rec = record_value(vec![
        f64_field(b"kind", 1.0),
        f64_field(b"x", 10.0),
        f64_field(b"y", 20.0),
    ]);
    unsafe {
        *output = rec;
    }
    ErrorCode::Ok as u32
}

/// `poll_batch()` -> a record whose fields are themselves records: the
/// "list of records" event batch `{e0:{kind:1,x:5}, e1:{kind:2,x:7}}`.
unsafe extern "C" fn inv_poll_batch(
    _a: *const AbiValue,
    _n: usize,
    output: *mut AbiValue,
    _e: *mut AbiErrorV1,
) -> u32 {
    let e0 = record_value(vec![f64_field(b"kind", 1.0), f64_field(b"x", 5.0)]);
    let e1 = record_value(vec![f64_field(b"kind", 2.0), f64_field(b"x", 7.0)]);
    let batch = record_value(vec![
        AbiField {
            name: slice(b"e0"),
            value: e0,
        },
        AbiField {
            name: slice(b"e1"),
            value: e1,
        },
    ]);
    unsafe {
        *output = batch;
    }
    ErrorCode::Ok as u32
}

fn register_provider() {
    let functions = [
        FunctionDescriptorV1 {
            name: slice(b"poll_one"),
            arity: 0,
            reserved: 0,
            invoke: Some(inv_poll_one),
        },
        FunctionDescriptorV1 {
            name: slice(b"poll_batch"),
            arity: 0,
            reserved: 0,
            invoke: Some(inv_poll_batch),
        },
    ];
    let d = ExtensionDescriptorV1 {
        struct_size: size_of::<ExtensionDescriptorV1>() as u32,
        abi_version: ABI_VERSION_V1,
        name: slice(b"events"),
        version: slice(VERSION),
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
fn a_returned_record_exposes_its_fields_to_mlpl() {
    register_provider();
    let mut env = Environment::new();
    let src = "r = events:poll_one()\nr.kind + r.x + r.y";
    assert_eq!(scalar(&mut env, src), 31.0);
}

#[test]
fn a_returned_record_is_a_real_record_value() {
    register_provider();
    let mut env = Environment::new();
    match eval_value(&mut env, "events:poll_one()").unwrap() {
        Value::Record { fields } => {
            assert_eq!(fields.len(), 3);
            assert!(fields.contains_key("kind"));
        }
        other => panic!("expected Record, got {other:?}"),
    }
}

#[test]
fn a_list_of_records_reads_by_nested_field_access() {
    register_provider();
    let mut env = Environment::new();
    // Record-of-records: b.e0 and b.e1 are each an event record.
    let src = "b = events:poll_batch()\nb.e0.x + b.e1.x";
    assert_eq!(scalar(&mut env, src), 12.0);
}
