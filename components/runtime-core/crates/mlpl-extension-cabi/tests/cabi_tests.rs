//! A synthetic in-test C provider (real `extern "C"` invoke
//! trampolines + `#[repr(C)]` descriptors) exercises the host
//! adapter end to end: register -> lookup -> invoke, plus the
//! err / arg-passing / validation paths. The registry is a
//! process-global, so each test uses a DISTINCT namespace.

use std::mem::size_of;
use std::ptr;

use mlpl_extension_abi::ExtValue;
use mlpl_extension_cabi::{
    ABI_VERSION_V1, AbiErrorV1, AbiRecordView, AbiSlice, AbiValue, ErrorCode,
    ExtensionDescriptorV1, FunctionDescriptorV1, ValuePayload, ValueTag, register_c_extension,
};
use mlpl_extension_registry::lookup;

static VERSION: &[u8] = b"0.1.0";
static FAIL_MSG: &[u8] = b"c provider failure";

unsafe extern "C" fn inv_answer(
    _args: *const AbiValue,
    _n: usize,
    output: *mut AbiValue,
    _err: *mut AbiErrorV1,
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
    _args: *const AbiValue,
    _n: usize,
    _output: *mut AbiValue,
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

unsafe extern "C" fn inv_incr(
    args: *const AbiValue,
    n: usize,
    output: *mut AbiValue,
    _err: *mut AbiErrorV1,
) -> u32 {
    let x = unsafe { (*args.add(0)).payload.integer };
    let _ = n;
    unsafe {
        *output = AbiValue {
            tag: ValueTag::I64 as u32,
            reserved: 0,
            payload: ValuePayload { integer: x + 1 },
        };
    }
    ErrorCode::Ok as u32
}

unsafe extern "C" fn inv_check_record(
    args: *const AbiValue,
    n: usize,
    output: *mut AbiValue,
    _err: *mut AbiErrorV1,
) -> u32 {
    assert_eq!(n, 1);
    let root = unsafe { &*args };
    assert_eq!(root.tag, ValueTag::Record as u32);
    let root_view: &AbiRecordView = unsafe { &*root.payload.record };
    assert_eq!(root_view.field_count, 2);
    let fields = unsafe { std::slice::from_raw_parts(root_view.fields, root_view.field_count) };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(fields[0].name.data, fields[0].name.len) },
        b"body"
    );
    assert_eq!(fields[0].value.tag, ValueTag::Bytes as u32);
    let body = unsafe { fields[0].value.payload.slice };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(body.data, body.len) },
        b"abc"
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(fields[1].name.data, fields[1].name.len) },
        b"meta"
    );
    assert_eq!(fields[1].value.tag, ValueTag::Record as u32);
    let meta = unsafe { &*fields[1].value.payload.record };
    let nested = unsafe { &*meta.fields };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(nested.name.data, nested.name.len) },
        b"status"
    );
    assert_eq!(unsafe { nested.value.payload.integer }, 201);
    unsafe {
        *output = AbiValue {
            tag: ValueTag::Bool as u32,
            reserved: 0,
            payload: ValuePayload { boolean: 1 },
        }
    };
    ErrorCode::Ok as u32
}

fn text(bytes: &[u8]) -> AbiSlice {
    AbiSlice {
        data: bytes.as_ptr(),
        len: bytes.len(),
    }
}

fn func(
    name: &[u8],
    arity: u32,
    invoke: unsafe extern "C" fn(*const AbiValue, usize, *mut AbiValue, *mut AbiErrorV1) -> u32,
) -> FunctionDescriptorV1 {
    FunctionDescriptorV1 {
        name: text(name),
        arity,
        reserved: 0,
        invoke: Some(invoke),
    }
}

fn desc(name: &[u8], functions: &[FunctionDescriptorV1]) -> ExtensionDescriptorV1 {
    ExtensionDescriptorV1 {
        struct_size: size_of::<ExtensionDescriptorV1>() as u32,
        abi_version: ABI_VERSION_V1,
        name: text(name),
        version: text(VERSION),
        functions: functions.as_ptr(),
        function_count: functions.len(),
        metadata: AbiSlice {
            data: ptr::null(),
            len: 0,
        },
    }
}

#[test]
fn c_provider_registers_and_answers_42() {
    let functions = [func(b"answer", 0, inv_answer)];
    let d = desc(b"ctest_answer", &functions);
    unsafe { register_c_extension(&d) }.expect("register");
    let f = lookup("ctest_answer:answer").expect("registered");
    assert_eq!(f.arity, 0);
    assert_eq!((f.func)(&[]), Ok(ExtValue::I64(42)));
}

#[test]
fn c_provider_domain_failure_maps_to_ext_error() {
    let functions = [func(b"fail", 0, inv_fail)];
    let d = desc(b"ctest_fail", &functions);
    unsafe { register_c_extension(&d) }.expect("register");
    let f = lookup("ctest_fail:fail").expect("registered");
    let e = (f.func)(&[]).unwrap_err();
    assert_eq!(e.message, "c provider failure");
    assert!(!e.panicked);
}

#[test]
fn c_provider_receives_marshaled_arguments() {
    let functions = [func(b"incr", 1, inv_incr)];
    let d = desc(b"ctest_incr", &functions);
    unsafe { register_c_extension(&d) }.expect("register");
    let f = lookup("ctest_incr:incr").expect("registered");
    assert_eq!((f.func)(&[ExtValue::I64(41)]), Ok(ExtValue::I64(42)));
}

#[test]
fn c_provider_receives_nested_record_and_bytes_with_live_backing() {
    let functions = [func(b"check", 1, inv_check_record)];
    let d = desc(b"ctest_record_in", &functions);
    unsafe { register_c_extension(&d) }.expect("register");
    let f = lookup("ctest_record_in:check").expect("registered");
    let input = ExtValue::Record(vec![
        ("body".into(), ExtValue::Bytes(b"abc".to_vec())),
        (
            "meta".into(),
            ExtValue::Record(vec![("status".into(), ExtValue::I64(201))]),
        ),
    ]);
    assert_eq!((f.func)(&[input]), Ok(ExtValue::Bool(true)));
}

#[test]
fn oversized_outbound_record_is_rejected_before_provider_invocation() {
    let functions = [func(b"check", 1, inv_check_record)];
    let d = desc(b"ctest_record_cap", &functions);
    unsafe { register_c_extension(&d) }.expect("register");
    let f = lookup("ctest_record_cap:check").expect("registered");
    let fields = (0..1025)
        .map(|i| (format!("f{i}"), ExtValue::Nil))
        .collect();
    let error = (f.func)(&[ExtValue::Record(fields)]).unwrap_err();
    assert!(
        error.message.contains("1025 fields (max 1024)"),
        "{error:?}"
    );
}

#[test]
fn excessive_outbound_record_nesting_is_rejected() {
    let functions = [func(b"check", 1, inv_check_record)];
    let d = desc(b"ctest_record_depth", &functions);
    unsafe { register_c_extension(&d) }.expect("register");
    let f = lookup("ctest_record_depth:check").expect("registered");
    let mut value = ExtValue::Nil;
    for _ in 0..65 {
        value = ExtValue::Record(vec![("next".into(), value)]);
    }
    let error = (f.func)(&[value]).unwrap_err();
    assert!(
        error.message.contains("nesting exceeds boundary cap 64"),
        "{error:?}"
    );
}

#[test]
fn null_descriptor_is_rejected() {
    let e = unsafe { register_c_extension(ptr::null()) }.unwrap_err();
    assert!(e.contains("null"), "{e}");
}

#[test]
fn wrong_struct_size_is_rejected() {
    let functions = [func(b"answer", 0, inv_answer)];
    let mut d = desc(b"ctest_size", &functions);
    d.struct_size = 7;
    let e = unsafe { register_c_extension(&d) }.unwrap_err();
    assert!(e.contains("struct_size"), "{e}");
}

#[test]
fn wrong_abi_version_is_rejected() {
    let functions = [func(b"answer", 0, inv_answer)];
    let mut d = desc(b"ctest_abi", &functions);
    d.abi_version = 99;
    let e = unsafe { register_c_extension(&d) }.unwrap_err();
    assert!(e.contains("abi_version"), "{e}");
}

#[test]
fn null_invoke_is_rejected() {
    let functions = [FunctionDescriptorV1 {
        name: text(b"answer"),
        arity: 0,
        reserved: 0,
        invoke: None,
    }];
    let d = desc(b"ctest_noinvoke", &functions);
    let e = unsafe { register_c_extension(&d) }.unwrap_err();
    assert!(e.contains("null invoke"), "{e}");
}

#[test]
fn duplicate_function_name_is_rejected() {
    let functions = [
        func(b"answer", 0, inv_answer),
        func(b"answer", 0, inv_answer),
    ];
    let d = desc(b"ctest_dup", &functions);
    let e = unsafe { register_c_extension(&d) }.unwrap_err();
    assert!(e.contains("duplicate"), "{e}");
}

/// The safe `ExtDtype::wire_tag` must mirror the C `DTypeTag` codes;
/// they live in separate crates and would silently drift otherwise.
#[test]
fn wire_tag_mirrors_the_dtype_tag_codes() {
    use mlpl_extension_abi::ExtDtype;
    use mlpl_extension_cabi::DTypeTag;
    assert_eq!(ExtDtype::U8.wire_tag(), DTypeTag::U8 as u32);
    assert_eq!(ExtDtype::I64.wire_tag(), DTypeTag::I64 as u32);
    assert_eq!(ExtDtype::F32.wire_tag(), DTypeTag::F32 as u32);
    assert_eq!(ExtDtype::F64.wire_tag(), DTypeTag::F64 as u32);
}
