//! A cdylib native-extension fixture for dynamic loading (B3). It
//! exports the V1 entry point `sw_mlpl_extension_v1`, which returns a
//! descriptor with one arity-0 function, `answer` (-> 42.0). A loader
//! `dlopen`s this library, resolves the entry, registers the
//! descriptor through the same C ABI the static path uses, and invokes
//! `testext:answer`.

use std::mem::size_of;

use mlpl_extension_cabi::{
    ABI_VERSION_V1, AbiErrorV1, AbiSlice, AbiValue, ErrorCode, ExtensionDescriptorV1,
    FunctionDescriptorV1, ValuePayload, ValueTag,
};

/// The value `testext:answer` returns -- the single source of truth
/// the dlopen test asserts against.
pub const ANSWER: f64 = 42.0;

static NAME: &[u8] = b"testext";
static VER: &[u8] = b"0.1.0";
static FNAME: &[u8] = b"answer";

/// A borrowed span over a `'static` byte string.
fn slice(bytes: &'static [u8]) -> AbiSlice {
    AbiSlice {
        data: bytes.as_ptr(),
        len: bytes.len(),
    }
}

unsafe extern "C" fn inv_answer(
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

/// The V1 C entry point (`sw_mlpl_extension_v1`) a provider exports.
/// Returns a leaked descriptor -- it lives for the process, and the
/// host copies it into host-owned memory at register time.
#[unsafe(no_mangle)]
pub extern "C" fn sw_mlpl_extension_v1() -> *const ExtensionDescriptorV1 {
    let functions = Box::leak(Box::new([FunctionDescriptorV1 {
        name: slice(FNAME),
        arity: 0,
        reserved: 0,
        invoke: Some(inv_answer),
    }]));
    Box::leak(Box::new(ExtensionDescriptorV1 {
        struct_size: size_of::<ExtensionDescriptorV1>() as u32,
        abi_version: ABI_VERSION_V1,
        name: slice(NAME),
        version: slice(VER),
        functions: functions.as_ptr(),
        function_count: functions.len(),
        metadata: AbiSlice {
            data: std::ptr::null(),
            len: 0,
        },
    }))
}
