//! A cdylib native-extension fixture for dynamic loading (B3). It
//! exports the V1 entry point `sw_mlpl_extension_v1`, returning a
//! descriptor for namespace `testext` with `answer` (-> 42.0) and
//! `stats(array)` (-> a `{sum, count}` record). A loader `dlopen`s this
//! library, resolves the entry, registers the descriptor through the
//! same C ABI the static path uses, and dispatches its functions.
//!
//! The exported functions and helpers live in `provider`; `lib.rs` is
//! the entry-point facade.

mod provider;

use std::mem::size_of;

use mlpl_extension_cabi::{
    ABI_VERSION_V1, AbiSlice, ExtensionDescriptorV1, FunctionDescriptorV1, InvokeFnV1,
};

use provider::{inv_answer, inv_sum, slice};

/// The value `testext:answer` returns -- the single source of truth
/// the dlopen tests assert against.
pub const ANSWER: f64 = 42.0;

static NAME: &[u8] = b"testext";
static VER: &[u8] = b"0.1.0";

/// One exported function's descriptor.
fn func(name: &'static [u8], arity: u32, invoke: InvokeFnV1) -> FunctionDescriptorV1 {
    FunctionDescriptorV1 {
        name: slice(name),
        arity,
        reserved: 0,
        invoke: Some(invoke),
    }
}

/// The V1 C entry point (`sw_mlpl_extension_v1`) a provider exports.
/// Returns a leaked descriptor -- it lives for the process, and the
/// host copies it into host-owned memory at register time.
#[unsafe(no_mangle)]
pub extern "C" fn sw_mlpl_extension_v1() -> *const ExtensionDescriptorV1 {
    let functions = Box::leak(Box::new([
        func(b"answer", 0, inv_answer),
        func(b"sum", 1, inv_sum),
    ]));
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
