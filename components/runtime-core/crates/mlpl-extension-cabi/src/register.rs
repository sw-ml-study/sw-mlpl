//! The host adapter: accept a provider's `#[repr(C)]`
//! `ExtensionDescriptorV1` pointer, validate it, wrap each C
//! invoke trampoline in a capturing closure that marshals scalars
//! and contains panics, and register the result into the safe
//! extension registry. Static linking only (no dlopen); the
//! provider keeps its descriptor + code alive for the process.

use std::collections::BTreeSet;
use std::mem::size_of;

use mlpl_extension_abi::{ExtError, ExtFn, ExtFnDesc, ExtValue};

use crate::model::{
    ABI_VERSION_V1, AbiErrorV1, AbiValue, ErrorCode, ExtensionDescriptorV1, InvokeFnV1,
};
use crate::validate;

/// Largest function table the adapter will accept.
const MAX_FUNCTIONS: usize = 1024;

/// Register a C-ABI V1 extension into the process registry.
///
/// # Safety
/// `descriptor` must be null or point to a valid, live
/// `ExtensionDescriptorV1` (and its function table / slices) for
/// the duration of this call, matching the V1 layout in `model`.
pub unsafe fn register_c_extension(descriptor: *const ExtensionDescriptorV1) -> Result<(), String> {
    if descriptor.is_null() {
        return Err("null extension descriptor".to_string());
    }
    let d = unsafe { &*descriptor };
    if d.struct_size as usize != size_of::<ExtensionDescriptorV1>() {
        return Err(format!(
            "struct_size {} does not match V1 layout",
            d.struct_size
        ));
    }
    if d.abi_version != ABI_VERSION_V1 {
        return Err(format!("unsupported abi_version {}", d.abi_version));
    }
    let name = validate::read_required_text("name", d.name)?;
    let _version = validate::read_required_text("version", d.version)?;
    let functions = build_functions(d)?;
    let safe = mlpl_extension_abi::ExtensionDescriptorV1 {
        private_namespace: name.clone(),
        name,
        facade_mlpl: String::new(),
        functions,
    };
    mlpl_extension_registry::register(&safe).map_err(|e| format!("{e:?}"))
}

/// Validate + convert the C function table into safe descriptors.
fn build_functions(d: &ExtensionDescriptorV1) -> Result<Vec<ExtFnDesc>, String> {
    if d.function_count > MAX_FUNCTIONS {
        return Err(format!("function_count {} exceeds cap", d.function_count));
    }
    if d.function_count > 0 && d.functions.is_null() {
        return Err("non-zero function_count with null functions pointer".to_string());
    }
    let raw = match d.function_count {
        0 => &[][..],
        n => unsafe { std::slice::from_raw_parts(d.functions, n) },
    };
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(raw.len());
    for fd in raw {
        let (name, invoke) = validate::convert_function(fd, &mut seen)?;
        out.push(ExtFnDesc {
            name,
            arity: fd.arity as usize,
            signature_toml: String::new(),
            func: invoke_closure(invoke),
        });
    }
    Ok(out)
}

/// Wrap a C invoke trampoline in a boxed closure marshaling
/// scalars in/out and mapping a non-zero status to an `ExtError`.
/// (Host-side `call_contained` also wraps this in `catch_unwind`.)
fn invoke_closure(invoke: InvokeFnV1) -> ExtFn {
    std::sync::Arc::new(move |args: &[ExtValue]| -> Result<ExtValue, ExtError> {
        // Owns the array backing buffers for the duration of the call.
        let abi_args = crate::marshal_array::marshal_args(args).map_err(ExtError::new)?;
        let mut output = AbiValue::nil();
        let mut error = AbiErrorV1::none();
        let (args_ptr, count) = if abi_args.values.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (abi_args.values.as_ptr(), abi_args.values.len())
        };
        let status = unsafe { invoke(args_ptr, count, &mut output, &mut error) };
        if status == ErrorCode::Ok as u32 {
            crate::marshal_array_out::abi_to_ext(&output)
        } else {
            Err(crate::marshal_array_out::abi_error_to_ext(&error, status))
        }
    })
}
