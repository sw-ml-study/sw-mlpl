//! Canonical C-ABI V1 boundary + host adapter for native MLPL
//! extensions (demo-extensions upstream contract). A provider
//! exports a `#[repr(C)] ExtensionDescriptorV1` (see `model`);
//! [`register_c_extension`] validates it and registers each
//! function into the safe extension registry, so a colon-qualified
//! call (`namespace:function`) dispatches into the provider's C
//! code with scalars marshaled and panics contained.
//!
//! Scope: SCALAR values only (nil / bool / i64 / f64 / utf8 /
//! bytes). Dense arrays and native handles are a follow-up. Static
//! linking only -- dynamic (`dlopen`) loading is a separate saga.
//!
//! `lib.rs` is a facade: types live in `model`, marshaling in
//! `marshal`, and the adapter in `register`.

mod marshal;
mod model;
mod register;
mod validate;

pub use model::{
    ABI_VERSION_V1, AbiArrayView, AbiErrorV1, AbiHandle, AbiSlice, AbiValue, DTypeTag, ErrorCode,
    ExtensionDescriptorV1, ExtensionEntryV1, FunctionDescriptorV1, InvokeFnV1, ValuePayload,
    ValueTag,
};
pub use register::register_c_extension;
