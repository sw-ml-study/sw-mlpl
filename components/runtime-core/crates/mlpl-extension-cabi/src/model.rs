//! The canonical `#[repr(C)]` V1 extension boundary. Layout is
//! byte-for-byte identical to the demo-extensions provider ABI
//! (`mlpl-extension-abi/src/model.rs` + `error.rs`) so a real,
//! statically linked provider's `*const ExtensionDescriptorV1`
//! reinterprets here without copying. sw-mlpl owns registration,
//! so it publishes this layout; providers link against it (or an
//! identical local copy) and hand the host a descriptor pointer.
//!
//! No behavior lives here -- these are pure data shapes. All
//! fields are `pub` so an in-process provider can build them as
//! struct literals. Adapter logic is in `register` / `marshal`.

/// The only ABI version this host adapter accepts.
pub const ABI_VERSION_V1: u32 = 1;

/// A borrowed byte span (`data`, `len`). Not owned by the host;
/// valid only for the duration of the call that produced it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AbiSlice {
    pub data: *const u8,
    pub len: usize,
}

/// Discriminant for `AbiValue::tag`. Scalars 0..=5, `DenseArray`,
/// and `NativeHandle` all marshal across the boundary in both
/// directions.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueTag {
    Nil = 0,
    Bool = 1,
    I64 = 2,
    F64 = 3,
    Utf8 = 4,
    Bytes = 5,
    DenseArray = 6,
    NativeHandle = 7,
    Record = 8,
}

/// Opaque cross-boundary handle: a provider-issued resource
/// reference marshaled by value (tag `NativeHandle`). The host
/// treats all fields as opaque bits.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AbiHandle {
    pub extension_id: u64,
    pub type_id: u64,
    pub slot: u32,
    pub generation: u32,
}

/// Element dtype for `AbiArrayView` (arrays-handles follow-up).
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DTypeTag {
    U8 = 1,
    I64 = 2,
    F32 = 3,
    F64 = 4,
}

/// A dense array view across the boundary (arrays-handles
/// follow-up; not yet marshaled).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AbiArrayView {
    pub dtype: u32,
    pub rank: u32,
    pub data: AbiSlice,
    pub shape: *const usize,
    pub strides: *const isize,
}

/// One named field of a record view (`name`, `value`). The value is
/// itself an `AbiValue`, so records nest.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AbiField {
    pub name: AbiSlice,
    pub value: AbiValue,
}

/// A structured record across the boundary: a borrowed array of
/// `field_count` named fields. Provider-owned; the host copies it
/// during the call (the borrowed-span contract).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AbiRecordView {
    pub fields: *const AbiField,
    pub field_count: usize,
}

/// The payload union selected by `AbiValue::tag`.
#[repr(C)]
#[derive(Clone, Copy)]
pub union ValuePayload {
    pub boolean: u8,
    pub integer: i64,
    pub float: f64,
    pub slice: AbiSlice,
    pub array: *const AbiArrayView,
    pub handle: AbiHandle,
    pub record: *const AbiRecordView,
}

/// A tagged value crossing the boundary in either direction.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AbiValue {
    pub tag: u32,
    pub reserved: u32,
    pub payload: ValuePayload,
}

impl AbiValue {
    /// The `Nil` value -- used as the initial invoke output slot.
    #[must_use]
    pub const fn nil() -> Self {
        Self {
            tag: ValueTag::Nil as u32,
            reserved: 0,
            payload: ValuePayload { integer: 0 },
        }
    }
}

/// Error status codes an `InvokeFnV1` may report.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    Ok = 0,
    InvalidArgument = 1,
    ExtensionFailure = 2,
    Panic = 3,
}

/// Out-param an invoke fills when it returns a non-zero status.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AbiErrorV1 {
    pub code: u32,
    pub reserved: u32,
    pub message: AbiSlice,
}

impl AbiErrorV1 {
    /// A cleared error -- the initial invoke error slot.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            code: ErrorCode::Ok as u32,
            reserved: 0,
            message: AbiSlice {
                data: std::ptr::null(),
                len: 0,
            },
        }
    }
}

/// A single exported function. Returns an `ErrorCode` as `u32`;
/// on `Ok` it fills `output`, otherwise it fills `error`.
pub type InvokeFnV1 = unsafe extern "C" fn(
    arguments: *const AbiValue,
    argument_count: usize,
    output: *mut AbiValue,
    error: *mut AbiErrorV1,
) -> u32;

/// One entry in a descriptor's function table.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FunctionDescriptorV1 {
    pub name: AbiSlice,
    pub arity: u32,
    pub reserved: u32,
    pub invoke: Option<InvokeFnV1>,
}

/// The C entry point a provider exports (`sw_mlpl_extension_v1`).
pub type ExtensionEntryV1 = unsafe extern "C" fn() -> *const ExtensionDescriptorV1;

/// The top-level descriptor a provider hands the host adapter.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExtensionDescriptorV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub name: AbiSlice,
    pub version: AbiSlice,
    pub functions: *const FunctionDescriptorV1,
    pub function_count: usize,
    pub metadata: AbiSlice,
}
