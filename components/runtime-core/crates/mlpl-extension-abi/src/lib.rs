//! Host-side value/error boundary for native MLPL extensions
//! (demo-extensions upstream contract A2). `ExtValue` is the V1
//! scalar set exchanged with an extension function; `ExtError`
//! carries a failure or a contained panic; `call_contained`
//! invokes a provider function with panics CAUGHT so an extension
//! cannot unwind through the host.
//!
//! This is the SAFE-RUST boundary used by statically linked
//! providers (the first slice). The `#[repr(C)]` wire layout for
//! DYNAMIC loading is a follow-up saga; both paths marshal through
//! this same value shape.

use std::panic::{AssertUnwindSafe, catch_unwind};

mod dtype;
mod dtype_codec;
mod handle;
pub use dtype::ExtDtype;
pub use handle::ExtHandle;

/// The V1 value set crossed at the extension boundary: the scalars
/// plus a dense, row-major, contiguous numeric array (rank 1..=8).
/// The host carries array elements as `f64` (its numeric type); the
/// C-ABI adapter narrows/widens to `dtype` on the wire.
#[derive(Clone, Debug, PartialEq)]
pub enum ExtValue {
    Nil,
    Bool(bool),
    I64(i64),
    F64(f64),
    Str(String),
    Bytes(Vec<u8>),
    Array {
        dtype: ExtDtype,
        shape: Vec<usize>,
        data: Vec<f64>,
    },
    /// An opaque provider-issued native handle (a persistent
    /// resource reference, e.g. a viewer object). Carried by
    /// value; only a provider return mints one.
    Handle(ExtHandle),
}

/// A recoverable extension failure, or a contained panic
/// (`panicked = true`). The host maps a domain failure to an MLPL
/// `err(...)` Result and a contained panic to a hard error.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtError {
    pub message: String,
    pub panicked: bool,
}

impl ExtError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            panicked: false,
        }
    }
}

/// An extension function: scalars in, one scalar or an `ExtError`
/// out. A boxed closure (not a bare `fn`) so a provider can
/// CAPTURE state -- e.g. the C-ABI adapter wraps each C invoke
/// trampoline. A plain `fn` coerces into it via `Arc::new`.
pub type ExtFn = std::sync::Arc<dyn Fn(&[ExtValue]) -> Result<ExtValue, ExtError> + Send + Sync>;

/// One exported function's descriptor: its name within the
/// namespace, arity, a bounded TOML signature doc (for `help`),
/// and the function.
#[derive(Clone)]
pub struct ExtFnDesc {
    pub name: String,
    pub arity: usize,
    pub signature_toml: String,
    pub func: ExtFn,
}

/// A validated V1 extension: its public namespace (`hello`), the
/// private native namespace (`_hello`), the `module.mlpl` facade
/// text (unused until the `use`/facade saga), and its functions.
#[derive(Clone)]
pub struct ExtensionDescriptorV1 {
    pub name: String,
    pub private_namespace: String,
    pub facade_mlpl: String,
    pub functions: Vec<ExtFnDesc>,
}

/// Invoke `f` with panics CONTAINED: a panic becomes an
/// `ExtError { panicked: true }` instead of unwinding into the
/// host. `Ok`/`Err` domain results pass through unchanged.
pub fn call_contained(f: &ExtFn, args: &[ExtValue]) -> Result<ExtValue, ExtError> {
    match catch_unwind(AssertUnwindSafe(|| f(args))) {
        Ok(result) => result,
        Err(_) => Err(ExtError {
            message: "extension function panicked".to_string(),
            panicked: true,
        }),
    }
}
