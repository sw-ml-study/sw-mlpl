//! Image decode + resize for mlpl-eval. Extracted in saga 71 as
//! the first god-crate-decomposition step. Defines its own
//! `ImageError`; mlpl-eval converts via `From<ImageError> for
//! EvalError` at the boundary.

#[cfg(feature = "image-io")]
mod io;
#[cfg(feature = "image-io")]
mod pixels;

#[cfg(feature = "image-io")]
pub use io::{decode_and_resize, decode_and_resize_u8};

/// Image decode / resize errors. Always a single message string;
/// mlpl-eval wraps these into `EvalError::Unsupported`.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageError(pub String);

impl ImageError {
    #[cfg(feature = "image-io")]
    pub(crate) fn new(msg: String) -> Self {
        Self(msg)
    }
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ImageError {}
