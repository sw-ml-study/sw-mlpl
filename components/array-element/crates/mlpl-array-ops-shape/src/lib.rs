//! Shape-manipulation extension traits (reshape, transpose,
//! transpose_axes) for DenseArray. Body extracted from
//! mlpl-array in saga 53.

mod reshape;
mod transpose;
mod transpose_axes;

pub use reshape::ReshapeExt;
pub use transpose::TransposeExt;
pub use transpose_axes::TransposeAxesExt;

pub mod prelude {
    pub use super::{ReshapeExt, TransposeAxesExt, TransposeExt};
}
