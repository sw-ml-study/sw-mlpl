//! Shape-manipulation extension traits (reshape, transpose) for
//! DenseArray. Body extracted from mlpl-array in saga 53.

mod reshape;
mod transpose;

pub use reshape::ReshapeExt;
pub use transpose::TransposeExt;

pub mod prelude {
    pub use super::{ReshapeExt, TransposeExt};
}
