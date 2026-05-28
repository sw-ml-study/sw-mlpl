//! Reduction extension traits (reduce_axis, argmax_axis) for
//! DenseArray. Body extracted from mlpl-array in saga 53.

mod argmax;
mod axis;
mod reduce;

pub use argmax::ArgmaxAxisExt;
pub use reduce::ReduceAxisExt;

pub mod prelude {
    pub use super::{ArgmaxAxisExt, ReduceAxisExt};
}
