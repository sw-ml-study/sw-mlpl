//! Dense array and tensor types for MLPL.

mod dense;
mod display;
mod error;
mod indexing;
mod ops_binop;
mod ops_reshape;
mod ops_strides;
mod ops_transpose;
mod shape;

pub use dense::DenseArray;
pub use error::ArrayError;
pub use mlpl_core::LabeledShape;
pub use ops_strides::compute_strides;
pub use shape::Shape;
