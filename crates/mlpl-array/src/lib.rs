//! Dense array and tensor types for MLPL.

mod dense;
mod display;
mod error;
mod indexing;
mod ops;
mod shape;

pub use dense::DenseArray;
pub use error::ArrayError;
pub use mlpl_core::LabeledShape;
pub use shape::Shape;
