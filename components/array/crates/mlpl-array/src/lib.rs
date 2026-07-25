//! Dense array and tensor types for MLPL.

mod box_display;
mod dense;
mod display;
mod error;
mod indexing;
mod shape;

pub use box_display::box_display;
pub use dense::DenseArray;
pub use error::ArrayError;
pub use mlpl_core::LabeledShape;
pub use shape::Shape;
pub use shape::compute_strides;
