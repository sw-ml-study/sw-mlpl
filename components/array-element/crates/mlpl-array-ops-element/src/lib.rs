//! Element-wise extension trait (apply_binop) for DenseArray.
//! Body extracted from mlpl-array in saga 53.

mod binop;
mod merge_labels;

pub use binop::ApplyBinopExt;

pub mod prelude {
    pub use super::ApplyBinopExt;
}
