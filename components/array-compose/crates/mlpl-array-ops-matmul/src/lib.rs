//! Matrix-multiplication + dot-product extension traits for
//! DenseArray. Body extracted from mlpl-array in saga 53.

mod dot;
mod labels;
mod matmul;

pub use dot::DotExt;
pub use matmul::{MatmulExt, check_matmul_compat};

pub mod prelude {
    pub use super::{DotExt, MatmulExt};
}
