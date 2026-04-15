//! `mlpl` facade crate (compile-to-rust saga).
//!
//! Single dependency for end users who want both the `mlpl!`
//! procedural macro and the runtime it emits calls into. Add
//! `mlpl = "..."` to `[dependencies]` and write:
//!
//! ```ignore
//! use mlpl::{mlpl, DenseArray};
//!
//! fn sum_of_first_n(n: usize) -> DenseArray {
//!     mlpl! { reduce_add(iota(#n)) }  // (future: macro args)
//! }
//! ```
//!
//! The hidden `__rt` re-export is the path the macro emits. It is
//! intentionally unstable -- users should reach for the types and
//! functions via the flat re-exports (`mlpl::DenseArray`,
//! `mlpl::iota`, ...) rather than `mlpl::__rt::...`.

pub use mlpl_macro::mlpl;

/// Flat re-exports of the runtime surface so
/// `use mlpl::{DenseArray, Shape, iota, ...}` works.
pub use mlpl_rt::{
    ArrayError, DenseArray, LabeledShape, Shape, array_lit, iota, rank, reduce_add,
    reduce_add_axis, reshape, shape, transpose,
};

#[doc(hidden)]
pub use mlpl_rt as __rt;
