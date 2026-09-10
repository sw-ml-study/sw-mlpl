//! Shared axis-selection types for MLPL.
//!
//! `AxisSpec` / `AxisNames` name a set of axes once so every axis-selecting
//! builtin -- in the interpreter and in the compile-to-Rust path -- resolves
//! them the same way and cannot drift. Depends only on `mlpl-array` (for
//! `DenseArray` and its label metadata), so it is reachable from both the
//! eval layer and the compiler without a dependency cycle.

mod axis_spec;

pub use axis_spec::{AxisError, AxisNames, AxisSpec};
