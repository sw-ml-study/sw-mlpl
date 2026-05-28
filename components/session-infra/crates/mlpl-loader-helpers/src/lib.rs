//! Pure-data helpers for the loader builtins. Saga 33 step
//! 017.
//!
//! - `sandbox::resolve_in_sandbox`: walk `..` components
//!   manually so escapes are caught without touching the
//!   filesystem.
//! - `csv::parse_csv`: tabular `String` -> rank-2
//!   `DenseArray`, header-row auto-detect.
//!
//! Both helpers are pure (no env, no `Value`, no filesystem
//! I/O), so this sub-crate has no env-traits dependency. The
//! env-dependent `eval_load_*` wrappers stay in mlpl-eval
//! until a `HasDataDir` capability trait lands.

pub mod csv;
pub mod error;
pub mod sandbox;

pub use csv::parse_csv;
pub use error::LoaderHelperError;
pub use sandbox::resolve_in_sandbox;
