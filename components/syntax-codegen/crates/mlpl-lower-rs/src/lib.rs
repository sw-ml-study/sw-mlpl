//! Lower MLPL AST to Rust `TokenStream` (compile-to-rust saga).
//!
//! Shared codegen for the `mlpl!` proc macro and the `mlpl-build`
//! subcommand. Walks an `mlpl_parser::Expr` AST and emits Rust code
//! that, when compiled, produces the same result as the interpreter.
//!
//! This file is a FACADE (`mod` + `use` only). Behaviour lives in
//! the named modules:
//! - `model`   -- `LowerError`, `LowerConfig`, `Ctx` (+ impls).
//! - `lower`   -- program / statement / expression lowering.
//! - `fncall`  -- builtin + user-fn call lowering, static labels.
//! - `cval_lower` -- the compiled value model (`CVal`) wrapping.
//! - `fndef_lower` -- `def u:` user functions + shared body lowering.
//! - `control_lower` -- control flow (`if`; `while`/`for` to come).
//!
//! Every supported MLPL expression lowers to a Rust expression of
//! type `<rt>::DenseArray` (string / IO / result positions wrap as
//! `<rt>::CVal`); `<rt>` is `::mlpl_rt` by default or a facade path
//! (`::mlpl::__rt`) via `LowerConfig`. Unsupported constructs return
//! `LowerError::Unsupported`.

mod control_lower;
mod cval_lower;
mod fncall;
mod fndef_lower;
mod lower;
mod model;

pub use fncall::supported_builtin_names;
pub use lower::{lower, lower_with_config};
pub use model::{LowerConfig, LowerError};

pub(crate) use cval_lower::{lower_cval, lower_field, lower_record};
pub(crate) use fncall::{labels_of, lower_fncall};
pub(crate) use lower::lower_expr;
pub(crate) use model::Ctx;
