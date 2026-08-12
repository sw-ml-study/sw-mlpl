//! The compiled value model (`CVal`) and its stdout / args IO.
//!
//! This file is a FACADE (`mod` + `use` only). Behaviour lives in
//! the named modules:
//! - `value` -- the `CVal` enum, `arr`/`field` accessors, `From`, `Display`.
//! - `ctor`  -- the `record` / `result` compound constructors.
//! - `io`    -- `write_stdout` / `args` / `arg` + the bytes validator.
//! - `fsio`  -- sandboxed file reads (`read_bytes` / `file_size`).

mod ctor;
mod fsio;
mod io;
mod value;

pub use fsio::{file_size, read_bytes, read_bytes_range};
pub use io::{arg, cli_args, write_stdout};
pub use value::CVal;
