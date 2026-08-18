//! The compiled value model (`CVal`) and its stdout / args IO.
//!
//! This file is a FACADE (`mod` + `use` only). Behaviour lives in
//! the named modules:
//! - `value` -- the `CVal` enum, `arr`/`field` accessors, `From`, `Display`.
//! - `ctor`  -- the `record` / `result` compound constructors.
//! - `io`    -- `write_stdout` / `args` / `arg` + `array_to_bytes`
//!   (the byte validator, shared with the `mlpl-rt-fsio` crate).
//! - `text`  -- `tokenize_bytes` / `decode_bytes` / `to_int`.

mod finish;
mod io;
mod proc;
mod stdin_chunk;
mod text;
mod value;

pub use finish::finish_program;
pub use io::{arg, array_to_bytes, cli_args, write_stdout};
pub use proc::{eprint, exit, print, read_stdin};
pub use stdin_chunk::read_stdin_chunk;
pub use text::{decode_bytes, disp, to_int, tokenize_bytes};
pub use value::CVal;
